#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <chrono>
#include <macros.hpp>
#include <gpu_btree.h>
#include <memory_reclaimer.hpp>
namespace cg = cooperative_groups;
struct BlockGroup {
    __device__ unsigned thread_rank() const { return threadIdx.x; }
    __device__ unsigned size() const { return blockDim.x; }
    __device__ void sync() const { __syncthreads(); }
    __device__ bool all(bool value) const { return __syncthreads_and(value); }
};
constexpr int READERS=8, ROOTS=16, THREADS=128;
using Reclaimer=smr::DEBR_device<384>;
struct Step { uint32_t ok,used,epoch,pinned,generation,root,slot; uint64_t ns; };
struct Reader { uint32_t slot,generation,root,ack_before,ack_after; uint64_t enter_ns,exit_ns,checksum; };
struct State {
    uint32_t *data,*generation,*status,*refs,*heads,*c;
    Step* steps; Reader* readers; uint64_t* start;
    int slots,words,attempts;
};
enum Counter { ERR,USED,PEAK,FREED,READY,START,DONE,ACTIVE,ACK,FAIL,FIRSTFAIL,ACK_PINNED,COUNT };
__device__ uint32_t load(uint32_t* p) {return cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);}
__device__ void store(uint32_t* p,uint32_t x) {cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*p).store(x,cuda::memory_order_release);}
__host__ __device__ uint32_t payload(uint32_t gen,uint32_t root,uint32_t word) {
    uint32_t x=(gen+1)*0x9e3779b9u ^ (root+1)*0x85ebca6bu ^ (word+1)*0xc2b2ae35u;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return (x&0x007fffffu)|0x3f000000u;
}
__device__ uint64_t now() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ void pause_us(uint32_t us) {uint64_t t=now();while(now()-t < uint64_t(us)*1000) __nanosleep(100);}
struct Allocator {
    using size_type=uint32_t; State s;
    __device__ void deallocate(int&,uint32_t address,uint32_t) {
        if(address>=uint32_t(s.slots)){atomicAdd(s.c+ERR,1);return;}
        if(load(s.refs+address)!=0 || atomicCAS(s.status+address,2,0)!=2) atomicAdd(s.c+ERR,1);
        atomicSub(s.c+USED,1);atomicAdd(s.c+FREED,1);
    }
};
__global__ void init(State s) {
    for(int i=threadIdx.x+blockIdx.x*blockDim.x;i<ROOTS*s.words;i+=blockDim.x*gridDim.x)
        s.data[i]=payload(0,i/s.words,i%s.words);
    if(blockIdx.x==0&&threadIdx.x==0){
        for(int i=0;i<ROOTS;++i){s.heads[i]=i;s.status[i]=1;}
        s.c[USED]=s.c[PEAK]=ROOTS;s.c[FIRSTFAIL]=0xffffffffu;
    }
}
__global__ void run(State s,smr::DEBR_host host,int late,int pre_us,int hold_us) {
    BlockGroup tile;
    __shared__ uint32_t bags[Reclaimer::compute_shmem_requirements()];
    auto rec=Reclaimer{host,bags,READERS+1,tile};
    Allocator allocator{s};int ctx=0;
    __shared__ uint32_t slot,gen,old;
    __shared__ uint64_t checksums[THREADS];
    auto grid=cg::this_grid();
    if(blockIdx.x==READERS && threadIdx.x==0)*s.start=now();
    grid.sync();
    if(blockIdx.x<READERS){
        int id=blockIdx.x;
        if(!late){
            rec.leave_qstate(tile,id,allocator,ctx);
            if(threadIdx.x==0){slot=load(s.heads+id);gen=s.generation[slot];atomicAdd(s.refs+slot,1);atomicAdd(s.c+ACTIVE,1);
                s.readers[id]={slot,gen,uint32_t(id),load(s.c+ACK),0,now()-*s.start,0,0};}
            tile.sync();
        }
        if(threadIdx.x==0)atomicAdd(s.c+READY,1);
        while(!load(s.c+START))__nanosleep(100);
        if(threadIdx.x==0)pause_us(pre_us);
        tile.sync();
        if(late){
            rec.leave_qstate(tile,id,allocator,ctx);
            if(threadIdx.x==0){slot=load(s.heads+id);gen=s.generation[slot];atomicAdd(s.refs+slot,1);atomicAdd(s.c+ACTIVE,1);
                s.readers[id]={slot,gen,uint32_t(id),load(s.c+ACK),0,now()-*s.start,0,0};}
            tile.sync();
        }
        if(threadIdx.x==0)pause_us(hold_us);
        tile.sync();
        uint64_t sum=0;
        for(int repeat=0;repeat<4;++repeat)
            for(int word=threadIdx.x;word<s.words;word+=THREADS){
                uint32_t value=((volatile uint32_t*)s.data)[uint64_t(slot)*s.words+word];
                if(value!=payload(gen,id,word))atomicAdd(s.c+ERR,1);
                sum+=value;
            }
        checksums[threadIdx.x]=sum;tile.sync();
        if(threadIdx.x==0){
            sum=0;for(int i=0;i<THREADS;++i)sum+=checksums[i];
            s.readers[id].checksum=sum;s.readers[id].ack_after=load(s.c+ACK);s.readers[id].exit_ns=now()-*s.start;
            atomicSub(s.refs+slot,1);atomicSub(s.c+ACTIVE,1);
        }
        rec.enter_qstate(tile,id);
        if(threadIdx.x==0)atomicAdd(s.c+DONE,1);
    }else{
        while(load(s.c+READY)!=READERS)__nanosleep(100);
        if(threadIdx.x==0)store(s.c+START,1);
        tile.sync();
        for(int step=0;step<s.attempts;++step){
            rec.leave_qstate(tile,READERS,allocator,ctx);
            if(threadIdx.x==0){
                slot=0xffffffffu;gen=step+1;
                for(int offset=0;offset<s.slots;++offset){
                    uint32_t candidate=(step+offset)%s.slots;
                    if(atomicCAS(s.status+candidate,0,1)==0){slot=candidate;break;}
                }
                if(slot!=0xffffffffu){uint32_t used=atomicAdd(s.c+USED,1)+1;atomicMax(s.c+PEAK,used);s.generation[slot]=gen;}
                else{atomicAdd(s.c+FAIL,1);atomicCAS(s.c+FIRSTFAIL,0xffffffffu,step);}
            }
            tile.sync();
            if(slot!=0xffffffffu){
                for(int word=threadIdx.x;word<s.words;word+=THREADS)
                    s.data[uint64_t(slot)*s.words+word]=payload(gen,step%ROOTS,word);
                tile.sync();
                if(threadIdx.x==0){old=load(s.heads+step%ROOTS);store(s.heads+step%ROOTS,slot);store(s.status+old,2);
                    atomicAdd(s.c+ACK,1);if(load(s.c+ACTIVE))atomicAdd(s.c+ACK_PINNED,1);}
                tile.sync();rec.retire(old,tile,allocator,ctx);
            }else{if(threadIdx.x==0)pause_us(2);tile.sync();}
            rec.enter_qstate(tile,READERS);
            if(threadIdx.x==0)s.steps[step]={uint32_t(slot!=0xffffffffu),load(s.c+USED),load(host.d_current_epoch_),load(s.c+ACTIVE),gen,uint32_t(step%ROOTS),slot,now()-*s.start};
        }
        while(load(s.c+DONE)!=READERS)__nanosleep(100);
        for(int i=0;i<16;++i){rec.leave_qstate(tile,READERS,allocator,ctx);rec.enter_qstate(tile,READERS);}
    }
    grid.sync();
}
template<class T>T* alloc(size_t n){T* p;cuda_try(cudaMalloc(&p,n*sizeof(T)));cuda_try(cudaMemset(p,0,n*sizeof(T)));return p;}
void trial(int dim,int slots,int pre,int hold,char scope,int attempt,int observation) {
    State s{};s.slots=slots;s.words=64*dim;s.attempts=attempt;
    s.data=alloc<uint32_t>(size_t(slots)*s.words);s.generation=alloc<uint32_t>(slots);s.status=alloc<uint32_t>(slots);s.refs=alloc<uint32_t>(slots);
    s.heads=alloc<uint32_t>(ROOTS);s.c=alloc<uint32_t>(COUNT);s.steps=alloc<Step>(attempt);s.readers=alloc<Reader>(READERS);s.start=alloc<uint64_t>(1);
    // Each block's each overflow bag can hold every attempted retirement.
    uint32_t external=(READERS+1)*3*(attempt+128);
    smr::DEBR_host reclaimer(READERS+1,external);
    init<<<16,128>>>(s);cuda_try(cudaDeviceSynchronize());
    int late=scope=='L';void* args[]={&s,&reclaimer,&late,&pre,&hold};
    auto begin=std::chrono::steady_clock::now();
    cuda_try(cudaLaunchCooperativeKernel((void*)run,READERS+1,THREADS,args));cuda_try(cudaDeviceSynchronize());
    double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
    std::vector<uint32_t> c(COUNT),heads(ROOTS),status(slots),generation(slots),refs(slots),data(size_t(slots)*s.words);
    std::vector<Step> steps(attempt);std::vector<Reader> readers(READERS);
    auto copy=[](auto& dst,auto* src){cuda_try(cudaMemcpy(dst.data(),src,dst.size()*sizeof(*src),cudaMemcpyDeviceToHost));};
    copy(c,s.c);copy(heads,s.heads);copy(status,s.status);copy(generation,s.generation);copy(refs,s.refs);copy(data,s.data);copy(steps,s.steps);copy(readers,s.readers);
    auto require=[](bool pass,const char* why){if(!pass)throw std::runtime_error(why);};
    require(c[ERR]==0,"device invariant failed");require(c[USED]==ROOTS,"post-reader drain lost capacity");
    require(c[ACK]+c[FAIL]==uint32_t(attempt)&&c[FREED]==c[ACK],"ack/free conservation failed");
    std::vector<bool> owned(slots,false);std::vector<uint32_t> expected_gen(ROOTS,0);
    for(int i=0;i<attempt;++i)if(steps[i].ok)expected_gen[i%ROOTS]=i+1;
    for(int root=0;root<ROOTS;++root){uint32_t p=heads[root];require(p<uint32_t(slots)&&!owned[p],"root ownership");owned[p]=true;
        require(generation[p]==expected_gen[root],"lost acknowledged update");
        for(int w=0;w<s.words;++w)require(data[size_t(p)*s.words+w]==payload(expected_gen[root],root,w),"final payload mismatch");}
    for(int i=0;i<slots;++i)require(refs[i]==0&&status[i]==uint32_t(owned[i]),"slot/refs conservation");
    for(int i=0;i<READERS;++i){uint64_t expected=0;for(int w=0;w<s.words;++w)expected+=payload(readers[i].generation,i,w);expected*=4;
        require(readers[i].checksum==expected&&readers[i].exit_ns>=readers[i].enter_ns,"reader oracle mismatch");}
    printf("{\"scope\":\"%c\",\"dim\":%d,\"slots\":%d,\"pre_us\":%d,\"hold_us\":%d,\"observation\":%d,\"attempts\":%d,\"ack\":%u,\"failed\":%u,\"first_fail\":%lld,\"peak_slots\":%u,\"freed\":%u,\"final_live\":%u,\"ack_while_pinned\":%u,\"kernel_wall_us\":%.3f,\"layout_bytes\":%llu,\"debr_allocated_bytes\":%u,\"oracle\":true,\"readers\":[",scope,dim,slots,pre,hold,observation,attempt,c[ACK],c[FAIL],c[FIRSTFAIL]==0xffffffffu?-1LL:(long long)c[FIRSTFAIL],c[PEAK],c[FREED],c[USED],c[ACK_PINNED],wall,(unsigned long long)(size_t(slots)*(s.words+3)*4+ROOTS*4),(READERS+1+external+1)*4);
    for(int i=0;i<READERS;++i){auto r=readers[i];printf("%s[%u,%u,%u,%u,%u,%llu,%llu,%llu]",i?",":"",r.root,r.slot,r.generation,r.ack_before,r.ack_after,(unsigned long long)r.enter_ns,(unsigned long long)r.exit_ns,(unsigned long long)r.checksum);}
    printf("],\"steps\":[");for(int i=0;i<attempt;++i){auto t=steps[i];printf("%s[%u,%u,%u,%u,%u,%u,%u,%llu]",i?",":"",t.ok,t.used,t.epoch,t.pinned,t.generation,t.root,t.slot,(unsigned long long)t.ns);}puts("]}");fflush(stdout);
    cudaFree(s.data);cudaFree(s.generation);cudaFree(s.status);cudaFree(s.refs);cudaFree(s.heads);cudaFree(s.c);cudaFree(s.steps);cudaFree(s.readers);cudaFree(s.start);
}
int main(int argc,char**argv){try{
    if(argc!=9)throw std::runtime_error("usage: probe D slots pre_us hold_us EL|LE trials attempts warmups");
    int dim=std::stoi(argv[1]),slots=std::stoi(argv[2]),pre=std::stoi(argv[3]),hold=std::stoi(argv[4]),trials=std::stoi(argv[6]),attempts=std::stoi(argv[7]),warmups=std::stoi(argv[8]);
    std::string order=argv[5];
    if((dim!=128&&dim!=768)||slots<=ROOTS||slots>4096||pre<0||pre>10000||hold<0||hold>10000||(order!="EL"&&order!="LE")||trials<1||trials>100||attempts<1||attempts>8192||warmups<0||warmups>10)throw std::runtime_error("unsupported contract");
    cudaDeviceProp prop{};cuda_try(cudaGetDeviceProperties(&prop,0));if(!prop.cooperativeLaunch)throw std::runtime_error("cooperative residency unsupported");
    int resident=0;cuda_try(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,run,THREADS,0));if(resident*prop.multiProcessorCount<READERS+1)throw std::runtime_error("resident grid too large");
    for(char scope:order)for(int r=-warmups;r<trials;++r)trial(dim,slots,pre,hold,scope,attempts,r);
    fprintf(stderr,"PASS: native DEBR; protected payload oracle; acknowledgments and final capacity\n");
}catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
