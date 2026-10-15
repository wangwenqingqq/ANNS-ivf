#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Id = unsigned long long;
constexpr int K=10, WARPS=8;
constexpr Id BAD=~Id(0);
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t n):n(n) { if(n) ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& x) { if(x.size()>n) throw std::runtime_error("upload overflow"); if(!x.empty()) ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> x(n); if(n) ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return x; }
};
float hashval(Id id,int d) {
    uint32_t x=uint32_t(id+1)*0x9e3779b9u ^ uint32_t(d+1)*0x85ebca6bu;
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
    return float(int(x&65535)-32768)/32768.f;
}
float center(int list,int d) { return hashval(1000000000ULL+list,d)*0.5f; }
float value(Id id,int list,int d) { return center(list,d)+hashval(id,d); }

template<bool INDIRECT> __global__ void move_rows(float* data,Id* ids,int* map,
        const int* dst,const int* src,int count,int dim) {
    int i=blockIdx.x; if(i>=count) return;
    if constexpr(INDIRECT) { if(threadIdx.x==0) map[dst[i]]=map[src[i]]; }
    else {
        for(int d=threadIdx.x;d<dim;d+=blockDim.x) data[size_t(dst[i])*dim+d]=data[size_t(src[i])*dim+d];
        if(threadIdx.x==0) ids[dst[i]]=ids[src[i]];
    }
}
__global__ void move_directory(int* map,const int* dst,const int* src,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) map[dst[i]]=map[src[i]];
}
template<bool INDIRECT> __global__ void insert_rows(float* data,Id* ids,int* map,
        const int* dst,const int* physical,const int* input_index,
        const float* incoming,const Id* newids,int count,int dim) {
    int i=blockIdx.x; if(i>=count) return;
    int row=INDIRECT?physical[i]:dst[i], input=input_index[i];
    for(int d=threadIdx.x;d<dim;d+=blockDim.x) data[size_t(row)*dim+d]=incoming[size_t(input)*dim+d];
    if(threadIdx.x==0) { ids[row]=newids[input]; if constexpr(INDIRECT) map[dst[i]]=row; }
}
__device__ void add(float* ds,Id* is,float d,Id id) {
    if(d<ds[K-1] || (d==ds[K-1] && id<is[K-1])) {
        int j=K-1;
        while(j>0 && (d<ds[j-1] || (d==ds[j-1] && id<is[j-1]))) { ds[j]=ds[j-1]; is[j]=is[j-1]; --j; }
        ds[j]=d; is[j]=id;
    }
}
template<bool INDIRECT> __global__ void scan(const float* data,const Id* ids,const int* map,
        const int* counts,const float* queries,const int* probes,int dim,int capacity,int np,
        float* distances,Id* labels) {
    int q=blockIdx.x/np, p=blockIdx.x%np, list=probes[q*np+p];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    float ds[K]; Id is[K];
    for(int k=0;k<K;++k) { ds[k]=CUDART_INF_F; is[k]=BAD; }
    for(int i=warp;i<counts[list];i+=WARPS) {
        int logical=list*capacity+i, row=INDIRECT?map[logical]:logical;
        float d2=0;
        for(int d=lane;d<dim;d+=32) { float delta=queries[size_t(q)*dim+d]-data[size_t(row)*dim+d]; d2+=delta*delta; }
        for(int shift=16;shift;shift>>=1) d2+=__shfl_down_sync(0xffffffff,d2,shift);
        if(lane==0) add(ds,is,d2,ids[row]);
    }
    __shared__ float sd[WARPS*K]; __shared__ Id si[WARPS*K];
    if(lane==0) for(int k=0;k<K;++k) { sd[warp*K+k]=ds[k]; si[warp*K+k]=is[k]; }
    __syncthreads();
    if(threadIdx.x==0) {
        for(int i=K;i<WARPS*K;++i) if(si[i]!=BAD) add(ds,is,sd[i],si[i]);
        for(int k=0;k<K;++k) { distances[blockIdx.x*K+k]=ds[k]; labels[blockIdx.x*K+k]=is[k]; }
    }
}
__global__ void merge(const float* partial,const Id* plabels,int np,float* out,Id* ids) {
    if(threadIdx.x) return;
    float ds[K]; Id is[K]; for(int k=0;k<K;++k) { ds[k]=CUDART_INF_F; is[k]=BAD; }
    for(int j=0;j<np*K;++j) { int ix=blockIdx.x*np*K+j; if(plabels[ix]!=BAD) add(ds,is,partial[ix],plabels[ix]); }
    for(int k=0;k<K;++k) { out[blockIdx.x*K+k]=ds[k]; ids[blockIdx.x*K+k]=is[k]; }
}

struct Spec { int n=8192,lists=32,capacity=512,queries=32,dim; };
struct Row { Id id; int owner; };
struct Plan { std::vector<int> dst,src,insert,physical,input; int moves() const { return int(dst.size()); } };
struct Layout {
    Spec s; char kind;
    std::vector<std::vector<Id>> host;
    std::vector<std::vector<int>> phys;
    Buffer<float> data; Buffer<Id> ids; Buffer<int> map,counts;
    Layout(Spec s,char kind):s(s),kind(kind),host(s.lists),phys(s.lists),
        data(size_t(s.lists)*s.capacity*s.dim),ids(s.lists*s.capacity),
        map(kind=='I'?s.lists*s.capacity:0),counts(s.lists) {
        std::vector<float> payload(data.n,0); std::vector<Id> iid(ids.n,BAD);
        std::vector<int> directory(ids.n,-1),permutation(ids.n); std::iota(permutation.begin(),permutation.end(),0);
        std::mt19937 rng(20261015); std::shuffle(permutation.begin(),permutation.end(),rng);
        int each=s.n/s.lists;
        for(int l=0;l<s.lists;++l) for(int j=0;j<each;++j) {
            Id id=l*each+j; int logical=l*s.capacity+j;
            int row=kind=='I'?permutation[id]:logical;
            host[l].push_back(id); phys[l].push_back(row); iid[row]=id; directory[logical]=row;
            for(int d=0;d<s.dim;++d) payload[size_t(row)*s.dim+d]=value(id,l,d);
        }
        data.put(payload); ids.put(iid); if(kind=='I') map.put(directory);
        counts.put(std::vector<int>(s.lists,each));
    }
    size_t bytes() const { return data.n*sizeof(float)+ids.n*sizeof(Id)+map.n*sizeof(int)+counts.n*sizeof(int); }
    Plan plan(const std::vector<Id>& removed,const std::vector<Row>& incoming) {
        std::set<Id> gone(removed.begin(),removed.end());
        std::vector<int> freed;
        if(kind=='I') {
            std::map<Id,int> address;
            for(int l=0;l<s.lists;++l) for(size_t j=0;j<host[l].size();++j) address.emplace(host[l][j],phys[l][j]);
            for(Id id:removed) freed.push_back(address.at(id));
        }
        Plan out;
        for(int l=0;l<s.lists;++l) {
            auto old=host[l]; auto oldphys=phys[l]; std::vector<int> added;
            for(size_t i=0;i<incoming.size();++i) if(incoming[i].owner==l) added.push_back(int(i));
            int deleted=0; for(Id id:old) deleted+=gone.count(id);
            int final=int(old.size())-deleted+int(added.size());
            if(final<0 || final>s.capacity) throw std::runtime_error("declared list capacity exceeded");
            int prefix=kind=='F'?final:int(old.size())-deleted;
            std::vector<Id> next(final,BAD); std::vector<int> nextphys(final,-1),holes,donors;
            for(int j=0;j<int(old.size());++j) if(!gone.count(old[j])) {
                if(j<prefix) { next[j]=old[j]; nextphys[j]=oldphys[j]; } else donors.push_back(j);
            }
            for(int j=0;j<prefix;++j) if(next[j]==BAD) holes.push_back(j);
            std::vector<int> fresh;
            if(kind=='F') {
                if(holes.size()<added.size()) throw std::runtime_error("invalid fusion plan");
                fresh.assign(holes.begin(),holes.begin()+added.size()); holes.erase(holes.begin(),holes.begin()+added.size());
            } else for(int j=prefix;j<final;++j) fresh.push_back(j);
            if(holes.size()!=donors.size() || fresh.size()!=added.size()) throw std::runtime_error("plan count mismatch");
            for(size_t j=0;j<holes.size();++j) {
                int dst=holes[j],src=donors[j]; out.dst.push_back(l*s.capacity+dst); out.src.push_back(l*s.capacity+src);
                next[dst]=old[src]; nextphys[dst]=kind=='I'?oldphys[src]:l*s.capacity+dst;
            }
            for(size_t j=0;j<fresh.size();++j) {
                int pos=fresh[j],input=added[j],logical=l*s.capacity+pos;
                int physical=kind=='I'?freed[input]:logical;
                out.insert.push_back(logical); out.physical.push_back(physical); out.input.push_back(input);
                next[pos]=incoming[input].id; nextphys[pos]=physical;
            }
            if(std::find(next.begin(),next.end(),BAD)!=next.end()) throw std::runtime_error("unfilled plan hole");
            host[l]=std::move(next); phys[l]=std::move(nextphys);
        }
        return out;
    }
    void validate(const std::map<Id,int>& oracle) const {
        auto x=data.get(); auto ii=ids.get(); auto cc=counts.get(); auto directory=map.get();
        std::set<int> physical; std::map<Id,int> actual;
        for(int l=0;l<s.lists;++l) {
            if(cc[l]!=int(host[l].size())) throw std::runtime_error("device count differs");
            for(int j=0;j<cc[l];++j) {
                int logical=l*s.capacity+j,row=kind=='I'?directory[logical]:logical;
                if(row<0 || row>=int(ids.n) || !physical.insert(row).second) throw std::runtime_error("duplicate/out-of-range physical row");
                Id id=ii[row]; if(id!=host[l][j] || !actual.emplace(id,l).second) throw std::runtime_error("ID/owner mismatch");
                for(int d=0;d<s.dim;++d) if(x[size_t(row)*s.dim+d]!=value(id,l,d)) throw std::runtime_error("payload mismatch");
            }
        }
        if(actual!=oracle || actual.size()!=size_t(s.n)) throw std::runtime_error("live-set conservation failed");
    }
};
struct Workspace {
    Spec s; Buffer<int> dst,src,insert,physical,input,probes;
    Buffer<float> incoming,queries,partial,dist; Buffer<Id> fresh,plabels,labels;
    std::vector<float> hq; std::vector<std::vector<int>> routing;
    cudaEvent_t begin,end;
    Workspace(Spec s,int batch):s(s),dst(batch),src(batch),insert(batch),physical(batch),input(batch),probes(s.queries*s.lists),
        incoming(size_t(batch)*s.dim),queries(size_t(s.queries)*s.dim),partial(s.queries*s.lists*K),dist(s.queries*K),fresh(batch),
        plabels(s.queries*s.lists*K),labels(s.queries*K),hq(queries.n),routing(s.queries) {
        ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        for(int q=0;q<s.queries;++q) {
            int owner=q%s.lists; Id id=owner*(s.n/s.lists)+(q*17)%(s.n/s.lists);
            for(int d=0;d<s.dim;++d) hq[size_t(q)*s.dim+d]=value(id,owner,d)+0.2f*hashval(9999999+q,d);
            std::vector<std::pair<double,int>> scores;
            for(int l=0;l<s.lists;++l) { double d2=0; for(int d=0;d<s.dim;++d) { double x=hq[size_t(q)*s.dim+d]-center(l,d); d2+=x*x; } scores.emplace_back(d2,l); }
            std::sort(scores.begin(),scores.end()); for(auto [v,l]:scores) routing[q].push_back(l);
        }
        queries.put(hq);
    }
    ~Workspace() { cudaEventDestroy(begin); cudaEventDestroy(end); }
    float elapsed() { ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end)); ck(cudaGetLastError()); float ms; ck(cudaEventElapsedTime(&ms,begin,end)); return ms*1000; }
    float update(Layout& f,const Plan& p) {
        dst.put(p.dst); src.put(p.src); insert.put(p.insert); physical.put(p.physical); input.put(p.input);
        std::vector<int> counts; for(auto& list:f.host) counts.push_back(int(list.size())); f.counts.put(counts);
        ck(cudaEventRecord(begin));
        if(p.moves()) {
            if(f.kind=='I') move_directory<<<(p.moves()+255)/256,256>>>(f.map.p,dst.p,src.p,p.moves());
            else move_rows<false><<<p.moves(),128>>>(f.data.p,f.ids.p,f.map.p,dst.p,src.p,p.moves(),s.dim);
        }
        int n=int(p.insert.size());
        if(f.kind=='I') insert_rows<true><<<n,128>>>(f.data.p,f.ids.p,f.map.p,insert.p,physical.p,input.p,incoming.p,fresh.p,n,s.dim);
        else insert_rows<false><<<n,128>>>(f.data.p,f.ids.p,f.map.p,insert.p,physical.p,input.p,incoming.p,fresh.p,n,s.dim);
        return elapsed();
    }
    float search(Layout& f,int np) {
        std::vector<int> pr; for(auto& route:routing) pr.insert(pr.end(),route.begin(),route.begin()+np); probes.put(pr);
        ck(cudaEventRecord(begin));
        if(f.kind=='I') scan<true><<<s.queries*np,256>>>(f.data.p,f.ids.p,f.map.p,f.counts.p,queries.p,probes.p,s.dim,s.capacity,np,partial.p,plabels.p);
        else scan<false><<<s.queries*np,256>>>(f.data.p,f.ids.p,f.map.p,f.counts.p,queries.p,probes.p,s.dim,s.capacity,np,partial.p,plabels.p);
        merge<<<s.queries,32>>>(partial.p,plabels.p,np,dist.p,labels.p);
        return elapsed();
    }
    using Reference=std::vector<std::vector<std::pair<double,Id>>>;
    Reference reference(const std::map<Id,int>& oracle) const {
        Reference ref(s.queries);
        for(int q=0;q<s.queries;++q) {
            for(auto [id,l]:oracle) { double d2=0; for(int d=0;d<s.dim;++d) { double x=hq[size_t(q)*s.dim+d]-value(id,l,d); d2+=x*x; } ref[q].emplace_back(d2,id); }
            std::sort(ref[q].begin(),ref[q].end());
        }
        return ref;
    }
    double check_search(const std::map<Id,int>& oracle,const Reference& ref,int np) {
        auto ll=labels.get(); auto dd=dist.get(); int recall=0;
        for(int q=0;q<s.queries;++q) {
            std::set<int> selected(routing[q].begin(),routing[q].begin()+np); std::vector<std::pair<double,Id>> expected;
            for(auto pair:ref[q]) if(selected.count(oracle.at(pair.second)) && expected.size()<K) expected.push_back(pair);
            std::set<Id> seen;
            for(int k=0;k<K;++k) {
                Id id=ll[q*K+k];
                if(k>=int(expected.size())) { if(id!=BAD) throw std::runtime_error("nonempty search padding"); continue; }
                if(id!=expected[k].second || !std::isfinite(dd[q*K+k]) || std::abs(dd[q*K+k]-expected[k].first)>1e-4*std::max(1.,expected[k].first) || !seen.insert(id).second)
                    throw std::runtime_error("CPU exact selected-list top-k mismatch");
                for(int j=0;j<K;++j) recall+=id==ref[q][j].second;
            }
        }
        return double(recall)/(s.queries*K);
    }
};
std::vector<Id> select_ids(const std::map<Id,int>& oracle,int batch,const std::string& workload,std::mt19937& rng) {
    std::vector<Id> selected;
    if(workload=="fifo") { for(auto [id,l]:oracle) { selected.push_back(id); if(int(selected.size())==batch) break; } }
    else if(workload=="random") { for(auto [id,l]:oracle) selected.push_back(id); std::shuffle(selected.begin(),selected.end(),rng); selected.resize(batch); }
    else {
        std::vector<std::pair<double,Id>> scores;
        for(auto [id,l]:oracle) { double u=(double(rng())+1.)/(double(rng.max())+2.); scores.emplace_back(-std::log(u)/(l<4?20.:1.),id); }
        std::partial_sort(scores.begin(),scores.begin()+batch,scores.end()); for(int i=0;i<batch;++i) selected.push_back(scores[i].second);
    }
    return selected;
}
int main(int argc,char** argv) {
    try {
        if(argc<6) throw std::runtime_error("usage: probe DIM BATCH fifo|random|hot ROUNDS PFI|IFP [small]");
        int dim=std::stoi(argv[1]),batch=std::stoi(argv[2]),rounds=std::stoi(argv[4]); std::string workload=argv[3],order=argv[5];
        bool small=argc>6 && std::string(argv[6])=="small";
        Spec s; s.dim=dim; if(small) { s.n=256; s.lists=4; s.capacity=128; s.queries=8; }
        if((dim!=128 && dim!=768) || batch<1 || batch>s.n/4 || rounds<1 || rounds>2000 || (order!="PFI" && order!="IFP") || (workload!="fifo" && workload!="random" && workload!="hot")) throw std::runtime_error("invalid contract arguments");
        ck(cudaSetDevice(0)); Layout packed(s,'P'),fused(s,'F'),indirect(s,'I'); Workspace w(s,batch);
        std::map<char,Layout*> layouts{{'P',&packed},{'F',&fused},{'I',&indirect}};
        std::map<Id,int> oracle; for(int l=0;l<s.lists;++l) for(Id id:packed.host[l]) oracle.emplace(id,l);
        for(auto [k,f]:layouts) f->validate(oracle);
        std::mt19937 rng(20261015); Id next=s.n; int warmup=small?0:20, primary_np=std::min(4,s.lists);
        std::cout << "kind,dim,batch,workload,order,step,variant,update_us,query_us,moved_survivors,host_plan_us,layout_bytes,recall,nprobe,oracle_checked\n";
        std::cout.precision(10);
        for(int step=0;step<warmup+rounds;++step) {
            auto removed=select_ids(oracle,batch,workload,rng); std::vector<Row> incoming;
            for(Id id:removed) {
                int old=oracle.at(id),owner=(old+1)%s.lists;
                if(workload=="hot" && s.lists>4) owner=old<4?(old+1)%4:4+(old-4+1)%(s.lists-4);
                incoming.push_back({next++,owner});
            }
            std::vector<float> data(size_t(batch)*dim); std::vector<Id> iid;
            for(int i=0;i<batch;++i) { iid.push_back(incoming[i].id); for(int d=0;d<dim;++d) data[size_t(i)*dim+d]=value(incoming[i].id,incoming[i].owner,d); }
            w.incoming.put(data); w.fresh.put(iid);
            for(Id id:removed) if(oracle.erase(id)!=1) throw std::runtime_error("delete failed");
            for(auto row:incoming) if(!oracle.emplace(row.id,row.owner).second) throw std::runtime_error("duplicate new ID");
            bool validate=step==0 || step==warmup+rounds-1;
            Workspace::Reference reference; if(validate) reference=w.reference(oracle);
            for(char kind:order) {
                auto& f=*layouts.at(kind); auto begin=std::chrono::steady_clock::now(); Plan plan=f.plan(removed,incoming);
                double plan_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
                float update=w.update(f,plan),query=w.search(f,primary_np); double recall=-1;
                if(validate) { f.validate(oracle); recall=w.check_search(oracle,reference,primary_np); }
                if(step>=warmup) std::cout << "epoch," << dim << ',' << batch << ',' << workload << ',' << order << ',' << step-warmup << ',' << kind << ',' << update << ',' << query << ',' << plan.moves() << ',' << plan_us << ',' << f.bytes() << ',' << recall << ',' << primary_np << ',' << validate << '\n';
            }
        }
        auto ref=w.reference(oracle);
        for(int np:{1,std::min(4,s.lists),s.lists}) for(char kind:order) {
            auto& f=*layouts.at(kind); f.validate(oracle); for(int i=0;i<10;++i) w.search(f,np);
            for(int i=0;i<20;++i) {
                float query=w.search(f,np); double recall=w.check_search(oracle,ref,np);
                std::cout << "quality," << dim << ',' << batch << ',' << workload << ',' << order << ',' << i << ',' << kind << ",0," << query << ",0,0," << f.bytes() << ',' << recall << ',' << np << ",1\n";
            }
        }
        ck(cudaDeviceSynchronize()); std::cerr << "PASS: live IDs/payloads/ownership and selected-list top-k; fixed pool, serialized diagnostic only\n";
    } catch(const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 2; }
}
