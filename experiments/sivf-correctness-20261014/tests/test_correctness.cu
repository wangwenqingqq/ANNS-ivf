#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Compile the pinned upstream implementations, not a reimplementation.
#ifdef SIVF_COUNTED
#include "counted_append.cu"
#else
#include <faiss/gpu/impl/SIVFAppend.cu>
#endif
#include <faiss/gpu/impl/SIVFSearch.cu>
#include <faiss/gpu/SIVFDeletion.cu>

using faiss::idx_t;
using namespace faiss::gpu;

__global__ void audit_release(SlabManagerDevice m) { m.free_slab(1); }
__global__ void audit_allocate(SlabManagerDevice m, int* result) {
#ifdef SIVF_FREE_PAUSED
    auto start=clock64();
    while(atomicAdd(&audit_release_ready,0)==0 && clock64()-start<1000000000ULL) {}
#endif
    *result=m.allocate_slab();
#ifdef SIVF_FREE_PAUSED
    atomicExch(&audit_release_resume,1);
#endif
}

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void gpu_sync() { check(cudaGetLastError()); check(cudaDeviceSynchronize()); }

template<class T> struct Buffer {
    T* p = nullptr;
    size_t n;
    explicit Buffer(size_t size) : n(size) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& x) {
        if (x.size() > n) throw std::runtime_error("buffer overflow in harness");
        check(cudaMemcpy(p, x.data(), x.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> x(n);
        check(cudaMemcpy(x.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost));
        return x;
    }
};

float value(idx_t id, int d) {
    uint32_t x = uint32_t(id + 1) * 0x9e3779b9u ^ uint32_t(d + 1) * 0x85ebca6bu;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    return float(int(x & 0xffffu) - 32768) / 1024.0f;
}

struct Fixture {
    int dim, pool, lists, max_ids;
    Buffer<SlabMetadata> md;
    Buffer<float> payload;
    Buffer<AddressTableEntry> att;
    Buffer<int> free_list, top, heads, deleted;
    Buffer<idx_t> ids;
    SlabManagerDevice manager;
    std::map<idx_t, int> oracle;

    Fixture(int d=8, int p=128, int l=1, int m=2048)
        : dim(d), pool(p), lists(l), max_ids(m), md(p), payload(size_t(p)*32*d),
          att(m), free_list(p), top(1), heads(l), deleted(1), ids(size_t(p)*32) {
        manager = {md.p, payload.p, att.p, free_list.p, top.p, p, d};
        reset();
    }
    void reset() {
        gpu_sync();
        md.put(std::vector<SlabMetadata>(pool));
        std::vector<int> f(pool); std::iota(f.begin(), f.end(), 0);
        free_list.put(f); top.put({pool}); heads.put(std::vector<int>(lists, -1));
        check(cudaMemset(att.p, 0xff, max_ids*sizeof(AddressTableEntry)));
        check(cudaMemset(ids.p, 0xff, ids.n*sizeof(idx_t)));
        check(cudaMemset(payload.p, 0xff, payload.n*sizeof(float)));
        oracle.clear();
    }
    void insert(const std::vector<idx_t>& input, int list=0, cudaStream_t stream=nullptr) {
        Buffer<idx_t> di(input.size()), assign(input.size());
        Buffer<float> data(input.size()*dim);
        std::vector<float> host(input.size()*dim);
        for (size_t i=0;i<input.size();++i) {
            if (input[i]<0 || input[i]>=max_ids) throw std::runtime_error("invalid harness ID");
            for (int d=0;d<dim;++d) host[i*dim+d]=value(input[i],d);
            oracle[input[i]]=list;
        }
        di.put(input); assign.put(std::vector<idx_t>(input.size(),list)); data.put(host);
        runSIVFAppend(manager, heads.p, ids.p, int(input.size()), dim, assign.p, data.p, di.p, stream);
        gpu_sync();
    }
    void fill(int n) { for (int i=0;i<n;++i) insert({i}); }
    int erase(const std::vector<idx_t>& input) {
        Buffer<idx_t> di(input.size()); di.put(input); deleted.put({0});
        sivf_delete_kernel<<<(input.size()+255)/256,256>>>(manager,di.p,int(input.size()),deleted.p);
        gpu_sync();
        for (auto id:input) oracle.erase(id);
        return deleted.get()[0];
    }
    std::set<int> reachable() const {
        auto meta=md.get(); auto h=heads.get(); std::set<int> out;
        for (int head:h) {
            std::set<int> seen;
            for (int s=head;s!=-1;) {
                if (s<0 || s>=pool || !seen.insert(s).second) break;
                out.insert(s); s=meta[s].next_slab_idx;
            }
        }
        return out;
    }
    void search(idx_t query, int list, const std::string& path, int k=10) {
        Buffer<float> q(dim), dist(k); Buffer<idx_t> coarse(1), labels(k);
        std::vector<float> v(dim); for(int d=0;d<dim;++d) v[d]=value(query,d);
        q.put(v); coarse.put({list});
        runSIVFSearch(manager,heads.p,ids.p,1,dim,k,1,q.p,coarse.p,dist.p,labels.p,nullptr);
        gpu_sync(); auto ld=labels.get(); auto dd=dist.get();
        std::vector<std::pair<double,idx_t>> ref;
        for (auto [id,owner]:oracle) if(owner==list) {
            double d2=0; for(int d=0;d<dim;++d) { double diff=v[d]-value(id,d); d2+=diff*diff; }
            ref.emplace_back(d2,id);
        }
        std::sort(ref.begin(),ref.end());
        std::ofstream out(path); out.precision(10);
        out << "{\"query_id\":" << query << ",\"list\":" << list << ",\"labels\":[";
        for(int i=0;i<k;++i) out << (i?",":"") << ld[i];
        out << "],\"distances\":[";
        for(int i=0;i<k;++i) out << (i?",":"") << dd[i];
        out << "],\"reference\":[";
        for(size_t i=0;i<std::min(size_t(k),ref.size());++i)
            out << (i?",":"") << "{\"id\":" << ref[i].second << ",\"distance\":" << ref[i].first << "}";
        out << "]}\n";
        bool ok=true;
        for(int i=0;i<k;++i) {
            if(i<int(ref.size())) ok &= ld[i]==ref[i].second && std::abs(dd[i]-ref[i].first)<=1e-4*std::max(1.0,ref[i].first);
            else ok &= ld[i]==-1;
        }
        std::cout << "SEARCH query=" << query << " list=" << list << " exact_topk=" << (ok?"PASS":"FAIL") << "\n";
    }
    void snapshot(const std::string& path) {
        gpu_sync(); auto meta=md.get(); auto f=free_list.get(); auto t=top.get()[0];
        auto h=heads.get(); auto a=att.get(); auto physical=ids.get(); auto data=payload.get();
        std::ofstream out(path); out.precision(9);
        out << "{\"dim\":" << dim << ",\"pool_size\":" << pool << ",\"heads\":[";
        for(int i=0;i<lists;++i) out << (i?",":"") << h[i];
        out << "],\"free_list_top\":" << t << ",\"free_list\":[";
        for(int i=0;i<std::max(0,std::min(pool,t));++i) out << (i?",":"") << f[i];
        out << "],\"address_table\":[";
        for(int i=0;i<max_ids;++i) out << (i?",":"") << "\"" << a[i].packed_address << "\"";
        out << "],\"oracle\":["; int oi=0;
        for(auto [id,list]:oracle) {
            out << (oi++?",":"") << "{\"id\":" << id << ",\"list\":" << list << ",\"vector\":[";
            for(int d=0;d<dim;++d) out << (d?",":"") << value(id,d);
            out << "]}";
        }
        out << "],\"slabs\":[";
        for(int s=0;s<pool;++s) {
            out << (s?",":"") << "{\"index\":" << s << ",\"next\":" << meta[s].next_slab_idx
                << ",\"bitmap\":" << meta[s].validity_bitmap << ",\"count\":" << meta[s].valid_count << ",\"records\":[";
            int ri=0;
            for(int slot=0;slot<32;++slot) if((meta[s].validity_bitmap>>slot)&1u) {
                out << (ri++?",":"") << "{\"slot\":" << slot << ",\"id\":" << physical[s*32+slot] << ",\"vector\":[";
                for(int d=0;d<dim;++d) {
                    float x=data[(size_t(s)*32+slot)*dim+d];
                    out << (d?",":""); if(std::isfinite(x)) out << x; else out << "null";
                }
                out << "]}";
            }
            out << "]}";
        }
        out << "]}\n";
        auto r=reachable(); std::set<int> free(f.begin(),f.begin()+std::max(0,std::min(pool,t)));
        int shared=0; for(int s:r) shared+=free.count(s);
        std::cout << "SNAPSHOT " << std::filesystem::path(path).filename().string() << " live=" << oracle.size()
                  << " reachable=" << r.size() << " free=" << t << " reachable_and_free=" << shared
                  << " orphan=" << pool-int(free.size())-int(r.size())+shared << "\n";
    }
};

int main(int argc,char** argv) {
    try {
        if(argc<3) throw std::runtime_error("usage: test_correctness CASE OUTPUT_DIR [ROUNDS]");
        std::string name=argv[1], dir=argv[2]; std::filesystem::create_directories(dir);
        check(cudaSetDevice(0));
        if(name=="smoke" || name=="T1_hole" || name=="T1_tail") {
            Fixture f(name=="smoke"?128:8); f.fill(32); f.snapshot(dir+"/A.json");
            if(name=="smoke") { f.search(17,0,dir+"/search.json"); return 0; }
            int head=f.heads.get()[0]; auto ids=f.ids.get();
            idx_t target=ids[head*32+(name=="T1_hole"?5:31)], tail=ids[head*32+31];
            std::cout << "DELETE id=" << target << " tail_id=" << tail << " deleted=" << f.erase({target}) << "\n";
            f.snapshot(dir+"/B.json");
            std::cout << "REPEAT_DELETE count=" << f.erase({target}) << "\n"; f.snapshot(dir+"/B_repeat.json");
            f.insert({100}); f.snapshot(dir+"/C.json"); f.search(tail,0,dir+"/search.json");
        } else if(name=="T2_head" || name=="T2_middle") {
            Fixture f(8,128,2); f.fill(name=="T2_head"?64:96); f.snapshot(dir+"/A.json");
            auto meta=f.md.get(); int slab=f.heads.get()[0];
            if(name=="T2_middle") slab=meta[slab].next_slab_idx;
            auto physical=f.ids.get(); std::vector<idx_t> remove;
            for(int i=0;i<32;++i) if((meta[slab].validity_bitmap>>i)&1u) remove.push_back(physical[slab*32+i]);
            std::cout << "EMPTY_SLAB slab=" << slab << " deleted=" << f.erase(remove) << "\n";
            f.snapshot(dir+"/B.json"); f.insert({100},1); f.snapshot(dir+"/C.json");
            f.search(100,0,dir+"/search_list0.json"); f.search(100,1,dir+"/search_list1.json");
        } else if(name=="T3") {
            int rounds=argc>3?std::stoi(argv[3]):1000;
            if(rounds<1) throw std::runtime_error("rounds must be positive");
            std::ofstream csv(dir+"/trials.csv"); csv << "batch,trial,free,reachable,orphan,committed,head_cas_failed,slab_allocated,slab_reclaimed,insert_success,insert_failed\n";
            for(int batch:{64,256,1024}) {
                Fixture f(8,batch*64,1,batch+1); std::vector<idx_t> in(batch); std::iota(in.begin(),in.end(),0);
                int failures=0,max_orphan=0,total_orphan=0,min_commit=batch;
                for(int trial=0;trial<rounds;++trial) {
                    f.reset();
#ifdef SIVF_COUNTED
                    unsigned long long zero[5]={}; check(cudaMemcpyToSymbol(audit_counts,zero,sizeof(zero)));
#endif
                    f.insert(in); auto r=f.reachable(); int free=f.top.get()[0];
                    auto a=f.att.get(); int committed=0; for(int i=0;i<batch;++i) committed+=a[i].packed_address!=UINT64_MAX;
                    int orphan=f.pool-free-int(r.size()); csv << batch << ',' << trial << ',' << free << ',' << r.size() << ',' << orphan << ',' << committed;
#ifdef SIVF_COUNTED
                    unsigned long long counts[5]; check(cudaMemcpyFromSymbol(counts,audit_counts,sizeof(counts)));
                    for(auto c:counts) csv << ',' << c;
#else
                    csv << ",NA,NA,NA,NA,NA";
#endif
                    csv << '\n';
                    failures+=orphan>0; max_orphan=std::max(max_orphan,orphan); total_orphan+=orphan; min_commit=std::min(min_commit,committed);
                    if(trial==0) f.snapshot(dir+"/batch_"+std::to_string(batch)+"_trial0.json");
                }
                std::cout << "T3 batch=" << batch << " trials=" << rounds << " trials_with_orphans=" << failures
                          << " total_orphans=" << total_orphan << " max_orphans=" << max_orphan << " min_committed=" << min_commit << "\n";
            }
        } else if(name=="T5") {
            int rounds=argc>3?std::stoi(argv[3]):1000;
            if(rounds<1) throw std::runtime_error("rounds must be positive");
            Fixture f(8,2,1,2); Buffer<int> result(1);
            cudaStream_t release, allocate;
            check(cudaStreamCreateWithFlags(&release,cudaStreamNonBlocking));
            check(cudaStreamCreateWithFlags(&allocate,cudaStreamNonBlocking));
            std::ofstream csv(dir+"/publication.csv"); csv << "trial,allocated,ready,resume\n";
            int stale=0,success=0,empty=0;
            for(int trial=0;trial<rounds;++trial) {
                f.reset(); f.top.put({0}); result.put({-2});
#ifdef SIVF_FREE_PAUSED
                int zero=0; check(cudaMemcpyToSymbol(audit_release_ready,&zero,sizeof(zero)));
                check(cudaMemcpyToSymbol(audit_release_resume,&zero,sizeof(zero)));
#endif
                audit_release<<<1,1,0,release>>>(f.manager);
                audit_allocate<<<1,1,0,allocate>>>(f.manager,result.p);
                gpu_sync(); int got=result.get()[0],ready=-1,resume=-1;
#ifdef SIVF_FREE_PAUSED
                check(cudaMemcpyFromSymbol(&ready,audit_release_ready,sizeof(ready)));
                check(cudaMemcpyFromSymbol(&resume,audit_release_resume,sizeof(resume)));
#endif
                csv << trial << ',' << got << ',' << ready << ',' << resume << '\n';
                stale+=got==0; success+=got==1; empty+=got==-1;
            }
            std::cout << "T5 trials=" << rounds << " stale_allocations=" << stale << " correct_allocations=" << success << " empty_returns=" << empty << '\n';
            check(cudaStreamDestroy(release)); check(cudaStreamDestroy(allocate));
        } else if(name=="T4") {
            int rounds=argc>3?std::stoi(argv[3]):1000;
            if(rounds<1) throw std::runtime_error("rounds must be positive");
            Fixture f; constexpr int nq=32,k=10;
            Buffer<idx_t> insert_ids(8),assign(8),delete_ids(8),coarse(nq),labels(nq*k);
            Buffer<float> vectors(8*f.dim),queries(nq*f.dim),dist(nq*k);
            std::vector<idx_t> in(8),gone(8); std::iota(in.begin(),in.end(),100); std::iota(gone.begin(),gone.end(),0);
            insert_ids.put(in); assign.put(std::vector<idx_t>(8,0)); delete_ids.put(gone); coarse.put(std::vector<idx_t>(nq,0));
            std::vector<float> v(8*f.dim),q(nq*f.dim);
            for(int i=0;i<8;++i) for(int d=0;d<f.dim;++d) v[i*f.dim+d]=value(in[i],d);
            for(int i=0;i<nq;++i) for(int d=0;d<f.dim;++d) q[i*f.dim+d]=value(i,d);
            vectors.put(v); queries.put(q);
            cudaStream_t streams[3]; cudaEvent_t start[3],end[3],gate;
            check(cudaEventCreate(&gate));
            for(int i=0;i<3;++i) { check(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking)); check(cudaEventCreate(&start[i])); check(cudaEventCreate(&end[i])); }
            std::ofstream csv(dir+"/overlap.csv"); csv << "trial,payload_mismatch,search_start_ms,search_end_ms,insert_start_ms,insert_end_ms,delete_start_ms,delete_end_ms\n";
            int total_bad=0;
            for(int trial=0;trial<rounds;++trial) {
                f.reset(); f.fill(32); f.deleted.put({0});
                for(auto id:gone) f.oracle.erase(id); for(auto id:in) f.oracle[id]=0;
                check(cudaEventRecord(gate));
                for(int i=0;i<3;++i) check(cudaStreamWaitEvent(streams[i],gate,0));
                check(cudaEventRecord(start[0],streams[0]));
                runSIVFSearch(f.manager,f.heads.p,f.ids.p,nq,f.dim,k,1,queries.p,coarse.p,dist.p,labels.p,streams[0]);
                check(cudaEventRecord(end[0],streams[0]));
                check(cudaEventRecord(start[1],streams[1]));
                runSIVFAppend(f.manager,f.heads.p,f.ids.p,8,f.dim,assign.p,vectors.p,insert_ids.p,streams[1]);
                check(cudaEventRecord(end[1],streams[1]));
                check(cudaEventRecord(start[2],streams[2]));
                sivf_delete_kernel<<<1,256,0,streams[2]>>>(f.manager,delete_ids.p,8,f.deleted.p);
                check(cudaEventRecord(end[2],streams[2])); gpu_sync();
                auto ll=labels.get(); auto dd=dist.get(); int bad=0;
                for(int i=0;i<nq*k;++i) if(ll[i]!=-1) {
                    idx_t id=ll[i];
                    if(!((id>=0 && id<32)||(id>=100 && id<108))) { ++bad; continue; }
                    double expected=0; for(int d=0;d<f.dim;++d) { double delta=q[(i/k)*f.dim+d]-value(id,d); expected+=delta*delta; }
                    bad+=!std::isfinite(dd[i]) || std::abs(dd[i]-expected)>1e-4*std::max(1.0,expected);
                }
                total_bad+=bad; csv << trial << ',' << bad;
                for(int i=0;i<3;++i) { float a,b; check(cudaEventElapsedTime(&a,gate,start[i])); check(cudaEventElapsedTime(&b,gate,end[i])); csv << ',' << a << ',' << b; }
                csv << '\n';
                if(trial==0 || trial==rounds-1) f.snapshot(dir+"/trial_"+std::to_string(trial)+".json");
            }
            std::cout << "T4 trials=" << rounds << " inconsistent_returned_pairs=" << total_bad << " (event windows bound operations; not a snapshot-semantics claim)\n";
            for(int i=0;i<3;++i) { check(cudaEventDestroy(start[i])); check(cudaEventDestroy(end[i])); check(cudaStreamDestroy(streams[i])); }
            check(cudaEventDestroy(gate));
        } else if(name=="T3_growth") {
            Fixture f(8,131072,1,64001);
            std::ofstream csv(dir+"/growth.csv"); csv << "batch,attempted,free,reachable,orphan\n";
            for(int trial=0;trial<1000;++trial) {
                std::vector<idx_t> in(64); std::iota(in.begin(),in.end(),trial*64);
                f.insert(in); auto r=f.reachable(); int free=f.top.get()[0];
                int orphan=f.pool-free-int(r.size());
                csv << trial << ',' << (trial+1)*64 << ',' << free << ',' << r.size() << ',' << orphan << '\n';
                if(trial==0 || trial==999) f.snapshot(dir+"/batch_"+std::to_string(trial)+".json");
                if(free<1024) throw std::runtime_error("growth pool margin exhausted; stop, do not compare rates");
            }
        } else if(name.rfind("dim_",0)==0) {
            int dim=std::stoi(name.substr(4)); if(dim<1 || dim>1024) throw std::runtime_error("invalid dimension");
            Fixture f(dim,128); f.fill(32); f.snapshot(dir+"/state.json"); f.search(17,0,dir+"/search.json");
        } else if(name=="T7_pool") {
            Fixture f(8,1); f.fill(32); f.snapshot(dir+"/A.json");
            f.insert({100}); f.snapshot(dir+"/B.json");
            int committed=0; for(auto a:f.att.get()) committed+=a.packed_address!=UINT64_MAX;
            std::cout << "T7 low_level attempted=33 committed=" << committed << " (no ntotal API claim)\n";
        } else throw std::runtime_error("unknown case");
        gpu_sync(); return 0;
    } catch(const std::exception& e) { std::cerr << "HARNESS_ERROR " << e.what() << '\n'; return 2; }
}
