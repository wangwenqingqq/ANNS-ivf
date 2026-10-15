#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace faiss;
using namespace faiss::gpu;

void checked(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// Only exposes existing state for tests; no upstream lifecycle or algorithm changes.
struct ObservedIndex : GpuIndexSIVF {
    using GpuIndexSIVF::GpuIndexSIVF;
    SlabManagerDevice view() { return slab_manager_->getDeviceView(); }
    int head() {
        int result; checked(cudaMemcpy(&result,list_heads_->data(),sizeof(int),cudaMemcpyDeviceToHost)); return result;
    }
};

int main() {
    try {
        StandardGpuResources res; res.setTempMemory(0);
        GpuIndexIVFConfig config; config.device=0;
        ObservedIndex index(&res,8,1,METRIC_L2,config);
        index.initSlabManager(128,1);
        std::vector<float> centroid(8,0.0f);
        index.quantizer->add(1,centroid.data()); index.is_trained=true;
        std::vector<float> vec(8,1.0f); idx_t id=0;
        index.add_with_ids(1,vec.data(),&id); checked(cudaDeviceSynchronize());
        auto manager=index.view(); auto before=index.ntotal;
        // Controlled OOM: temporarily make the free stack empty after a successful control insertion.
        int free_before; checked(cudaMemcpy(&free_before,manager.free_list_top,sizeof(int),cudaMemcpyDeviceToHost));
        int h=index.head(); SlabMetadata meta;
        checked(cudaMemcpy(&meta,manager.slab_metadata+h,sizeof(meta),cudaMemcpyDeviceToHost));
        for(int i=0;i<31;++i) { id=i+1; index.add_with_ids(1,vec.data(),&id); }
        checked(cudaDeviceSynchronize()); before=index.ntotal;
        int zero=0; checked(cudaMemcpy(manager.free_list_top,&zero,sizeof(int),cudaMemcpyHostToDevice));
        id=100; bool threw=false;
        try { index.add_with_ids(1,vec.data(),&id); checked(cudaDeviceSynchronize()); }
        catch(const std::exception& e) { threw=true; std::cout << "API_EXCEPTION " << e.what() << '\n'; }
        AddressTableEntry entry; checked(cudaMemcpy(&entry,manager.address_table+100,sizeof(entry),cudaMemcpyDeviceToHost));
        std::cout << "T7_PUBLIC controlled_oom=true control_first_insert=1"
                  << " initial_free=" << free_before << " first_count=" << meta.valid_count
                  << " before_ntotal=" << before << " after_ntotal=" << index.ntotal
                  << " new_id_committed=" << (entry.packed_address!=UINT64_MAX)
                  << " exception=" << threw << '\n';
        return 0;
    } catch(const std::exception& e) { std::cerr << "HARNESS_ERROR " << e.what() << '\n'; return 2; }
}
