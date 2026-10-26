#include "llama-memory-hybrid-nsa.h"

#include "llama-model.h"

llama_memory_hybrid_nsa::llama_memory_hybrid_nsa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_summary) :
    llama_memory_hybrid(model, type_k, type_v, v_trans, kv_size, n_pad, 0, LLAMA_SWA_TYPE_NONE,
        type_r, type_s, rs_size, n_seq_max, n_rs_seq, offload, unified, filter_attn, filter_recr),
    hparams_summary(model.hparams),
    mem_summary([&] {
        const uint32_t block_size = model.hparams.indexer_block_size;
        GGML_ASSERT(block_size > 0);
        const uint32_t summary_size = (kv_size + block_size - 1) / block_size;
        return new llama_kv_cache(model, hparams_summary, type_k, type_v, v_trans, offload, unified,
            summary_size, n_seq_max, n_pad, 0, LLAMA_SWA_TYPE_NONE, nullptr, filter_summary, nullptr, nullptr, "nsa_");
    }()) {}

llama_kv_cache * llama_memory_hybrid_nsa::get_mem_summary() const {
    return mem_summary.get();
}
