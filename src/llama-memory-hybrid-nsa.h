#pragma once

#include "llama-memory-hybrid.h"

#include <memory>

class llama_memory_hybrid_nsa : public llama_memory_hybrid {
public:
    llama_memory_hybrid_nsa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_summary);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    llama_kv_cache * get_mem_summary() const;

private:
    llama_hparams hparams_summary;
    const std::unique_ptr<llama_kv_cache> mem_summary;
};

class llama_memory_hybrid_nsa_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_hybrid_nsa_context(llama_memory_status status);

    explicit llama_memory_hybrid_nsa_context(llama_memory_hybrid_nsa * mem);

    llama_memory_hybrid_nsa_context(
            llama_memory_hybrid_nsa * mem,
                      llama_context * lctx,
                               bool   optimize);

    llama_memory_hybrid_nsa_context(
            llama_memory_hybrid_nsa * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_summary,
          std::vector<llama_ubatch>   ubatches);

    bool next() override;
    bool apply() override;

    const llama_kv_cache_context * get_summary() const;

private:
    const llama_memory_context_ptr ctx_summary;
    size_t i_cur = 0;
};
