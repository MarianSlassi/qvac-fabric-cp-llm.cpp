#include "llama-memory-hybrid-nsa.h"

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"

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
    const layer_filter_cb & filter_recr) :
    llama_memory_hybrid(model, type_k, type_v, v_trans, kv_size, n_pad, 0, LLAMA_SWA_TYPE_NONE,
        type_r, type_s, rs_size, n_seq_max, n_rs_seq, offload, unified, filter_attn, filter_recr) {

    const uint32_t block_size = model.hparams.indexer_block_size;
    GGML_ASSERT(block_size > 0);

    const uint32_t n_blocks = kv_size / block_size;
    mem_summary = std::make_unique<llama_memory_nsa_summary>(model, n_blocks, n_seq_max, filter_attn);
}

llama_memory_nsa_summary * llama_memory_hybrid_nsa::get_summary() const {
    return mem_summary.get();
}

llama_memory_context_ptr llama_memory_hybrid_nsa::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (embd_all) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                const bool unified = (get_mem_attn()->get_n_stream() == 1);
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;
                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (!get_mem_recr()->prepare(ubatches)) {
            return std::make_unique<llama_memory_hybrid_nsa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_attn = get_mem_attn()->prepare(ubatches);
        if (sinfos_attn.empty()) {
            return std::make_unique<llama_memory_hybrid_nsa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_nsa_context>(
                this, std::move(sinfos_attn), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_hybrid_nsa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_nsa::init_full() {
    return std::make_unique<llama_memory_hybrid_nsa_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_nsa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_nsa_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_nsa::get_can_shift() const {
    // block summaries are indexed by absolute block number
    return false;
}

void llama_memory_hybrid_nsa::clear(bool data) {
    llama_memory_hybrid::clear(data);
    mem_summary->clear(data);
}

bool llama_memory_hybrid_nsa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    (void) seq_id;
    (void) p0;
    (void) p1;
    // summaries are indexed by absolute block number and cannot be shifted or
    // partially removed; the caller has to clear the context instead
    LLAMA_LOG_WARN("%s: partial sequence removal is not supported by the NSA summary cache\n", __func__);
    return false;
}

void llama_memory_hybrid_nsa::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    (void) src;
    (void) dst;
    (void) p0;
    (void) p1;
    GGML_ABORT("%s: sequence copy is not supported by the NSA summary cache\n", __func__);
}

void llama_memory_hybrid_nsa::seq_keep(llama_seq_id seq_id) {
    (void) seq_id;
    GGML_ABORT("%s: sequence keep is not supported by the NSA summary cache\n", __func__);
}

void llama_memory_hybrid_nsa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    (void) seq_id;
    (void) p0;
    (void) p1;
    (void) shift;
    GGML_ABORT("%s: context shifting is not supported by the NSA summary cache\n", __func__);
}

void llama_memory_hybrid_nsa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    (void) seq_id;
    (void) p0;
    (void) p1;
    (void) d;
    GGML_ABORT("%s: context shifting is not supported by the NSA summary cache\n", __func__);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_nsa::memory_breakdown() const {
    auto res = llama_memory_hybrid::memory_breakdown();
    for (const auto & item : mem_summary->breakdown()) {
        res[item.first] += item.second;
    }
    return res;
}

void llama_memory_hybrid_nsa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_summary->state_write(io);
    }
}

void llama_memory_hybrid_nsa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_memory_hybrid::state_read(io, seq_id, flags);
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_summary->state_read(io);
    }
}

llama_memory_hybrid_nsa_context::llama_memory_hybrid_nsa_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_nsa_context::llama_memory_hybrid_nsa_context(llama_memory_hybrid_nsa * mem) :
    llama_memory_hybrid_context(mem),
    mem_summary(mem->get_summary()) {}

llama_memory_hybrid_nsa_context::llama_memory_hybrid_nsa_context(
        llama_memory_hybrid_nsa * mem,
        llama_context * lctx,
        bool optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem_summary(mem->get_summary()) {}

llama_memory_hybrid_nsa_context::llama_memory_hybrid_nsa_context(
        llama_memory_hybrid_nsa * mem,
        slot_info_vec_t sinfos_attn,
        std::vector<llama_ubatch> ubatches) :
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), std::move(ubatches)),
    mem_summary(mem->get_summary()) {}

const llama_memory_nsa_summary * llama_memory_hybrid_nsa_context::get_summary() const {
    return mem_summary;
}
