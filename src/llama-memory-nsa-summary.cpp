#include "llama-memory-nsa-summary.h"

#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

llama_memory_nsa_summary::llama_memory_nsa_summary(
        const llama_model & model,
                 uint32_t   n_blocks,
                 uint32_t   n_seqs,
    const layer_filter_cb & filter) {
    const auto & hparams = model.hparams;

    uint32_t n_embd_head = 0;
    uint32_t n_head_kv   = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }
        if (n_embd_head == 0) {
            n_embd_head = hparams.n_embd_head_k(il);
            n_head_kv   = hparams.n_head_kv(il);
        }
        GGML_ASSERT(hparams.n_embd_head_k(il) == n_embd_head);
        GGML_ASSERT(hparams.n_head_kv(il)   == n_head_kv);
    }

    if (n_embd_head == 0) {
        return;
    }

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        layer l;

        ggml_init_params ip = {
            ggml_tensor_overhead()*2 + 1024,
            nullptr,
            true,
        };
        l.ctx = ggml_init(ip);

        l.k = ggml_new_tensor_4d(l.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_blocks, n_seqs);
        l.v = ggml_new_tensor_4d(l.ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_blocks, n_seqs);
        ggml_set_name(l.k, "nsa_summary_k");
        ggml_set_name(l.v, "nsa_summary_v");

        ggml_backend_buffer_type_t buft = model.select_buft(il);
        if (!buft) {
            buft = ggml_backend_cpu_buffer_type();
        }
        l.buf = ggml_backend_alloc_ctx_tensors_from_buft(l.ctx, buft);
        GGML_ASSERT(l.buf != nullptr);
        ggml_backend_buffer_clear(l.buf, 0);

        ids.push_back(il);
        layers.push_back(l);
    }
}

llama_memory_nsa_summary::~llama_memory_nsa_summary() {
    for (auto & l : layers) {
        if (l.buf) {
            ggml_backend_buffer_free(l.buf);
        }
        if (l.ctx) {
            ggml_free(l.ctx);
        }
    }
}

ggml_tensor * llama_memory_nsa_summary::get_k(uint32_t il) const {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == il) {
            return layers[i].k;
        }
    }
    return nullptr;
}

ggml_tensor * llama_memory_nsa_summary::get_v(uint32_t il) const {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == il) {
            return layers[i].v;
        }
    }
    return nullptr;
}

void llama_memory_nsa_summary::clear(bool data) {
    if (!data) {
        return;
    }
    for (auto & l : layers) {
        if (l.buf) {
            ggml_backend_buffer_clear(l.buf, 0);
        }
    }
}

size_t llama_memory_nsa_summary::size() const {
    size_t res = 0;
    for (const auto & l : layers) {
        if (l.buf) {
            res += ggml_backend_buffer_get_size(l.buf);
        }
    }
    return res;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_nsa_summary::breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & l : layers) {
        if (l.buf) {
            res[ggml_backend_buffer_get_type(l.buf)] += ggml_backend_buffer_get_size(l.buf);
        }
    }
    return res;
}

void llama_memory_nsa_summary::state_write(llama_io_write_i & io) const {
    for (const auto & l : layers) {
        if (!l.buf) {
            continue;
        }
        for (ggml_tensor * t : { l.k, l.v }) {
            std::vector<char> host(ggml_nbytes(t));
            ggml_backend_tensor_get(t, host.data(), 0, host.size());
            io.write(host.data(), host.size());
        }
    }
}

void llama_memory_nsa_summary::state_read(llama_io_read_i & io) {
    for (auto & l : layers) {
        if (!l.buf) {
            continue;
        }
        for (ggml_tensor * t : { l.k, l.v }) {
            std::vector<char> host(ggml_nbytes(t));
            io.read(host.data(), host.size());
            ggml_backend_tensor_set(t, host.data(), 0, host.size());
        }
    }
}
