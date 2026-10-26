#include "models.h"

#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_hybrid_gated_deltanet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE, hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, hparams.indexer_local_blocks);
    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, true);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer_all, true);

    if (hparams.ssm_d_conv != 4 || hparams.ssm_d_state != 128 || hparams.ssm_n_group != 8 ||
        hparams.ssm_dt_rank != 8 || hparams.indexer_n_head != 2 || hparams.indexer_head_size != 64 ||
        hparams.indexer_top_k != 16 || hparams.indexer_block_size != 64 || hparams.indexer_local_blocks != 8) {
        throw std::runtime_error("unsupported hybrid-gated-deltanet NSA parameters");
    }
}

void llama_model_hybrid_gated_deltanet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", il), { n_embd, n_ff }, 0);

        if (hparams.is_recr(il)) {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", il), { n_embd, 3*n_embd }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_embd }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, 3*n_embd }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", il), { 8 }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, il), { 8 }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, 8 }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), { n_embd, 8 }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), { 128 }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", il), { n_embd, n_embd }, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, 32*64 }, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", il), { n_embd, 2*64 }, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", il), { n_embd, 2*64 }, 0);
            layer.wg = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, 3*32 }, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { 32*64, n_embd }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_hybrid_gated_deltanet::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_hybrid_gated_deltanet::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * pos = build_inp_pos();
    ggml_tensor * out_ids = build_inp_out_ids();
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = cur;
        ggml_tensor * residual = cur;
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = hparams.is_recr(il) ? build_gdn(inp->get_recr(), cur, il) : build_nsa(inp->get_attn(), cur, pos, il);
        if (il == n_layer - 1 && out_ids) {
            cur = ggml_get_rows(ctx0, cur, out_ids);
            residual = ggml_get_rows(ctx0, residual, out_ids);
        }
        cur = ggml_add(ctx0, cur, residual);
        residual = cur;
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur, model.layers[il].ffn_up, nullptr, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, cur, residual);
        cur = build_cvec(cur, il);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    res->t_logits = build_lora_mm(model.output, cur, model.output_s);
    ggml_build_forward_expand(gf, res->t_logits);
}

ggml_tensor * llama_model_hybrid_gated_deltanet::graph::build_nsa(
        llm_graph_input_attn_kv * inp, ggml_tensor * cur, ggml_tensor * pos, int il) {
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    GGML_ASSERT(ubatch.equal_seqs() && ubatch.n_tokens == nt*ns);
    ggml_tensor * q = ggml_reshape_4d(ctx0, build_lora_mm(model.layers[il].wq, cur), 64, 32, nt, ns);
    ggml_tensor * k = ggml_reshape_4d(ctx0, build_lora_mm(model.layers[il].wk, cur), 64, 2, nt, ns);
    ggml_tensor * v = ggml_reshape_4d(ctx0, build_lora_mm(model.layers[il].wv, cur), 64, 2, nt, ns);
    q = ggml_rope_ext(ctx0, q, pos, nullptr, 64, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx0, k, pos, nullptr, 64, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, inp->mctx->cpy_k(ctx0, k, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, inp->mctx->cpy_v(ctx0, v, inp->get_v_idxs(), il));
    ggml_tensor * ck = inp->mctx->get_k(ctx0, il);
    ggml_tensor * cv = inp->mctx->get_v(ctx0, il);
    ggml_tensor * mask = inp->get_kq_mask();
    const int64_t block_size = hparams.indexer_block_size;
    const int64_t n_blocks = ck->ne[2]/block_size;
    // summaries only cover complete blocks; the raw cache stays whole so that the
    // selected and local branches can still see the tail of the current block
    ggml_tensor * ck_blocks = ggml_view_4d(ctx0, ck, 64, 2, n_blocks*block_size, ck->ne[3], ck->nb[1], ck->nb[2], ck->nb[3], 0);
    ggml_tensor * cv_blocks = ggml_view_4d(ctx0, cv, 64, 2, n_blocks*block_size, cv->ne[3], cv->nb[1], cv->nb[2], cv->nb[3], 0);
    ggml_tensor * ks = ggml_nsa_block_summary(ctx0, ck_blocks, block_size);
    ggml_tensor * vs = ggml_nsa_block_summary(ctx0, cv_blocks, block_size);
    ggml_tensor * selected = ggml_nsa_select_blocks(ctx0, q, ks, mask, block_size, hparams.indexer_top_k);
    const int64_t local_size = hparams.indexer_local_blocks*block_size;
    ggml_tensor * branches = ggml_nsa_attention(ctx0, q, ck, cv, ks, vs, selected, mask, block_size, local_size);
    ggml_tensor * gates = ggml_reshape_4d(ctx0, build_lora_mm(model.layers[il].wg, cur), 3, 32, nt, ns);
    ggml_tensor * out = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 64, 32, nt, ns);
    out = ggml_fill(ctx0, out, 0.0f);
    for (int branch = 0; branch < 3; ++branch) {
        ggml_tensor * value = ggml_view_4d(ctx0, branches, 64, 32, nt, ns, branches->nb[1], branches->nb[2], branches->nb[3], branch*nt*branches->nb[2]);
        ggml_tensor * gate = ggml_view_4d(ctx0, gates, 1, 32, nt, ns, gates->nb[1], gates->nb[2], gates->nb[3], branch*gates->nb[0]);
        out = ggml_add(ctx0, out, ggml_mul(ctx0, value, ggml_sigmoid(ctx0, gate)));
    }
    out = ggml_reshape_3d(ctx0, out, 64*32, nt, ns);
    out = ggml_reshape_2d(ctx0, build_lora_mm(model.layers[il].wo, out, model.layers[il].wo_s), hparams.n_embd, nt*ns);
    return out;
}

ggml_tensor * llama_model_hybrid_gated_deltanet::graph::build_gdn(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    GGML_ASSERT(ubatch.equal_seqs() && ubatch.n_tokens == nt*ns);
    GGML_ASSERT(model.layers[il].wqkv->ne[0] == cur->ne[0]);
    GGML_ASSERT(model.layers[il].wqkv_gate->ne[0] == cur->ne[0]);
    GGML_ASSERT(model.layers[il].ssm_alpha->ne[0] == cur->ne[0]);
    GGML_ASSERT(model.layers[il].ssm_beta->ne[0] == cur->ne[0]);
    ggml_tensor * mixed = build_lora_mm(model.layers[il].wqkv, cur);
    mixed = ggml_reshape_3d(ctx0, mixed, 3*1024, nt, ns);
    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, cur);
    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, build_lora_mm(model.layers[il].ssm_beta, cur), 1, 8, nt, ns));
    ggml_tensor * alpha = ggml_reshape_3d(ctx0, build_lora_mm(model.layers[il].ssm_alpha, cur), 8, nt, ns);
    ggml_tensor * gate = ggml_reshape_4d(ctx0, ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt)), model.layers[il].ssm_a), 1, 8, nt, ns);
    ggml_tensor * conv = build_conv_state(inp, inp->mctx->get_r_l(il), mixed, 4, 3072, il);
    conv = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv, model.layers[il].ssm_conv1d));
    const int64_t row = ggml_row_size(conv->type, 3072);
    ggml_tensor * q = ggml_l2_norm(ctx0, ggml_view_4d(ctx0, conv, 128, 8, nt, ns, ggml_row_size(conv->type, 128), row, row*nt, 0), hparams.f_norm_rms_eps);
    ggml_tensor * k = ggml_l2_norm(ctx0, ggml_view_4d(ctx0, conv, 128, 8, nt, ns, ggml_row_size(conv->type, 128), row, row*nt, 1024*ggml_element_size(conv)), hparams.f_norm_rms_eps);
    ggml_tensor * v = ggml_view_4d(ctx0, conv, 128, 8, nt, ns, ggml_row_size(conv->type, 128), row, row*nt, 2048*ggml_element_size(conv));
    ggml_tensor * state = build_rs(inp, inp->mctx->get_s_l(il), hparams.n_embd_s(), ns);
    state = ggml_reshape_4d(ctx0, state, 128, 128, 8, ns);
    ggml_tensor * output = build_recurrent_attn(inp, inp->mctx->get_s_l(il), q, k, v, gate, beta, state, il);
    output = build_norm(output, model.layers[il].ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, ggml_reshape_4d(ctx0, z, 128, 8, nt, ns)));
    output = ggml_reshape_3d(ctx0, output, 128*8, nt, ns);
    return ggml_reshape_2d(ctx0, build_lora_mm(model.layers[il].ssm_out, output, model.layers[il].ssm_out_s), hparams.n_embd, nt*ns);
}
