// Operator tests for the native sparse attention ggml ops.
// Runs on the CPU backend and, when available, on a GPU backend (Metal here).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int D = 64;
static constexpr int HQ = 32;
static constexpr int HKV = 2;
static constexpr int BS = 64;
static constexpr int T = 128;
static constexpr int NKV = 128;

static float dot(const float * a, const float * b) {
    float s = 0.0f;
    for (int i = 0; i < D; ++i) s += a[i] * b[i];
    return s;
}

static void softmax(const std::vector<float> & s, std::vector<float> & p) {
    const float m = *std::max_element(s.begin(), s.end());
    float sum = 0.0f;
    p.resize(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        p[i] = expf(s[i] - m);
        sum += p[i];
    }
    for (float & v : p) v /= sum;
}

struct case_data {
    std::vector<float> q, k, v, ks, vs, mask;
};

static case_data make_case(std::mt19937 & rng) {
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    case_data c;
    c.q.resize((size_t) D * HQ * T);
    c.k.resize((size_t) D * HKV * NKV);
    c.v.resize((size_t) D * HKV * NKV);
    c.ks.resize((size_t) D * HKV * (NKV / BS));
    c.vs.resize((size_t) D * HKV * (NKV / BS));
    c.mask.assign((size_t) NKV * T, -INFINITY);
    for (float & x : c.q) x = dis(rng);
    for (float & x : c.k) x = dis(rng);
    for (float & x : c.v) x = dis(rng);
    for (int b = 0; b < NKV / BS; ++b) {
        for (int h = 0; h < HKV; ++h) {
            for (int d = 0; d < D; ++d) {
                float sk = 0.0f, sv = 0.0f;
                for (int j = 0; j < BS; ++j) {
                    const int t = b * BS + j;
                    sk += c.k[(size_t) d + D * ((size_t) h + HKV * t)];
                    sv += c.v[(size_t) d + D * ((size_t) h + HKV * t)];
                }
                c.ks[(size_t) d + D * ((size_t) h + HKV * b)] = sk / BS;
                c.vs[(size_t) d + D * ((size_t) h + HKV * b)] = sv / BS;
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j <= t; ++j) {
            c.mask[(size_t) j + (size_t) NKV * t] = 0.0f;
        }
    }
    return c;
}

struct result {
    std::vector<int32_t> sel;
    std::vector<float> out;
};

static result run(const case_data & c, ggml_backend_dev_t dev) {
    struct ggml_init_params ip = { 256u*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * tq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HQ, T, 1);
    struct ggml_tensor * tk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HKV, NKV, 1);
    struct ggml_tensor * tv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HKV, NKV, 1);
    struct ggml_tensor * tks = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HKV, NKV/BS, 1);
    struct ggml_tensor * tvs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HKV, NKV/BS, 1);
    struct ggml_tensor * tm  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NKV, T, 1, 1);

    struct ggml_tensor * sel = ggml_nsa_select_blocks(ctx, tq, tks, tm, BS, 2);
    struct ggml_tensor * out = ggml_nsa_attention(ctx, tq, tk, tv, tks, tvs, sel, tm, BS, 8);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sel);
    ggml_build_forward_expand(gf, out);

    struct ggml_backend * be = ggml_backend_dev_init(dev, nullptr);
    struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    (void) buf;

    ggml_backend_tensor_set(tq,  c.q.data(),    0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk,  c.k.data(),    0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv,  c.v.data(),    0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tks, c.ks.data(),   0, ggml_nbytes(tks));
    ggml_backend_tensor_set(tvs, c.vs.data(),   0, ggml_nbytes(tvs));
    ggml_backend_tensor_set(tm,  c.mask.data(), 0, ggml_nbytes(tm));

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed on %s\n", ggml_backend_dev_name(dev));
    }

    result r;
    r.sel.resize((size_t) 2 * HKV * T);
    r.out.resize((size_t) D * HQ * 3 * T);
    ggml_backend_tensor_get(sel, r.sel.data(), 0, ggml_nbytes(sel));
    ggml_backend_tensor_get(out, r.out.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return r;
}

int main() {
    std::mt19937 rng(1234);
    const case_data c = make_case(rng);

    ggml_backend_load_all();
    ggml_backend_dev_t cpu  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

    const result r_cpu = run(c, cpu);

    int fails = 0;
    int checks = 0;

    for (int t = 0; t < T; ++t) {
        const int ib_q = t / BS;
        for (int h = 0; h < HKV; ++h) {
            bool has_q = false;
            for (int s = 0; s < 2; ++s) {
                const int ib = r_cpu.sel[(size_t) s + 2*((size_t) h + HKV*t)];
                if (ib < 0) continue;
                checks++;
                if (ib > ib_q) {
                    printf("FAIL: future block %d at t=%d\n", ib, t);
                    fails++;
                }
                if (ib == ib_q) has_q = true;
            }
            if (!has_q) {
                printf("FAIL: current block missing at t=%d h=%d\n", t, h);
                fails++;
            }
        }
    }

    std::vector<float> ref((size_t) D * HQ * 3 * T, 0.0f);
    const float scale = 1.0f / sqrtf((float) D);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < HQ; ++h) {
            const int hk = h / (HQ/HKV);
            auto qv = c.q.data() + (size_t) D*h + (size_t) D*HQ*t;
            for (int br = 0; br < 3; ++br) {
                std::vector<int> idx;
                std::vector<float> scores;
                if (br == 0) {
                    const int nb = (t + 1) / BS;
                    for (int b = 0; b < nb; ++b) {
                        idx.push_back(b);
                        scores.push_back(dot(qv, c.ks.data() + (size_t) D*hk + (size_t) D*HKV*b) * scale);
                    }
                } else if (br == 1) {
                    for (int s = 0; s < 2; ++s) {
                        const int ib = r_cpu.sel[(size_t) s + 2*((size_t) hk + HKV*t)];
                        if (ib < 0) continue;
                        for (int j = 0; j < BS; ++j) {
                            const int tk_ = ib*BS + j;
                            if (tk_ > t) break;
                            idx.push_back(tk_);
                            scores.push_back(dot(qv, c.k.data() + (size_t) D*hk + (size_t) D*HKV*tk_) * scale);
                        }
                    }
                } else {
                    for (int tk_ = std::max(0, t - 7); tk_ <= t; ++tk_) {
                        idx.push_back(tk_);
                        scores.push_back(dot(qv, c.k.data() + (size_t) D*hk + (size_t) D*HKV*tk_) * scale);
                    }
                }
                std::vector<float> p;
                softmax(scores, p);
                float * dst = ref.data() + (size_t) D*h + (size_t) D*HQ*(br*T + t);
                for (size_t i = 0; i < idx.size(); ++i) {
                    const int j = idx[i];
                    const float * vv = (br == 0 ? c.vs.data() : c.v.data()) + (size_t) D*hk + (size_t) D*HKV*j;
                    for (int d = 0; d < D; ++d) dst[d] += p[i] * vv[d];
                }
            }
        }
    }

    auto compare = [&](const result & r, const char * name) {
        double max_diff = 0.0;
        for (int br = 0; br < 3; ++br) {
            for (int t = 0; t < T; ++t) {
                for (int h = 0; h < HQ; ++h) {
                    const float * a = r.out.data() + (size_t) D*h + (size_t) D*HQ*(br*T + t);
                    const float * b = ref.data()   + (size_t) D*h + (size_t) D*HQ*(br*T + t);
                    for (int d = 0; d < D; ++d) max_diff = std::max(max_diff, (double) fabsf(a[d] - b[d]));
                }
            }
        }
        printf("%s: attention max diff %.6g\n", name, max_diff);
        return max_diff;
    };

    const double d_cpu = compare(r_cpu, "cpu");
    if (d_cpu > 1e-5) fails++;

    if (gpu) {
        const result r_gpu = run(c, gpu);
        const double d_gpu = compare(r_gpu, ggml_backend_dev_name(gpu));
        if (d_gpu > 1e-4) fails++;
    } else {
        printf("no GPU backend, metal check skipped\n");
    }

    printf("selection checks: %d, fails: %d\n", checks, fails);
    if (fails) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
