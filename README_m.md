# Gated DeltaNet + NSA in qvac-fabric-llm.cpp

This is my work on native support for `Idiap/gated-deltanet-nsa-0.4B-10B`.
Inference is plain C++/ggml. Python is only used for conversion and for the
reference check.

## What is in the tree

- `conversion/hybrid_gated_deltanet.py` converts the HF checkpoint to GGUF.
- `src/models/hybrid-gated-deltanet.cpp` loads the model and builds the graph.
- `src/llama-arch.*`, `src/llama-model.cpp`, `src/llama-context.cpp` register
  the architecture and the hybrid memory.
- `ggml/include/ggml.h`, `ggml/src/ggml.c`,
  `ggml/src/ggml-cpu/ops.cpp` add four new ops:
  `NSA_BLOCK_SUMMARY`, `NSA_SUMMARY_STORE`, `NSA_SELECT_BLOCKS`, `NSA_ATTENTION`.
- `ggml/src/ggml-metal/*` implements the same ops on Metal. The attention and
  summary-store kernels use one threadgroup per row with 32 lanes.
- `ggml/src/ggml-vulkan/*` implements the same ops on Vulkan (f32 K/V).
- `src/llama-memory-nsa-summary.*` and `src/llama-memory-hybrid-nsa.*` keep
  the per-block mean K/V summary buffer.
- `tests/test-gdn-op.cpp`, `tests/test-nsa-ops.cpp`, `tests/test-nsa-model.cpp`
  and `tests/nsa_reference.py` are the checks. `tests/test-backend-ops.cpp`
  has an `NSA_ATTENTION` case that runs on all backends.
- `tests/run-nsa-checks.sh` runs everything.

I also fixed the tokenizer prefix for this checkpoint. The model uses a
sentencepiece-style BPE, so `tokenizer.ggml.add_space_prefix=true` is written
to the GGUF and the BPE path in `src/llama-vocab.cpp` now uses it. After this
change the token IDs match the HF `LlamaTokenizer` for the strings I tried.
The model graph now derives head counts, head sizes, inner size, conv size and
RoPE length from hparams. The only literals left are the early validation
checks in `load_arch_hparams`, so a different checkpoint is rejected instead of
silently using the wrong shape.

## Incremental block summaries

Every NSA layer owns a small `[64, n_kv_head, n_blocks, n_seqs]` buffer with
mean K/V per completed 64-token block. The graph calls
`ggml_nsa_summary_store` with a two-value block range that `set_input` updates
from the ubatch position. When a block becomes complete, only that block is
pooled and written. The block range input makes this work with graph reuse:
the graph is built once and the range changes per token. The summary store is
on CPU, Metal and Vulkan. This removes the previous O(context) re-pooling on
every graph.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j
```

Vulkan (on macOS this needs MoltenVK, vulkan-loader and shaderc/glslang):

```bash
cmake -S . -B build-vk \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-vk -j
```

## Convert

```bash
uv run --project . python convert_hf_to_gguf.py \
  gated-deltanet-nsa-0.4B-10B \
  --outtype f32 \
  --outfile gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf
```

The checkpoint is already in `gated-deltanet-nsa-0.4B-10B/` and the GGUF is
about 1.33 GiB.

## Run

CPU:

```bash
build/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256
```

Metal:

```bash
build/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ot token_embd.weight=MTL0
```

The `-ot` line puts the input embedding on Metal too. Without it, llama.cpp
keeps the tiny `GET_ROWS` embedding node on the CPU/BLAS backend. That is the
usual llama.cpp behaviour, not a fallback of the NSA ops.

Example CPU output:

```
 The capital of France is Paris. It is located in the north of the country.
```

Example Metal output:

```
 The capital of France is Paris. It is located in the northwest of the country.
```

Vulkan:

```bash
build-vk/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ctk f32 -ctv f32 -fa off -ot token_embd.weight=Vulkan0
```

The Vulkan NSA shaders are f32 K/V and f32 mask only. `-fa off` keeps the mask
f32; `-ctk f32 -ctv f32` keeps the cache f32. With the default f16 cache the
Vulkan shaders are not used, so the command above is the tested one.

## Tests

```bash
build/bin/test-gdn-op
build/bin/test-nsa-ops
NSA_TEST_MODEL=.../model-f32.gguf build/bin/test-nsa-model /tmp/nsa.bin
NSA_TEST_GPU_LAYERS=99 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu.bin
NSA_TEST_GPU_LAYERS=99 NSA_TEST_KV_F32=1 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu_f32.bin
build/bin/test-backend-ops -o NSA_ATTENTION
build/bin/test-backend-ops -o NSA_BLOCK_SUMMARY
uv run --project . python tests/nsa_reference.py \
  --model gated-deltanet-nsa-0.4B-10B \
  --prompt "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum." \
  --compare /tmp/nsa.bin
```

`test-gdn-op` compares the Gated DeltaNet op with a naive recurrence for 1, 64,
128, 192, 193, 256 and 512 tokens. `test-nsa-ops` builds a 128-token causal
case, runs the three ops, checks that selection never picks a future block, and
compares the three attention branches with a small C++ reference.
`test-nsa-model` runs the whole model and compares prompt processing with
token-by-token generation. `test-backend-ops -o NSA_ATTENTION` runs the
selection + attention graph on CPU, Metal and Vulkan;
`-o NSA_BLOCK_SUMMARY` covers the pooling op. On this machine both pass on
Metal and Vulkan (3/3 and 2/2 cases).

## Results

Benchmarks on an Apple M4, 6 threads, f32 weights. Metal uses f16 KV;
Vulkan uses f32 KV and `-fa off` because its NSA shaders are f32 only.

| backend | pp128 | tg32 | RSS |
|---|---:|---:|---:|
| CPU | 441 t/s | 63.0 t/s | ~1.5 GB |
| Metal | 2066 t/s | 58.1 t/s | ~1.5 GB |
| Vulkan | 188 t/s | 20.2 t/s | ~1.5 GB |

The model is 358M params, 1.33 GiB in f32. Q8_0 is 365 MiB and also gives
coherent output on CPU and Metal. I did not benchmark Q8_0.

Metal is much faster for prompt processing now that attention and summary
store use 32-lane threadgroups. Token generation is roughly on par with CPU.
Selection is still one thread per row and is the main Metal gap. Vulkan on
macOS uses MoltenVK and the shaders are deliberately simple, so it is the
slowest path.

Metal assignment check:

```bash
GGML_SCHED_DEBUG=2 build/bin/llama-completion \
  -m ... -p "The capital of France is" -n 1 -ngl 99 \
  -ot token_embd.weight=MTL0 -v 2>&1 | grep NSA_
```

The log shows `NSA_BLOCK_`, `NSA_SUMMARY`, `NSA_SELECT_` and `NSA_ATTENT_` on
`MTL0`, not on CPU. With `-ot token_embd.weight=MTL0` the scheduler reports
`graph splits = 1`. Without `-ot` the only CPU/BLAS node is the token embedding
`GET_ROWS`.

The same check with `-ngl 99 -ctk f32 -ctv f32 -fa off
-ot token_embd.weight=Vulkan0` shows the same four nodes on `Vulkan0` and
`graph splits = 1`.

## Correctness

Reference: I could not run the official `fla` kernels here because they need
Triton and a CUDA GPU. Instead I wrote `tests/nsa_reference.py`, a slow
PyTorch port of the fla math (GDN recurrence, RoPE, block mean pooling,
compressed/selected/local NSA branches, gates). It runs on CPU and is only
used for validation.

What I measured:

- Ops, 128 tokens: max abs diff vs the C++ reference is 2.1e-7 on CPU,
  2.4e-7 on Metal and Vulkan. 384 selection checks passed.
- End to end, 23 tokens: max logit diff 0.00166, mean 0.00016, top-1 23/23.
- End to end, 98 tokens: max logit diff 0.00496, mean 0.00023, top-1 98/98.
- Metal vs reference, 23 tokens: max 0.0101, mean 0.00080, top-1 23/23.
- Vulkan vs reference, 23 tokens: max 0.00804, mean 0.00077, top-1 23/23.
- GDN op, 1..512 tokens: max abs diff 4.1e-10 against the naive recurrence.
- Prefill vs token-by-token, CPU: max 9.8e-4 for 23 tokens, 1.8e-3 for 98,
  1.4e-3 for 512. Top-1 always matches.
- Prefill vs token-by-token, Metal: max 0.0095 for 23 tokens, 0.0134 for 512,
  top-1 0/512. The CPU path is the exact reference.
- Prefill vs token-by-token, Vulkan: max 0.0094 for 23 tokens, 1/98 top-1
  mismatch at 98 tokens. The CPU path is the exact reference.

My tolerances were: ops 1e-5 on CPU and 1e-4 on GPU; full model top-1 must
match, and the max logit difference should stay under 0.01 on GPU and 0.005 on
CPU for short contexts. These are not bit-exact checks. The remaining
differences come from f32/f16 rounding and from the different GDN kernels.

Two bugs found during validation:

1. The KV tensor is allocated for the full context. `n_kv` is not the number
   of used cells. The first version assumed the batch was at the tail of the
   cache, so for short prompts the NSA branches read padding and were zero.
   The ops now find the query cell by scanning the causal mask for the last
   visible cell. This was the real cause of the 193-token prefill/decode
   mismatch. The GDN op itself was fine: it passes the naive 1..512 token test
   and one-shot prefill matches decode up to 512 without any ubatch cap.
2. The tokenizer lost the leading `▁` because the BPE path did not apply
   `add_space_prefix`. Fixed in the converter and in `src/llama-vocab.cpp`.

## Known limitations

- Metal NSA kernels support f16/f32 K/V. Quantized KV cache is not supported
  by the Metal path. The CPU path can read quantized K/V.
- Vulkan NSA shaders support f32 K/V and f32 mask only. Use `-ctk f32 -ctv f32
  -fa off`; otherwise the scheduler falls back to CPU for these ops.
- Batched/multi-sequence prompt processing is not tested. The NSA graph
  assumes equal sequences and one stream.
- Selection is still one thread per row on Metal and Vulkan.
- No CUDA and no mobile work.
- The reference script is a PyTorch port, not the official fla kernels. The
  commands to run the official kernels on a CUDA machine are not included.
- The model is a base LM, so there is no chat template. Completion is the
  correct mode; a chat template would have to come from an instruct model.
- Summary state is append-only. `seq_rm` invalidates the summary buffer; state
  save/load is implemented but rollback and context shifting were not
  stress-tested.

## What I would do next

1. Optimize selection on Metal/Vulkan (shared-memory top-k).
2. Add f16 Vulkan shaders so the default KV cache works without `-fa off`.
3. Add a CUDA backend and run the official fla comparison there.
4. Test multi-sequence, state save/load, rollback and longer contexts.
5. Add f16/quantized KV coverage to the operator test.

## Short assessment

Working end to end: conversion, GGUF loading, CPU inference, Metal inference,
Vulkan inference, incremental summaries, Q8_0 inference, operator tests, model
prefill/decode tests, and a reference comparison. The new ops pass the
project `test-backend-ops -o NSA_ATTENTION` on Metal and Vulkan.

The GDN op is native and tested; there is no ubatch workaround and the model
dims are derived from hparams. The main weaknesses are the scalar selection
kernels, the f32-only Vulkan shaders, the single-stream test coverage, and the
lack of CUDA/mobile support. The reference check uses a PyTorch port because
the official fla kernels do not run on this machine.
