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
  `ggml/src/ggml-cpu/ops.cpp` add three new ops:
  `NSA_BLOCK_SUMMARY`, `NSA_SELECT_BLOCKS`, `NSA_ATTENTION`.
- `ggml/src/ggml-metal/*` implements the same three ops on Metal.
- `tests/test-nsa-ops.cpp`, `tests/test-nsa-model.cpp` and
  `tests/nsa_reference.py` are the checks.
- `tests/run-nsa-checks.sh` runs everything.

I also fixed the tokenizer prefix for this checkpoint. The model uses a
sentencepiece-style BPE, so `tokenizer.ggml.add_space_prefix=true` is written
to the GGUF and the BPE path in `src/llama-vocab.cpp` now uses it. After this
change the token IDs match the HF `LlamaTokenizer` for the strings I tried.

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
  -p "The capital of France is" -n 32 -c 256 -ngl 99
```

Example CPU output:

```
 The capital of France is Paris. It is located in the north of the country.
```

Example Metal output:

```
 The capital of France is Paris. It is located in the northwest of the country.
```

## Tests

```bash
build/bin/test-nsa-ops
NSA_TEST_MODEL=.../model-f32.gguf build/bin/test-nsa-model /tmp/nsa.bin
NSA_TEST_GPU_LAYERS=99 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu.bin
uv run --project . python tests/nsa_reference.py \
  --model gated-deltanet-nsa-0.4B-10B \
  --prompt "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum." \
  --compare /tmp/nsa.bin
```

`test-nsa-ops` builds a 128-token causal case, runs the three ops, checks that
selection never picks a future block, and compares the three attention
branches with a small C++ reference. `test-nsa-model` runs the whole model and
compares prompt processing with token-by-token generation.

## Results

Benchmarks on an Apple M4, 6 threads, f32 weights, f16 KV cache:

| backend | pp128 | tg32 | RSS |
|---|---:|---:|---:|
| CPU | 394 t/s | 39.6 t/s | 1.50 GB |
| Metal | 388 t/s | 32.0 t/s | 1.54 GB |

The model is 358M params, 1.33 GiB in f32. Q8_0 is 365 MiB and also gives
coherent output on CPU and Metal. I did not benchmark Q8_0.

Metal is not faster here. The NSA kernels are simple: one thread per row, no
tiling and no simd reductions. For a 358M model the launch and synchronization
cost is also large. I kept the simple kernels because they are correct and do
not fall back to CPU.

Metal assignment check:

```bash
GGML_SCHED_DEBUG=2 build/bin/llama-completion \
  -m ... -p "The capital of France is" -n 1 -ngl 99 -v 2>&1 | grep NSA_
```

The log shows `NSA_BLOCK_`, `NSA_SELECT_` and `NSA_ATTENT_` on `MTL0`, not on
CPU. The graph has two splits; they are not the NSA ops.

## Correctness

Reference: I could not run the official `fla` kernels here because they need
Triton and a CUDA GPU. Instead I wrote `tests/nsa_reference.py`, a slow
PyTorch port of the fla math (GDN recurrence, RoPE, block mean pooling,
compressed/selected/local NSA branches, gates). It runs on CPU and is only
used for validation.

What I measured:

- Ops, 128 tokens: max abs diff vs the C++ reference is 2.1e-7 on CPU and
  2.4e-7 on Metal. 384 selection checks passed.
- End to end, 23 tokens: max logit diff 0.00166, mean 0.00016, top-1 23/23.
- End to end, 98 tokens: max logit diff 0.00496, mean 0.00023, top-1 98/98.
- Metal vs reference, 23 tokens: max 0.00832, mean 0.00079, top-1 23/23.
- Prefill vs token-by-token, CPU: max 9.8e-4 for 23 tokens, 1.8e-3 for 98,
  1.5e-3 for 512. Top-1 always matches.
- Prefill vs token-by-token, Metal: max 7.7e-3 for 23 tokens. Top-1 matches.

My tolerances were: ops 1e-5 on CPU and 1e-4 on GPU; full model top-1 must
match, and the max logit difference should stay under 0.01 on GPU and 0.005 on
CPU for short contexts. These are not bit-exact checks. The remaining
differences come from f32/f16 rounding and from the different GDN kernels.

Two bugs found during validation:

1. The KV tensor is allocated for the full context. `n_kv` is not the number
   of used cells. The first version assumed the batch was at the tail of the
   cache, so for short prompts the NSA branches read padding and were zero.
   The ops now find the query cell by scanning the causal mask for the last
   visible cell.
2. The fused GDN chunk kernel in this fork returns a wrong final state when
   one ubatch contains more than one 64-token chunk. A 193-token prefill
   diverged from decode; 192 and 64-token chunks matched. I cap `n_ubatch` to
   64 for this architecture in `llama-context.cpp`. This is a workaround, not
   a fix in the GDN kernel.

## Known limitations

- `n_ubatch` is capped at 64 for this architecture because of the GDN chunk
  bug above. Larger ubatches are silently split by the scheduler.
- Metal NSA kernels only support f16/f32 K/V. Quantized KV cache is not
  supported by the Metal path. The CPU path can read quantized K/V.
- Batched/multi-sequence prompt processing is not tested. The NSA graph
  assumes equal sequences and one stream.
- Only Metal is implemented as a GPU backend. CUDA and Vulkan are not done.
- No mobile work.
- The reference script is a PyTorch port, not the official fla kernels. The
  commands to run the official kernels on a CUDA machine are not included.
- The model is a base LM and there is no chat template.
- The Metal implementation is correct but not tuned. CPU is faster for this
  model.

## What I would do next

1. Fix the GDN chunk state bug and remove the 64-token ubatch cap.
2. Add F16/quantized KV coverage to the ops test.
3. Optimize the Metal NSA kernels with threadgroup reductions and vector loads.
4. Add a CUDA backend and run the official fla comparison there.
5. Test multi-sequence and state save/load.

## Short assessment

Working end to end: conversion, GGUF loading, CPU inference, Metal inference
without a CPU fallback for the new ops, Q8_0 inference, operator tests, model
prefill/decode tests, and a reference comparison.

The main weakness is the GDN chunk workaround. The model also has only a
single-stream path tested, the Metal kernels are slow, and there is no CUDA,
Vulkan or mobile support. The reference check uses a PyTorch port because the
official fla kernels do not run on this machine.
