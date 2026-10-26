#!/usr/bin/env python3
# Compare the GGUF tokenizer with the HF LlamaTokenizer for this checkpoint.
# This is validation only; inference does not use Python.

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

STRINGS = [
    "The capital of France is",
    "don't",
    "Hello, world!",
    " leading space",
    "1234",
    "Привет мир",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("NSA_MODEL_DIR", "gated-deltanet-nsa-0.4B-10B"))
    ap.add_argument("--gguf", default=os.environ.get("NSA_GGUF", ""))
    ap.add_argument("--tokenize", default=os.environ.get("NSA_TOKENIZE", ""))
    args = ap.parse_args()

    tokenize = args.tokenize or shutil.which("llama-tokenize")
    if not tokenize:
        print("SKIP: llama-tokenize not found")
        return 0
    if not args.gguf or not os.path.exists(args.gguf):
        print("SKIP: NSA_GGUF not set or missing")
        return 0
    if not os.path.isdir(args.model):
        print("SKIP: model directory not found")
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    fails = 0
    for text in STRINGS:
        ref = tok(text)["input_ids"]
        out = subprocess.check_output(
            [tokenize, "-m", args.gguf, "-p", text, "--ids"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        match = re.search(r"\[([^\]]*)\]", out)
        if not match:
            print(f"FAIL: no ids for {text!r}")
            fails += 1
            continue
        got = [int(x) for x in match.group(1).split(",") if x.strip()]
        if got != ref:
            print(f"FAIL: {text!r}: GGUF {got} != HF {ref}")
            fails += 1

    if fails:
        print("FAILED")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
