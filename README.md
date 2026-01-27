# TinyLM – Small Language Model from Scratch (C++ Inference)

TinyLM is a small, GPT-style language model built end-to-end to explore
ML systems, inference performance, and low-latency generation on consumer
hardware.

The project focuses on **understanding and implementing the full stack**
of a language model rather than maximizing raw model quality.

---

## Overview

This project includes:

- Custom SentencePiece tokenizer (BPE)
- Transformer language model trained from scratch
- RMSNorm-based architecture
- KV-cache autoregressive inference
- Nucleus (top-p) sampling with repetition penalty
- C++ inference engine for low-latency generation
- Binary weight export/import between Python and C++

The model achieves fluent English generation comparable to early GPT-1–era
models, while remaining lightweight and fast. The model is fast but stories and essays would require 
higher compute to have proper quality. Currently, my 2020 Macbook Pro has reached limits and better training
would require increased disk space, gpu, and/or longer training periods. 

---

## Model Architecture

- Transformer decoder-only LM
- Hidden size: 192
- Layers: 16
- Vocabulary size: 4096 (SentencePiece BPE)
- Context length: up to 256 tokens
- Activation: GELU
- Normalization: RMSNorm

---

## Training Pipeline

1. Prepare dataset (`data/train.txt`) by running python3 prepare.py
2. Train SentencePiece tokenizer using trainTokenizer.py
3. Encode dataset into `tokens.bin` using train.py
4. Train language model in PyTorch
5. Export weights as raw binary files
6. Load weights in C++ inference engine

Training uses cross-entropy loss with causal masking.

---

## Inference

The C++ inference engine supports:

- Autoregressive decoding
- KV-cache for O(1) per-token generation
- Temperature + nucleus sampling
- Repetition penalty to reduce loops

Example usage:

```bash
./SLM "prompt" tokens
