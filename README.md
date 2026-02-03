# TinyLM – Small Language Model from Scratch (C++ Inference & gRPC Serving)

TinyLM is a small, GPT-style language model built end-to-end to explore **ML systems, inference performance, and low-latency generation** on consumer hardware.

The project focuses on **understanding and implementing the full stack** of a language model—training, tokenization, weight export, C++ inference, and serving—rather than maximizing raw model quality.

---

## Overview

This project includes:

- Custom SentencePiece tokenizer (BPE)
- Transformer decoder-only language model trained from scratch
- RMSNorm-based architecture
- KV-cache autoregressive inference
- Nucleus (top-p) sampling with repetition penalty
- Quantized weights for faster CPU inference
- C++ inference engine for low-latency generation
- Binary weight export/import between Python and C++
- gRPC inference server
- Concurrent Python benchmarking client
- Multi-replica inference pool for throughput scaling

The model achieves fluent English generation comparable to early GPT-1–era models while remaining lightweight and fast.

Due to hardware limits (2020 MacBook Pro, CPU-only), model scale and training duration are constrained. Higher-quality long-form outputs (stories, essays) would require larger models, more training data, and GPU acceleration.

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

1. Prepare dataset

```bash
python3 prepare.py
```

2. Train SentencePiece tokenizer

```bash
python3 trainTokenizer.py
```

3. Encode dataset

```bash
python3 train.py
```

4. Train language model in PyTorch  
5. Export weights as raw binary files  
6. Load weights in C++ inference engine  

Training uses cross-entropy loss with causal masking.

---

## C++ Inference Engine

Features:

- Autoregressive decoding  
- KV-cache for O(1) per-token generation  
- Quantized weight loading  
- Temperature + nucleus sampling  
- Repetition penalty  

Example:

```bash
./SLM "Explain transformers simply." 64
```

---

## gRPC Inference Server

The project includes a C++ gRPC server that exposes the model as an inference service.

Architecture:

```
Client
  → gRPC Frontend
      → Round-robin Dispatcher
          → tinyLM Replica 0
          → tinyLM Replica 1
          → tinyLM Replica 2
          → tinyLM Replica 3
```

Each replica maintains its own weights and KV-cache, allowing true parallel inference.

Start server:

```bash
./slm
```

---

## Benchmarking

A Python benchmarking client measures:

- End-to-end latency  
- Tail latency (p90 / p95 / p99)  
- Throughput (requests/sec)  
- Scaling behavior under concurrency  

Run benchmark:

```bash
cd benchmarks
python3 benchmark.py
```

Example result (CPU, 4 replicas, 4 workers):

```
Throughput: 4.21 req/s
Mean latency: 948 ms
p99 latency: 979 ms
```

This demonstrates near-linear throughput scaling with replica count while maintaining sub-1s median latency.

---

## Performance Characteristics

- CPU-only inference  
- ~64 tokens/sec per replica  
- KV-cache reduces per-token compute  
- Quantization improves cache locality and speed  
- gRPC overhead is negligible compared to model compute  

---

## Project Goals

- Learn how modern language models work internally  
- Understand inference-time bottlenecks  
- Build practical ML systems infrastructure  
- Explore tradeoffs between latency, throughput, and model size  

---

## Future Improvements
   
- ONNX Runtime 
- Dockerized deployment  
- Visualization of benchmark histograms  
- Web dashboard for live performance metrics  

---

## Summary

TinyLM is an end-to-end small language model and inference system demonstrating:

- Model training from scratch  
- Efficient C++ inference  
- Quantization and KV-cache optimizations  
- gRPC-based serving  
- Concurrent load testing  
- Multi-replica scaling  

The project emphasizes **systems-level understanding** of language model inference rather than raw model scale.
