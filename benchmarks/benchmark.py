#!/usr/bin/env python3

import os
import sys
import time
import grpc
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PYGRPC_DIR = os.path.join(ROOT_DIR, "pygrpc")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PYGRPC_DIR)

import pygrpc.slm_pb2 as slm_pb2
import pygrpc.slm_pb2_grpc as slm_pb2_grpc

SERVER_ADDR = "localhost:50051"
PROMPT = "Explain transformers simply."
MAX_NEW_TOKENS = 64

WARMUP_REQUESTS = 2
TOTAL_REQUESTS = 40
WORKERS = 4
TIMEOUT_SECONDS = 3.0

def percentile(arr, p):
    return float(np.percentile(arr, p))

def sendRequest(stub):
    req = slm_pb2.GenerateRequest(
        prompt=PROMPT,
        maxNew=MAX_NEW_TOKENS
    )

    t0 = time.perf_counter()
    stub.generate(req, timeout=TIMEOUT_SECONDS)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0

def main():
    print("Connecting to:", SERVER_ADDR)

    channel = grpc.insecure_channel(SERVER_ADDR)
    stub = slm_pb2_grpc.SlmServiceStub(channel)

    print("Running", WARMUP_REQUESTS, "warmup requests...")
    for i in range(WARMUP_REQUESTS):
        try:
            sendRequest(stub)
        except Exception as e:
            print("Warmup failed:", e)

    latencies = []
    failures = 0

    print("Running", TOTAL_REQUESTS, "requests with", WORKERS, "workers...")

    startWall = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(sendRequest, stub)
                   for i in range(TOTAL_REQUESTS)]

        for f in tqdm(as_completed(futures), total=TOTAL_REQUESTS):
            try:
                latencies.append(f.result())
            except Exception as e:
                print("Request failed:", e)
                failures += 1

    totalTime = time.time() - startWall
    channel.close()

    if not latencies:
        print("No successful requests.")
        return

    lat = np.array(latencies)

    print("\nBENCHMARK RESULTS")
    print("Server:", SERVER_ADDR)
    print("Workers:", WORKERS)
    print("Successful:", len(lat))
    print("Failed:", failures)
    print("Total time:", f"{totalTime:.2f}s")
    print("Throughput:", f"{len(lat)/totalTime:.2f} req/s")
    print("Mean latency:", f"{lat.mean():.2f} ms")
    print("Median latency:", f"{percentile(lat,50):.2f} ms")
    print("p90 latency:", f"{percentile(lat,90):.2f} ms")
    print("p95 latency:", f"{percentile(lat,95):.2f} ms")
    print("p99 latency:", f"{percentile(lat,99):.2f} ms")
    print("Min latency:", f"{lat.min():.2f} ms")
    print("Max latency:", f"{lat.max():.2f} ms")
    print("Std dev:", f"{lat.std():.2f} ms")

if __name__ == "__main__":
    main()
