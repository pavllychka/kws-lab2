import argparse
import time

import numpy as np
import requests


def stats(name, values):
    print(f"{name}: mean={np.mean(values):.2f} ms, "
          f"p50={np.percentile(values, 50):.2f} ms, "
          f"p95={np.percentile(values, 95):.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--file", default="samples/yes.wav")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    with open(args.file, "rb") as f:
        audio = f.read()

    session = requests.Session()
    server, client = [], []
    for i in range(args.warmup + args.n):
        t0 = time.perf_counter()
        r = session.post(args.url, files={"file": ("a.wav", audio, "audio/wav")})
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        if i >= args.warmup:
            client.append(dt)
            server.append(r.json()["latency_ms"])
    stats("Latency (сервер)", server)
    stats("Latency (клієнт)", client)


if __name__ == "__main__":
    main()