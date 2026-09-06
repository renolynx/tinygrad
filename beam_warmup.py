#!/usr/bin/env python3
"""Pre-populate the BEAM search cache so interactive runs never pay the search cost.

Run once (it takes a while); results persist in ~/Library/Caches/tinygrad/cache.db.
Edit SHAPES to match the workloads you actually run.
"""
import os, time
os.environ.setdefault("DEV", "NV")
os.environ["BEAM"] = "2"

from tinygrad import Tensor, Device

MATMUL_SIZES = [512, 1024, 2048, 4096, 8192]
CONV_SHAPES = [(32, 3, 224, 224, 64), (32, 64, 56, 56, 128), (32, 128, 28, 28, 256)]

def warm(label, fn):
    st = time.perf_counter()
    fn()
    Device[Device.DEFAULT].synchronize()
    print(f"{label:<34} {time.perf_counter()-st:7.1f}s", flush=True)

if __name__ == "__main__":
    print(f"warming BEAM cache on {Device.DEFAULT}\n")
    total = time.perf_counter()

    for n in MATMUL_SIZES:
        a, b = Tensor.rand(n, n).realize(), Tensor.rand(n, n).realize()
        warm(f"matmul {n}x{n}", lambda: (a @ b).realize())

    for bs, cin, h, w, cout in CONV_SHAPES:
        x = Tensor.rand(bs, cin, h, w).realize()
        k = Tensor.rand(cout, cin, 3, 3).realize()
        warm(f"conv {bs}x{cin}x{h}x{w}->{cout}", lambda: x.conv2d(k, padding=1).realize())

    print(f"\ndone in {time.perf_counter()-total:.1f}s")
