"""Tile tuning on the target GPU (H200). Produces a table for SUPER_W4A8_CONFIG.

    python bench/tune.py --out h200_super.json [--Ms 1,8,32,128,512,2048,8192]

Weights are random NVFP4 at Super shapes (timing does not depend on values).
Each candidate is timed with CUDA events; FC1 and FC2 are tuned separately.
"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from super_nvfp4_w4a8.moe import moe_align, rowwise_quant, _gemm  # noqa: E402

E, H, I, TOPK = 512, 1024, 2688, 22


def rand_bank(N, K, dev):
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    s_rel = (torch.rand(E, N, K // 16, device=dev) * 60 + 4).to(torch.float8_e4m3fn)
    bscale = torch.full((E, N), 1e-3, device=dev)
    return packed, s_rel, bscale


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ms", default="1,8,32,128,512,2048,8192")
    a = ap.parse_args()
    dev = torch.device("cuda")
    w1, w2 = rand_bank(I, H, dev), rand_bank(H, I, dev)
    space = [dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_N=g, num_warps=w, num_stages=s)
             for bm, bn, bk, g, w, s in itertools.product((16, 32, 64, 128), (64, 128, 256), (64, 128, 256),
                                                          (1, 8), (4, 8), (3, 4, 5))
             if not (bm == 128 and bn == 256 and bk == 256)]
    table = {}
    for M in [int(v) for v in a.Ms.split(",")]:
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        tw = torch.rand(M, TOPK, device=dev)
        x8, xs = rowwise_quant(torch.randn(M, H, device=dev).bfloat16())
        h = torch.randn(M * TOPK, I, device=dev).bfloat16()
        h8, hs = rowwise_quant(h, relu2=True)
        y = torch.empty(M * TOPK, H, device=dev)
        best = {}
        for which in ("fc1", "fc2"):
            res = []
            for cfg in space:
                sid, eid, npp = moe_align(ids, cfg["BLOCK_M"], E)
                if which == "fc1":
                    fn = lambda: _gemm(x8, xs, *w1, h, tw.flatten(), sid, eid, npp, M * TOPK, TOPK, False, cfg)  # noqa
                else:
                    fn = lambda: _gemm(h8, hs, *w2, y, tw.flatten(), sid, eid, npp, M * TOPK, 1, True, cfg)  # noqa
                try:
                    res.append((timeit(fn), cfg))
                except Exception as e:  # out of resources etc.
                    continue
            res.sort(key=lambda t: t[0])
            best[which] = res[0][1]
            best[which + "_ms"] = res[0][0]
            print(M, which, res[0], flush=True)
        table[M] = best
        with open(a.out, "w") as f:
            json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
