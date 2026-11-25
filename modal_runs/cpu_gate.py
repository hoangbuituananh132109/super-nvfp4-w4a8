"""CPU-only gate inside the exact vLLM image (no GPU allocated).

1. pytest tests/test_cpu_reference.py
2. plugin registers; --quantization super_nvfp4_w4a8 resolves for the checkpoint's
   hf_quant_config.json and yields a stock ModelOpt config subclass
3. checkpoint header audit for one MoE layer (names, dtypes, shapes; no weight read)
4. stock ModelOptNvFp4FusedMoE API surface that the plugin relies on
5. Triton AOT compile of both kernels for sm_90a; PTX must contain wgmma + e4m3 cvt
Writes JSON to argv[1]; status CPU_PASS only if every check passed.
"""
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NVFP4 = os.environ.get("NVFP4_CKPT", "/cache/models/NVFP4/ff433f5493e25d631c9f12b5d55c674229923d02")


def binding():
    files = sorted(p for p in REPO.rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_pytest(res):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests/test_cpu_reference.py")],
                       capture_output=True, text=True, cwd=REPO)
    res["pytest"] = r.stdout[-4000:] + r.stderr[-2000:]
    assert r.returncode == 0, "pytest failed"


def check_plugin(res):
    import importlib.metadata as md
    res["packages"] = {p: md.version(p) for p in ("vllm", "torch", "triton")}
    from super_nvfp4_w4a8.vllm_plugin import register, NAME
    register()
    from vllm.model_executor.layers.quantization import get_quantization_config, QUANTIZATION_METHODS
    assert NAME in QUANTIZATION_METHODS, "not registered"
    cls = get_quantization_config(NAME)
    qcfg_path = Path(NVFP4, "hf_quant_config.json")
    if qcfg_path.exists():
        hfq = json.loads(qcfg_path.read_text())
    else:
        hfq = json.loads(Path(NVFP4, "config.json").read_text())["quantization_config"]
    res["hf_quant_algo"] = (hfq.get("quantization") or hfq).get("quant_algo")
    assert cls.override_quantization_method(hfq, NAME) == NAME
    inst = cls.from_config(hfq)
    res["config_class"] = type(inst).__name__
    res["config_mro"] = [c.__name__ for c in type(inst).__mro__[:4]]
    res["stock_get_name"] = inst.get_name()
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    sig = inspect.signature(ModelOptNvFp4FusedMoE.create_weights)
    res["create_weights_sig"] = str(sig)
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    res["apply_sig_base"] = str(inspect.signature(FusedMoEMethodBase.apply))
    # baseline APIs used by tests/bench_baselines.py
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new  # noqa: F401
    from vllm.scalar_type import scalar_types
    from flashinfer.fused_moe import ReLU2, cutlass_fused_moe  # noqa: F401
    from flashinfer.fused_moe.prepare import (_quantize_mxfp4_linear,  # noqa: F401
                                              preprocess_moe_weights_for_sm90_mixed_gemm_humming)
    _ = MoEActivation.RELU2_NO_MUL, scalar_types.float4_e2m1f.id
    for fn, names in ((prepare_nvfp4_moe_layer_for_marlin, ("layer", "w13", "w13_scale", "w13_scale_2", "is_act_and_mul")),
                      (fused_marlin_moe, ("global_scale1", "global_scale2", "workspace", "activation"))):
        params = inspect.signature(fn).parameters
        missing = [n for n in names if n not in params]
        assert not missing, f"{fn.__name__} lacks {missing}"
    sys.path.insert(0, str(REPO / "tests"))
    import bench_baselines  # noqa: F401  (import-level check of the GPU bench script)
    # large-M dense FP8 path APIs (super_nvfp4_w4a8/dense_fp8.py)
    from vllm import _custom_ops as _ops
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute, moe_unpermute  # noqa: F401
    from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation  # noqa: F401
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts  # noqa: F401
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    assert hasattr(_ops, "cutlass_moe_mm") and hasattr(_ops, "scaled_fp8_quant")
    assert "per_out_ch_quant" in inspect.signature(fp8_w8a8_moe_quant_config).parameters
    import super_nvfp4_w4a8.dense_fp8  # noqa: F401
    res["baseline_apis"] = "PASS"
    src = inspect.getsource(ModelOptNvFp4FusedMoE.create_weights)
    for name in ("w13_weight_scale_2", "w2_weight_scale_2", "w13_weight_scale", "w2_weight"):
        assert name in src, f"stock create_weights no longer defines {name}"


def check_ckpt(res):
    idx = json.loads(Path(NVFP4, "model.safetensors.index.json").read_text())["weight_map"]
    layer = None
    for k in sorted(idx):
        m = re.search(r"^backbone\.layers\.(\d+)\.mixer\.experts\.0\.up_proj\.weight$", k)
        if m:
            layer = int(m.group(1))
            break
    assert layer is not None, "no routed expert keys matched"
    from safetensors import safe_open
    want = {}
    for k, f in idx.items():
        if re.search(rf"^backbone\.layers\.{layer}\.mixer\.experts\.(0|511)\.(up|down)_proj\.", k):
            with safe_open(str(Path(NVFP4, f)), "pt") as h:
                sl = h.get_slice(k)
                want[k] = {"dtype": str(sl.get_dtype()), "shape": sl.get_shape()}
    res["ckpt_layer"] = layer
    res["ckpt_tensors"] = want
    up = next(v for k, v in want.items() if k.endswith("experts.0.up_proj.weight"))
    upsc = next(v for k, v in want.items() if k.endswith("experts.0.up_proj.weight_scale"))
    assert up["dtype"] in ("U8", "uint8") and upsc["dtype"] in ("F8_E4M3", "float8_e4m3fn"), (up, upsc)
    assert up["shape"][1] * 2 == upsc["shape"][1] * 16, (up, upsc)


def _compile(fn, sig, const, target, opts, div16=()):
    """AOT compile across Triton versions (constexpr keys: names, (idx,) tuples or idx).
    div16: argument names that the JIT would specialise as 16-divisible (aligned pointers, sizes)."""
    import triton
    from triton.compiler import ASTSource
    names = list(sig)
    errs = []
    for keyed in (lambda: const, lambda: {(names.index(k),): v for k, v in const.items()},
                  lambda: {names.index(k): v for k, v in const.items()}):
        try:
            attrs = {(names.index(k),): [["tt.divisibility", 16]] for k in div16} or None
            return triton.compile(ASTSource(fn, sig, constexprs=keyed(), attrs=attrs), target=target, options=opts)
        except Exception as e:  # noqa
            errs.append(repr(e)[:800])
    raise RuntimeError("Triton AOT compile failed: " + " | ".join(errs))


def _find_ptxas():
    import shutil
    try:
        import triton
        cand = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
        if cand.exists():
            return str(cand)
    except Exception:  # noqa: BLE001
        pass
    return shutil.which("ptxas")


def _ptxas_stats(ptxas, ptx, name):
    """Registers / spill bytes / stack from `ptxas -v` on the Triton PTX (CPU only)."""
    if not ptxas:
        return {"ptxas": "not found"}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td, f"{name}.ptx")
        src.write_text(ptx)
        r = subprocess.run([ptxas, "-arch=sm_90a", "-v", str(src), "-o", str(Path(td, "o.cubin"))],
                           capture_output=True, text=True, timeout=300)
    txt = r.stdout + r.stderr
    st = {"ptxas_rc": r.returncode}
    m = re.search(r"Used (\d+) registers", txt)
    st["regs"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    st["spill_store_B"], st["spill_load_B"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"(\d+) bytes stack frame", txt)
    st["stack_B"] = int(m.group(1)) if m else None
    if r.returncode:
        st["ptxas_err"] = txt[-800:]
    return st


def check_compile(res):
    import triton
    from triton.backends.compiler import GPUTarget
    from super_nvfp4_w4a8.kernels import nvfp4_w4a8_moe_gemm, rowwise_fp8_quant
    tgt = GPUTarget("cuda", 90, 32)
    sig = {"a_ptr": "*fp8e4nv", "a_scale_ptr": "*fp32", "b_ptr": "*u8", "s_ptr": "*fp8e4nv",
           "bscale_ptr": "*fp32", "c_ptr": "*bf16", "topk_w_ptr": "*fp32", "sorted_ids_ptr": "*i64",
           "expert_ids_ptr": "*i32", "num_post_pad_ptr": "*i32", "N": "i32", "K": "i32",
           "num_valid_slots": "i32", "stride_am": "i64", "stride_ak": "i64", "stride_be": "i64",
           "stride_bn": "i64", "stride_bk": "i64", "stride_se": "i64", "stride_sn": "i64",
           "stride_sk": "i64", "stride_bse": "i64", "stride_bsn": "i64", "stride_cm": "i64",
           "stride_cn": "i64"}
    base_const = dict(A_ROW_DIV=22, MUL_ROUTED_WEIGHT=False, BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
                      GROUP_N=8, EVEN_K=True, PROMOTE_ACC=True)
    for k in base_const:
        sig[k] = "constexpr"
    out = {"variants": {}}
    ptxas = _find_ptxas()
    out["ptxas"] = ptxas
    # every default tile config of moe.py (+ promote off, + odd-K), all must compile for sm_90a
    variants = [(bm, bn, bk, w, st, pr, ek)
                for (bm, bn, bk, w, st) in ((16, 128, 128, 4, 4), (32, 128, 128, 4, 4), (64, 128, 128, 4, 4),
                                            (128, 128, 128, 8, 3), (16, 64, 128, 4, 3), (16, 128, 64, 8, 3),
                                            (64, 128, 64, 8, 3))
                for pr in (True,) for ek in (True,)]
    variants += [(16, 128, 128, 4, 4, False, True), (64, 128, 128, 4, 4, True, False)]
    # informational spill map over the GPU tile-sweep grid (not gating)
    grid = [(bm, bn, bk, w, st, True, True) for bm in (16, 32, 64, 128) for bn in (64, 128)
            for bk in (64, 128) for w in (4, 8) for st in (2, 3)]
    variants += [v for v in grid if v not in variants]
    first_ptx = None
    for bm, bn, bk, w, st, pr, ek in variants:
        const = dict(base_const, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PROMOTE_ACC=pr, EVEN_K=ek)
        name = f"bm{bm}_bn{bn}_bk{bk}_w{w}_s{st}_promote{int(pr)}_evenk{int(ek)}"
        v = {}
        try:
            ck = _compile(nvfp4_w4a8_moe_gemm, dict(sig), const, tgt, {"num_warps": w, "num_stages": st})
            ptx = ck.asm["ptx"]
            v["wgmma"] = "wgmma" in ptx
            v["cvt_e4m3"] = bool(re.search(r"cvt\.rn\.satfinite\.e4m3x2", ptx))
            v["shared"] = ck.metadata.shared if hasattr(ck.metadata, "shared") else None
            v.update(_ptxas_stats(ptxas, ptx, name))
            if first_ptx is None:
                first_ptx = ptx
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
        out["variants"][name] = v
    if first_ptx:
        Path("gemm_sm90.ptx").write_text(first_ptx)
    out["gemm_wgmma"] = all(v.get("wgmma") for v in out["variants"].values())
    out["gemm_cvt_e4m3"] = all(v.get("cvt_e4m3") for v in out["variants"].values())
    out["compile_errors"] = [n for n, v in out["variants"].items() if "compile_error" in v]
    from super_nvfp4_w4a8.moe import _DEFAULT
    out["default_cfg_spills"] = {}
    for ub, c in _DEFAULT.items():
        name = (f"bm{c['BLOCK_M']}_bn{c['BLOCK_N']}_bk{c['BLOCK_K']}_w{c['num_warps']}"
                f"_s{c['num_stages']}_promote1_evenk1")
        out["default_cfg_spills"][ub] = out["variants"].get(name, {}).get("spill_store_B", "not compiled")
    sig2 = {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "K": "i32", "stride_xm": "i64",
            "stride_qm": "i64", "APPLY_RELU2": "constexpr", "BLOCK": "constexpr"}
    c2 = dict(APPLY_RELU2=True, BLOCK=4096)
    _compile(rowwise_fp8_quant, sig2, c2, tgt, {"num_warps": 8})
    # large-M decode kernels (flat uint16-store kernel: every GPU sweep config must compile)
    from super_nvfp4_w4a8.dense_fp8 import _nvfp4_to_fp8_flat_kernel, DECODE_SWEEP
    out["decode"] = {}
    sig3 = {"packed_ptr": "*u8", "srel_ptr": "*fp8e4nv", "out16_ptr": "*u16", "n_bytes": "i64",
            "BLOCK": "constexpr"}
    for blk, w in DECODE_SWEEP:
        name = f"flat_b{blk}_w{w}"
        v = {}
        try:
            ck = _compile(_nvfp4_to_fp8_flat_kernel, dict(sig3), {"BLOCK": blk}, tgt, {"num_warps": w},
                          div16=("packed_ptr", "srel_ptr", "out16_ptr", "n_bytes"))
            ptx = ck.asm["ptx"]
            v["cvt_e4m3"] = bool(re.search(r"cvt\.rn\.satfinite\.e4m3x2", ptx))
            v["vec_st"] = sorted(set(re.findall(r"st\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v["vec_ld"] = sorted(set(re.findall(r"ld\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v.update(_ptxas_stats(ptxas, ptx, name))
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
            out["compile_errors"].append(name)
        out["decode"][name] = v
    from super_nvfp4_w4a8.dense_fp8 import _marlin_nvfp4_to_fp8_kernel, MARLIN_DECODE_SWEEP
    for bn, bc, w, cb in MARLIN_DECODE_SWEEP:
        name = f"marlin_bn{bn}_bc{bc}_w{w}_c{int(cb)}"
        sig4 = {"q_ptr": "*i32", "srel_ptr": "*fp8e4nv", "out_ptr": "*u32" if cb else "*fp8e4nv",
                "N": "i32", "K": "i32", "BLOCK_N": "constexpr", "BLOCK_C": "constexpr", "COMBINE": "constexpr"}
        v = {}
        try:
            ck = _compile(_marlin_nvfp4_to_fp8_kernel, sig4, {"BLOCK_N": bn, "BLOCK_C": bc, "COMBINE": cb}, tgt,
                          {"num_warps": w}, div16=("q_ptr", "srel_ptr", "out_ptr", "N", "K"))
            ptx = ck.asm["ptx"]
            v["cvt_e4m3"] = bool(re.search(r"cvt\.rn\.satfinite\.e4m3x2", ptx))
            v["vec_st"] = sorted(set(re.findall(r"st\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v.update(_ptxas_stats(ptxas, ptx, name))
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
            out["compile_errors"].append(name)
        out["decode"][name] = v
    from super_nvfp4_w4a8.dense_fp8 import _marlin_tile_decode_ms_kernel
    for T, w in ((8, 2), (4, 2), (8, 4)):
        name = f"mtile_ms_T{T}_w{w}"
        sig6 = {"q_ptr": "*i32", "ms_ptr": "*u8", "shift_ptr": "*i8", "out_ptr": "*fp8e4nv", "N": "i32",
                "K": "i32", "T": "constexpr"}
        v = {}
        try:
            ck = _compile(_marlin_tile_decode_ms_kernel, sig6, {"T": T}, tgt, {"num_warps": w},
                          div16=("q_ptr", "ms_ptr", "shift_ptr", "out_ptr", "N", "K"))
            ptx = ck.asm["ptx"]
            v["vec_st"] = sorted(set(re.findall(r"st\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v.update(_ptxas_stats(ptxas, ptx, name))
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
            out["compile_errors"].append(name)
        out["decode"][name] = v
    from super_nvfp4_w4a8.dense_fp8 import _marlin_tile_decode_kernel, MARLIN_TILE_SWEEP
    for T, w in MARLIN_TILE_SWEEP:
        name = f"mtile_T{T}_w{w}"
        sig5 = {"q_ptr": "*i32", "srel_ptr": "*fp8e4nv", "out_ptr": "*fp8e4nv", "N": "i32", "K": "i32",
                "T": "constexpr"}
        v = {}
        try:
            ck = _compile(_marlin_tile_decode_kernel, sig5, {"T": T}, tgt, {"num_warps": w},
                          div16=("q_ptr", "srel_ptr", "out_ptr", "N", "K"))
            ptx = ck.asm["ptx"]
            v["vec_ld"] = sorted(set(re.findall(r"ld\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v["vec_st"] = sorted(set(re.findall(r"st\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v["shared"] = getattr(ck.metadata, "shared", None)
            v.update(_ptxas_stats(ptxas, ptx, name))
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
            out["compile_errors"].append(name)
        out["decode"][name] = v
    # fused W4A8 GEMM reading Marlin codes/scales (marlin_w4a8.py): compile + IR shape, informational
    from super_nvfp4_w4a8.marlin_w4a8 import marlin_w4a8_moe_gemm
    out["fused_marlin"] = {}
    sig7 = {"a_ptr": "*fp8e4nv", "a_scale_ptr": "*fp32", "q_ptr": "*i32", "ms_ptr": "*u8", "shift_ptr": "*i8",
            "bscale_ptr": "*fp32", "c_ptr": "*bf16", "topk_w_ptr": "*fp32", "sorted_ids_ptr": "*i32",
            "expert_ids_ptr": "*i32", "num_post_pad_ptr": "*i32", "N": "i32", "K": "i32",
            "num_valid_slots": "i32", "stride_am": "i32", "stride_cm": "i32"}
    for k in ("A_ROW_DIV", "MUL_ROUTED_WEIGHT", "BLOCK_M", "NT", "T", "GROUP_N"):
        sig7[k] = "constexpr"
    for bm, nt, T, w, st in ((16, 2, 8, 8, 3), (16, 1, 8, 4, 3), (32, 2, 8, 8, 3), (64, 2, 8, 8, 3),
                             (16, 2, 4, 8, 4), (64, 1, 8, 4, 3)):
        name = f"fm_bm{bm}_nt{nt}_T{T}_w{w}_s{st}"
        v = {}
        try:
            ck = _compile(marlin_w4a8_moe_gemm, dict(sig7),
                          dict(A_ROW_DIV=22, MUL_ROUTED_WEIGHT=False, BLOCK_M=bm, NT=nt, T=T, GROUP_N=8),
                          tgt, {"num_warps": w, "num_stages": st},
                          div16=("a_ptr", "q_ptr", "ms_ptr", "shift_ptr", "bscale_ptr", "c_ptr", "N", "K",
                                 "stride_am", "stride_cm"))
            ptx = ck.asm["ptx"]
            ttgir = ck.asm.get("ttgir", "")
            v["wgmma"] = "wgmma" in ptx
            v["mma_sync"] = "mma.sync" in ptx
            v["shared"] = getattr(ck.metadata, "shared", None)
            v["ir"] = {op: ttgir.count(op) for op in ("ttng.warp_group_dot ", "ttg.convert_layout",
                                                       "ttg.local_alloc", "ttg.local_load", "ttg.async_copy",
                                                       "tt.load", "tt.dot ")}
            v["vec_ld"] = sorted(set(re.findall(r"ld\.global(?:\.\w+)*\.v\d\.b\d+", ptx)))
            v.update(_ptxas_stats(ptxas, ptx, name))
            if name == "fm_bm16_nt2_T8_w8_s3":
                Path("fused_marlin.ttgir").write_text(ttgir)
        except Exception as e:  # noqa: BLE001
            v["compile_error"] = repr(e)[:1500]
            out["compile_errors"].append(name)
        out["fused_marlin"][name] = v
    res["compile"] = out
    assert not out["compile_errors"], f"sm_90a compile failed for {out['compile_errors']}"
    bad = {ub: s for ub, s in out["default_cfg_spills"].items() if s != 0}
    assert not bad, f"default tile configs spill registers (bytes per thread): {bad}"
    assert out["gemm_wgmma"], "no wgmma in PTX (FP8 tensor-core path not taken)"
    assert out["gemm_cvt_e4m3"], "no e4m3 cvt in PTX"


def check_cuda_ext(res):
    """nvcc build of cuda_ops (sm_90a) without a GPU; records the toolchain."""
    import shutil
    nv = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
    res["cuda_ext"] = {"nvcc": nv}
    if nv:
        res["cuda_ext"]["nvcc_version"] = subprocess.run([nv, "--version"], capture_output=True, text=True).stdout[-200:]
        os.environ["PATH"] = os.path.dirname(nv) + ":" + os.environ.get("PATH", "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from super_nvfp4_w4a8.cuda_ops import module
    module()
    res["cuda_ext"]["built"] = True


def check_lmeval(res):
    """lm-eval venv must import (it runs against the server in serve2 --evals)."""
    if not Path("/opt/lmeval/bin/lm_eval").exists():
        res["lmeval"] = "not installed"
        return
    r = subprocess.run(["/opt/lmeval/bin/python", "-c", "import lm_eval.evaluator, lm_eval.models; print('ok')"],
                       capture_output=True, text=True, timeout=600)
    res["lmeval"] = (r.stdout + r.stderr)[-1500:]
    assert r.returncode == 0, res["lmeval"]


def main():
    res = {"status": "RUNNING", "binding": binding(), "checks": {}}
    ok = True
    for name, fn in (("pytest", check_pytest), ("plugin", check_plugin), ("ckpt", check_ckpt),
                     ("compile", check_compile), ("cuda_ext", check_cuda_ext), ("lmeval", check_lmeval)):
        try:
            fn(res)
            res["checks"][name] = "PASS"
        except Exception:
            ok = False
            res["checks"][name] = traceback.format_exc()[-3000:]
    res["status"] = "CPU_PASS" if ok else "CPU_FAIL"
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
