"""vLLM plugin: serve an *unchanged* ModelOpt NVFP4 checkpoint with W4A8 routed experts.

    vllm serve <nvfp4-checkpoint> --quantization super_nvfp4_w4a8 ...

Everything except the routed experts is delegated to vLLM's stock ModelOpt FP4 config
(attention, Mamba, shared expert, latent projections, FP8/NVFP4 linears, the
checkpoint's exclude list, MTP). Weight loading of the routed experts is the stock
ModelOpt NVFP4 MoE loader; only ``process_weights_after_loading`` and ``apply`` differ.

Env:
  SUPER_W4A8_MODE=hybrid|fused  hybrid (default): M < SUPER_W4A8_DENSE_MIN_TOKENS (2048) -> stock Marlin
                              W4A16, else NVFP4->FP8 decode (CUDA) + CUTLASS FP8 grouped GEMM.
                              fused: the original all-Triton W4A8 kernel (slow, reference only)
  SUPER_W4A8_CUDA_DECODE=1    JIT CUDA decode kernel (cuda_ops.py); 0 = Triton decode
  SUPER_W4A8_MTP=stock|w4a8   MTP routed experts (default stock = vLLM's own NVFP4 backend)
  SUPER_W4A8_CHUNK=8192       token chunk for the expert workspace
  SUPER_W4A8_CONFIG=path.json tuned tile table produced by bench/tune.py
  SUPER_W4A8_CHECK=1          log weight-conversion error of expert 0 of every layer
"""
from __future__ import annotations

import os

NAME = "super_nvfp4_w4a8"
_READY_LOGGED = [0]


def register():
    import torch
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    try:  # 0.33 name used by the earlier working MXFP4 adapter
        from vllm.model_executor.layers.fused_moe import RoutedExperts as _MoELayer
    except ImportError:  # older vLLM
        from vllm.model_executor.layers.fused_moe import FusedMoE as _MoELayer
    from vllm.model_executor.utils import replace_parameter

    from .fmt import prepare_w4a8, weight_conversion_error
    from .moe import nvfp4_w4a8_moe

    class SuperModelOptNvFp4W4A8MoEMethod(FusedMoEMethodBase):
        def __init__(self, inner, moe):
            super().__init__(moe)
            if getattr(moe, "is_act_and_mul", False) or getattr(moe, "has_bias", False):
                raise NotImplementedError("super_nvfp4_w4a8 supports bias-free non-gated experts only")
            act = str(getattr(moe, "activation", "")).lower()
            if "relu2" not in act:
                raise NotImplementedError(f"super_nvfp4_w4a8 implements ReLU2 only, got {act}")
            for attr in ("tp_size", "ep_size", "dp_size"):
                if getattr(moe, attr, 1) != 1:
                    raise NotImplementedError(f"super_nvfp4_w4a8 is TP1/EP1/DP1 only ({attr}={getattr(moe, attr)})")
            self.inner = inner
            self._w = None

        @property
        def topk_indices_dtype(self):
            return torch.int32

        # vLLM 0.33 RoutedExperts.weight_loader picks the ModelOpt scale-loading path by class
        # name ("ModelOpt" in name) and then queries these on the quant method itself.
        def uses_weight_scale_2_pattern(self):
            return self.inner.uses_weight_scale_2_pattern()

        @property
        def use_global_sf(self):
            return getattr(self.inner, "use_global_sf", False)

        def create_weights(self, layer, *args, **kwargs):
            # stock ModelOpt NVFP4 parameters + loaders (no conversion at load time)
            return self.inner.create_weights(layer, *args, **kwargs)

        def get_fused_moe_quant_config(self, layer):
            return None

        def process_weights_after_loading(self, layer):
            if self._w is not None:
                return
            cap = torch.cuda.get_device_capability()
            if cap < (8, 9):
                raise RuntimeError(f"FP8 tensor cores required (SM89+), got SM{cap[0]}{cap[1]}")
            if os.environ.get("SUPER_W4A8_MODE", "hybrid") == "hybrid":
                # small M: stock Marlin W4A16; large M: NVFP4->FP8 decode + CUTLASS (dense_fp8.py)
                from .dense_fp8 import HybridNvfp4MoE
                raw = []
                for pfx in ("w13", "w2"):
                    packed = getattr(layer, f"{pfx}_weight").data
                    scale = getattr(layer, f"{pfx}_weight_scale").data
                    g = getattr(layer, f"{pfx}_weight_scale_2").data
                    if scale.dtype != torch.float8_e4m3fn:
                        scale = scale.view(torch.float8_e4m3fn)
                    raw.append((packed, scale, g.reshape(packed.shape[0], -1)[:, 0].contiguous()))
                self._hybrid = HybridNvfp4MoE(raw[0], raw[1])
                self._w = ("hybrid",)
                if self._hybrid.share_codes:
                    # Marlin owns repacked codes + processed scales and the decode reads those, so
                    # every checkpoint-layout expert tensor can go (codes, block scales, globals,
                    # unused W4A4 input scales).
                    dev = raw[0][0].device
                    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                                 "w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale",
                                 "w2_input_scale"):
                        p = getattr(layer, name, None)
                        if p is not None:
                            p.data = torch.empty(0, dtype=p.dtype, device=dev)
                    del raw
                    torch.cuda.empty_cache()
                if self._hybrid.flushed_scales:
                    print(f"SUPER_W4A8_WARN {getattr(layer, 'layer_name', '?')} "
                          f"{self._hybrid.flushed_scales} block scales flushed to 0 by Marlin", flush=True)
                if _READY_LOGGED[0] < 2:
                    from .dense_fp8 import _use_cuda_decode  # builds the CUDA ext now, not in the first forward
                    print(f"SUPER_NVFP4_W4A8_READY {getattr(layer, 'layer_name', 'routed_experts')} "
                          f"hybrid dense_min_tokens={self._hybrid.thr} share_codes={self._hybrid.share_codes} "
                          f"cuda_decode={_use_cuda_decode()} gemm={self._hybrid.gemm} "
                          f"fused_act={os.environ.get('SUPER_W4A8_FUSED_ACT', '1')}", flush=True)
                    _READY_LOGGED[0] += 1
                return
            mats = []
            for pfx in ("w13", "w2"):
                packed = getattr(layer, f"{pfx}_weight").data
                scale = getattr(layer, f"{pfx}_weight_scale").data
                g = getattr(layer, f"{pfx}_weight_scale_2").data
                if scale.dtype != torch.float8_e4m3fn:
                    scale = scale.view(torch.float8_e4m3fn)
                p, s_rel, bsc = prepare_w4a8(packed, scale, g)
                if os.environ.get("SUPER_W4A8_CHECK") == "1":
                    g1 = g.reshape(packed.shape[0], -1)[:1, 0] if g.numel() > 1 else g.reshape(1)
                    err = weight_conversion_error(packed[:1], scale[:1], g1, s_rel[:1], bsc[:1])
                    print(f"SUPER_W4A8_CHECK {getattr(layer, 'layer_name', '?')} {pfx} {err}", flush=True)
                replace_parameter(layer, f"{pfx}_weight_scale", s_rel)
                mats.append((p, s_rel, bsc))
            E, I, Hh = mats[0][0].shape
            if mats[1][0].shape != (E, 2 * Hh, I // 2):
                raise ValueError(f"unexpected expert shapes w13={tuple(mats[0][0].shape)} w2={tuple(mats[1][0].shape)}")
            for name in ("w13_input_scale", "w2_input_scale"):  # W4A4 calibration, unused by A8
                if hasattr(layer, name):
                    delattr(layer, name)
            self._w = tuple(mats)
            if _READY_LOGGED[0] < 2:
                print(f"SUPER_NVFP4_W4A8_READY {getattr(layer, 'layer_name', 'routed_experts')} "
                      f"E={E} I={I} H={2 * Hh} SM{cap[0]}{cap[1]} NVFP4-g16xFP8 triton", flush=True)
                _READY_LOGGED[0] += 1

        def apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs):
            if self._w is None:
                raise RuntimeError("expert weights were not prepared")
            if self._w[0] == "hybrid":
                return self._hybrid(x, topk_weights, topk_ids)
            return nvfp4_w4a8_moe(x, topk_weights, topk_ids, self._w[0], self._w[1])

    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    def _is_nvfp4_moe(method) -> bool:
        if isinstance(method, ModelOptNvFp4FusedMoE):
            return True
        n = type(method).__name__
        return "NvFp4" in n and "MoE" in n

    def _wrap_get_quant_method(stock_get):
        def get_quant_method(self, layer, prefix):
            base = stock_get(self, layer, prefix)
            if not isinstance(layer, _MoELayer):
                return base
            if base is None or not _is_nvfp4_moe(base):
                print(f"SUPER_W4A8_SKIP {prefix}: stock MoE method {type(base).__name__}", flush=True)
                return base
            if "mtp" in prefix.split(".") and os.environ.get("SUPER_W4A8_MTP", "stock") != "w4a8":
                return base
            return SuperModelOptNvFp4W4A8MoEMethod(base, layer.moe_config)
        return get_quant_method

    _subclass_cache = {}

    @register_quantization_config(NAME)
    class SuperNvfp4W4A8Config(QuantizationConfig):
        """Registered entry point. ``from_config`` builds the *stock* ModelOpt config
        that vLLM would have chosen (``modelopt_fp4`` or ``modelopt_mixed``) and retypes
        it to a subclass that only overrides ``get_quant_method`` for routed experts,
        so every other code path (isinstance checks, names, exclude lists, KV scales)
        is the stock one."""

        @classmethod
        def get_name(cls):
            return NAME

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.bfloat16, torch.half]

        @classmethod
        def get_min_capability(cls):
            return 89

        @classmethod
        def get_config_filenames(cls):
            return ["hf_quant_config.json"]

        @staticmethod
        def _stock_name(config) -> str:
            q = config.get("quantization", config) if isinstance(config, dict) else {}
            algo = str(q.get("quant_algo", "") or config.get("quant_algo", "")).upper()
            if "MIXED" in algo:
                return "modelopt_mixed"
            if "NVFP4" in algo or "FP4" in algo:
                return "modelopt_fp4"
            raise ValueError(f"{NAME} needs a ModelOpt NVFP4 / mixed-precision checkpoint, got quant_algo={algo!r}")

        @classmethod
        def from_config(cls, config):
            stock_cls = get_quantization_config(cls._stock_name(config))
            inner = stock_cls.from_config(config)
            sub = _subclass_cache.get(stock_cls)
            if sub is None:
                sub = type(f"SuperW4A8_{stock_cls.__name__}", (stock_cls,),
                           {"get_quant_method": _wrap_get_quant_method(stock_cls.get_quant_method)})
                _subclass_cache[stock_cls] = sub
            inner.__class__ = sub
            print(f"SUPER_W4A8_CONFIG stock={stock_cls.__name__}", flush=True)
            return inner

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, *args, **kwargs):
            if user_quant == NAME and hf_quant_cfg:
                return NAME
            return None

        def get_quant_method(self, layer, prefix):  # never used: from_config returns the stock subclass
            raise RuntimeError("unreachable")
