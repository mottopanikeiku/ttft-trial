"""Opt-in Ada conversion of vLLM block-FP8 linear weights to channel FP8.

Set ``VLLM_ADA_CHANNEL_FP8=1`` and add this directory to ``PYTHONPATH`` before
starting vLLM. Python imports ``sitecustomize`` in the API process and every
spawned engine process, so no installed vLLM files are modified.

RTX 6000 Ada (SM 8.9) has hardware FP8 tensor cores, but vLLM 0.19.1 does not
support its CUTLASS block-FP8 kernel there. Qwen3.6-27B-FP8 therefore uses a
Triton block-scaled GEMM. This shim dequantizes each loaded 128x128-scaled
weight once, requantizes each output channel to FP8, and selects vLLM's CUTLASS
FP8 GEMM. It changes weight quantization and can change model quality; it is an
explicit performance experiment, not a lossless kernel substitution.
"""

import os

if os.environ.get("VLLM_ADA_CHANNEL_FP8") == "1":
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization import fp8 as fp8_module
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms import current_platform

    _original_process = fp8_module.Fp8LinearMethod.process_weights_after_loading
    _converted_layers = 0

    def _process_weights_after_loading(self, layer):
        global _converted_layers

        _original_process(self, layer)
        if not self.block_quant:
            return

        capability = current_platform.get_device_capability()
        if capability is None or capability.to_int() != 89:
            raise RuntimeError(
                "VLLM_ADA_CHANNEL_FP8 requires compute capability 8.9"
            )
        if self.weight_block_size != [128, 128]:
            raise RuntimeError(
                "VLLM_ADA_CHANNEL_FP8 requires 128x128 block-FP8 weights"
            )

        weight = layer.weight
        rows, columns = weight.shape
        block_rows, block_columns = self.weight_block_size
        scale = layer.weight_scale_inv.to(torch.bfloat16)
        expanded_scale = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(
            block_columns, dim=1
        )[:rows, :columns]
        dequantized = weight.to(torch.bfloat16)
        dequantized.mul_(expanded_scale)
        del expanded_scale

        # Dynamic per-token quantization over [N, K] is per-output-channel when
        # applied to a weight matrix. CUTLASS consumes the transposed [K, N]
        # view and broadcasts the resulting N scales over output columns.
        channel_weight, channel_scale = ops.scaled_fp8_quant(
            dequantized, use_per_token_if_dynamic=True
        )
        del dequantized

        replace_parameter(layer, "weight", channel_weight.t())
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(channel_scale.flatten(), requires_grad=False),
        )
        del layer._parameters["weight_scale_inv"]
        layer.weight_block_size = None

        self.block_quant = False
        self.weight_block_size = None
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            out_dtype=self.out_dtype,
            module_name="AdaChannelFp8LinearMethod",
        )
        self.use_marlin = False

        _converted_layers += 1
        if _converted_layers == 1 or _converted_layers % 100 == 0:
            fp8_module.logger.warning(
                "Ada channel-FP8 experiment converted %d linear layer(s)",
                _converted_layers,
            )

    fp8_module.Fp8LinearMethod.process_weights_after_loading = (
        _process_weights_after_loading
    )
