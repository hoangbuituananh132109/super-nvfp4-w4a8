# Production image: stock vLLM + this plugin. No FlashInfer/CUTLASS rebuild.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.33.0
FROM ${VLLM_IMAGE}
COPY . /opt/super_nvfp4_w4a8
RUN python3 -m pip install --no-deps --no-build-isolation /opt/super_nvfp4_w4a8 \
 && python3 -c "import super_nvfp4_w4a8.kernels, super_nvfp4_w4a8.moe"
ENV VLLM_PLUGINS=super_nvfp4_w4a8
# Optional: tuned tile table produced on H200 by bench/tune.py
# COPY h200_super.json /opt/h200_super.json
# ENV SUPER_W4A8_CONFIG=/opt/h200_super.json
