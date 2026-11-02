# SGLang v0.5.22, linux/amd64; source baa9befa381511cd4d56f83980951e5a45fffc6d.
FROM docker.io/lmsysorg/sglang:v0.5.23-cu130@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

LABEL org.opencontainers.image.source="https://github.com/enthus-appdev/sglang-pcai" \
      org.opencontainers.image.title="sglang-pcai" \
      org.opencontainers.image.description="Pinned SGLang with an arguments-only PCAI serving entrypoint" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY checks/check_image.py /usr/local/lib/sglang-pcai/check_image.py
# CPU-only installed-version/source tripwires: no engine import, weights or GPU use.
RUN python3 /usr/local/lib/sglang-pcai/check_image.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "sglang.launch_server"]
CMD []
