# SGLang v0.5.22, linux/amd64; source baa9befa381511cd4d56f83980951e5a45fffc6d.
FROM docker.io/lmsysorg/sglang:v0.5.22-cu130@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae

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
