# RHEL 9 (Red Hat Universal Base Image) 開発用コンテナ
# BASE_IMAGE は registry.access.redhat.com/ubi9/ubi に差し替えても同じものです。
ARG BASE_IMAGE=docker.io/redhat/ubi9:latest
FROM ${BASE_IMAGE}

# 追加でインストールするパッケージ（空にすると dnf をスキップ）
ARG PACKAGES="python3 python3-pip git"
RUN if [ -n "${PACKAGES}" ]; then \
        dnf -y install ${PACKAGES} && dnf clean all; \
    fi

WORKDIR /work
CMD ["/bin/bash"]
