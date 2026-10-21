#!/usr/bin/env bash
# RHEL 9 コンテナをビルドして、このリポジトリを /work にマウントして起動する。
#   ./rhel/run.sh                 対話シェル
#   ./rhel/run.sh python3 x.py    コマンドを実行
set -euo pipefail

IMAGE="${IMAGE:-python-rhel9}"
ENGINE="${ENGINE:-$(command -v podman || command -v docker)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

build_args=()
[ -n "${PACKAGES+x}" ] && build_args+=(--build-arg "PACKAGES=${PACKAGES}")
[ -n "${BASE_IMAGE:-}" ] && build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")

"$ENGINE" build "${build_args[@]}" -t "$IMAGE" "$REPO_ROOT/rhel"

tty_flags=()
[ -t 0 ] && tty_flags=(-it)

exec "$ENGINE" run --rm "${tty_flags[@]}" -v "$REPO_ROOT:/work:Z" "$IMAGE" "$@"
