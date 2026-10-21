# Python
## RHEL 9 環境（コンテナ）

Red Hat 公式の UBI（Universal Base Image、無償で再配布可能な RHEL 9）を使います。
Docker または Podman が入っていれば、どの PC でも動きます。

```bash
./rhel/run.sh                    # RHEL 9 のシェルに入る（リポジトリは /work）
./rhel/run.sh python3 script.py  # RHEL 9 上でコマンドを実行
```

- 追加パッケージ: `PACKAGES="python3 python3-pip git gcc" ./rhel/run.sh`
- `dnf` を使わずベースイメージだけで起動: `PACKAGES="" ./rhel/run.sh`
- UBI のリポジトリは RHEL の一部パッケージのみです。RHEL 全体が必要な場合は、
  無償の Red Hat Developer Subscription を取得して VM（Hyper-V / VirtualBox / WSL など）にインストールしてください。
