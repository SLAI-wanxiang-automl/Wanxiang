#!/usr/bin/env bash
#
# 同步 upstream 子项目与 packages/ 目录。
#
# 本伞仓用 git subtree 导入了两个子项目，代码保持独立、提交历史完整保留。
# 如果在子项目的原仓库里改了代码，用 pull 拉回伞仓；
# 如果直接在 packages/ 里改了代码，用 push 回推到原仓库。
#
# 用法：
#   ./scripts/sync-upstream.sh pull energy-tsf
#   ./scripts/sync-upstream.sh pull energy-expert
#   ./scripts/sync-upstream.sh push energy-tsf
#   ./scripts/sync-upstream.sh push energy-expert
#
set -euo pipefail

ORG="${ORG:-SLAI-wanxiang-automl}"

# 子项目名 -> 原仓库名
declare -A UPSTREAM=(
  [energy-tsf]="EnergyTSF"
  [energy-expert]="EnergyExpert"
)

usage() {
  echo "用法: $0 {pull|push} {energy-tsf|energy-expert}" >&2
  exit 1
}

[[ $# -eq 2 ]] || usage
action="$1"
component="$2"

repo="${UPSTREAM[$component]:-}"
[[ -n "$repo" ]] || { echo "未知组件: $component" >&2; usage; }

url="https://github.com/${ORG}/${repo}.git"
prefix="packages/${component}"

case "$action" in
  pull)
    echo "==> subtree pull ${prefix} from ${url} main"
    git subtree pull --prefix="$prefix" "$url" main \
      -m "chore(${component}): merge upstream changes"
    ;;
  push)
    echo "==> subtree push ${prefix} to ${url} main"
    git subtree push --prefix="$prefix" "$url" main
    ;;
  *)
    usage
    ;;
esac
