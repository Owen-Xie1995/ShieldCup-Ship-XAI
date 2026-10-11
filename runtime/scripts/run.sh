#!/usr/bin/env bash
set -euo pipefail

profile=cpu
task=submission
image=""
data_dir=""
output_dir=""
print_command() {
  local argument
  for argument in "$@"; do printf "'%s' " "${argument//\'/\'\\\'\'}"; done
  printf '\n'
}
usage() {
  cat <<'USAGE'
Usage: scripts/run.sh --image NAME:TAG --output-dir DIR
       [--profile cpu|nvidia] [--task submission|smoke|probe] [--data-dir DIR]
       -- ARGUMENTS_FOR_RUNNER_OR_SMOKE_OR_PROBE
Data is mounted read-only at /data. Outputs are written to /output.
NVIDIA uses --gpus all and requires NVIDIA Container Toolkit on the host.
Use --device cuda in the Python command when validating an NVIDIA target.
USAGE
}
while (($#)); do
  case "$1" in
    --profile) profile="${2:?missing profile}"; shift 2 ;;
    --task) task="${2:?missing task}"; shift 2 ;;
    --image) image="${2:?missing image}"; shift 2 ;;
    --data-dir) data_dir="${2:?missing data directory}"; shift 2 ;;
    --output-dir) output_dir="${2:?missing output directory}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$image" && -n "$output_dir" ]] || { usage >&2; exit 2; }
case "$profile" in cpu|nvidia) ;; *) printf 'Unsupported profile: %s\n' "$profile" >&2; exit 2 ;; esac
case "$task" in
  submission) script=src/runner.py ;;
  smoke) script=src/smoke.py ;;
  probe) script=src/device_probe.py ;;
  *) printf 'Unsupported task: %s\n' "$task" >&2; exit 2 ;;
esac
command -v docker >/dev/null || { printf 'docker CLI is unavailable.\n' >&2; exit 127; }
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
run_command=(docker run --rm -e NVIDIA_TF32_OVERRIDE=0 --mount "type=bind,src=$output_dir,dst=/output")
if [[ -n "$data_dir" ]]; then
  data_dir="$(cd "$data_dir" && pwd)"
  run_command+=(--mount "type=bind,src=$data_dir,dst=/data,readonly")
fi
if [[ "$profile" == nvidia ]]; then run_command+=(--gpus all); fi
run_command+=(--entrypoint python "$image" "$script" "$@")
print_command "${run_command[@]}"
"${run_command[@]}"
