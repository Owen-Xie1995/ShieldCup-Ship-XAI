#!/usr/bin/env bash
set -euo pipefail

model_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile=cpu
platform=""
torch_index=""
image=""
base_image=python:3.12.9-slim-bookworm
evidence_dir=""
dry_run=0

print_command() {
  local argument
  for argument in "$@"; do printf "'%s' " "${argument//\'/\'\\\'\'}"; done
  printf '\n'
}

usage() {
  cat <<'USAGE'
Usage: scripts/build.sh [--profile cpu] [--platform linux/arm64|linux/amd64]
       [--torch-index cpu] [--image NAME:TAG]
       [--base-image TAG_OR_TAG@sha256:DIGEST] [--evidence-dir DIR] [--dry-run]
CPU defaults to the host architecture. The packaged constraints use official CPU wheels.
This recipe does not define a Jetson/JetPack GPU environment.
USAGE
}

while (($#)); do
  case "$1" in
    --profile) profile="${2:?missing profile}"; shift 2 ;;
    --platform) platform="${2:?missing platform}"; shift 2 ;;
    --torch-index) torch_index="${2:?missing torch index}"; shift 2 ;;
    --image) image="${2:?missing image name}"; shift 2 ;;
    --base-image) base_image="${2:?missing base image}"; shift 2 ;;
    --evidence-dir) evidence_dir="${2:?missing evidence directory}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$profile" in
  cpu)
    if [[ -z "$platform" ]]; then
      case "$(uname -m)" in
        arm64|aarch64) platform=linux/arm64 ;;
        x86_64|amd64) platform=linux/amd64 ;;
        *) printf 'Specify a supported --platform explicitly.\n' >&2; exit 2 ;;
      esac
    fi
    torch_index="${torch_index:-cpu}"
    [[ "$torch_index" == cpu ]] || { printf 'CPU profile requires --torch-index cpu.\n' >&2; exit 2; }
    ;;
  nvidia)
    printf "This packaged recipe uses CPU wheels. A GPU deployment needs a separate platform-specific recipe.\n" >&2; exit 2 ;;
  *) printf 'Unsupported profile: %s\n' "$profile" >&2; exit 2 ;;
esac
case "$platform" in linux/arm64|linux/amd64) ;; *) printf 'Unsupported platform: %s\n' "$platform" >&2; exit 2 ;; esac

arch="${platform#linux/}"
image="${image:-ship-xai:20261006-lock-${profile}-${arch}}"
evidence_dir="${evidence_dir:-${model_dir}/build_evidence/${profile}-${arch}}"
torch_index_url="https://download.pytorch.org/whl/${torch_index}"

build_command=(docker buildx build --load --progress plain --platform "$platform"
  --build-arg "BASE_IMAGE=$base_image" --build-arg "BUILD_PROFILE=$profile"
  --build-arg "TORCH_INDEX_URL=$torch_index_url"
  --metadata-file "$evidence_dir/build-metadata.json"
  --tag "$image" --file "$model_dir/Dockerfile" "$model_dir")
if ((dry_run)); then
  printf 'BUILD_RECIPE_ONLY (no Docker action performed)\n'
  print_command "${build_command[@]}"
  exit 0
fi

command -v docker >/dev/null || { printf 'docker CLI is unavailable. Build recipe remains unverified.\n' >&2; exit 127; }
docker info >/dev/null
docker buildx version >/dev/null
mkdir -p "$evidence_dir"
docker version > "$evidence_dir/docker-version.txt"
docker info > "$evidence_dir/docker-info.txt"
docker buildx ls > "$evidence_dir/buildx-ls.txt"
docker buildx imagetools inspect "$base_image" > "$evidence_dir/base-manifest.txt"
print_command "${build_command[@]}" > "$evidence_dir/build-command.txt"
"${build_command[@]}" 2>&1 | tee "$evidence_dir/build.log"
docker image inspect "$image" > "$evidence_dir/image-inspect.json"
docker run --rm --entrypoint python "$image" -c \
  'import json, sys, platform, importlib.metadata as m, torch; names=["torch","torchvision","numpy","scipy","quantus","pandas","captum","Pillow","opencv-python","scikit-image","scikit-learn"]; print(json.dumps({"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"versions":{n:m.version(n) for n in names},"torch_cuda_build":torch.version.cuda,"cuda_available_without_gpu_pass_through":torch.cuda.is_available()},indent=2))' \
  > "$evidence_dir/runtime-inventory.json"
docker run --rm --entrypoint cat "$image" /opt/submission/runtime-requirements-resolved.txt \
  > "$evidence_dir/runtime-requirements-resolved.txt"
printf 'Image built: %s\nEvidence: %s\nRun a container smoke check. GPU validation applies to a GPU deployment.\n' "$image" "$evidence_dir"
