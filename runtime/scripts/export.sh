#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: scripts/export.sh --image NAME:TAG --output PATH/image.tar [--smoke-log PASSED_SMOKE_LOG]\n'
}
image=""
output=""
smoke_log=""
while (($#)); do
  case "$1" in
    --image) image="${2:?missing image}"; shift 2 ;;
    --output) output="${2:?missing output path}"; shift 2 ;;
    --smoke-log) smoke_log="${2:?missing smoke log}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$image" && -n "$output" ]] || { usage >&2; exit 2; }
command -v docker >/dev/null || { printf 'docker CLI is unavailable. No image tar was created.\n' >&2; exit 127; }
[[ ! -e "$output" ]] || { printf 'Refusing to overwrite existing export: %s\n' "$output" >&2; exit 2; }
if [[ -n "$smoke_log" && ! -s "$smoke_log" ]]; then printf 'Smoke log is missing or empty.\n' >&2; exit 2; fi
mkdir -p "$(dirname "$output")"
docker image inspect "$image" > "${output}.image-inspect.json"
docker save --output "$output" "$image"
python3 - "$output" "$image" "$smoke_log" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
record={'image':sys.argv[2],'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'image_inspect':str(Path(str(p)+'.image-inspect.json').resolve()),'smoke_log':sys.argv[3] or None,'smoke_status':'caller_provided_log_requires_review' if sys.argv[3] else 'not_recorded','target_gpu_full9_status':'not_asserted_by_export'}
Path(str(p)+'.receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
PY
