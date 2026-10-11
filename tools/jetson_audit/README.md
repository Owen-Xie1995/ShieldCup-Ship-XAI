# Submission audit tools

Small, reusable tools developed during an independent acceptance review of a
teammate's ship-image XAI submission. They make file integrity, runtime claims
and metric contracts separately reviewable. The reviewer found a valid
`Max_Sensitivity > 2` value was rejected before scoring; the teammate's revised
runner accepted finite nonnegative values, matching the scorer's clipping rule.

The audit tooling and synthetic cases were developed in this project workspace
with Codex assistance. The teammate supplied the E017 explanation algorithm and
runner. The organizer supplied the model, evaluation wheel and scoring code.
This directory does not claim authorship of those algorithms or redistribute
their source, checkpoints, wheels, competition images or container archives.
See [the case study](../../docs/jetson_audit/README.md) for results and boundaries.

## Requirements and quick check

Python 3.10 or newer. Package/image inspection and the bundled regression tests
use the standard library. The AST comparison additionally needs NumPy and
pandas; it was rerun with Python 3.12.9, NumPy 2.2.6 and pandas 3.0.3.
Install those two packages in your own virtual environment if needed.
Zstandard-compressed OCI layers additionally need `zstandard`; gzip, bzip2,
xz and uncompressed layers use the standard library. Docker and Torch are not
required for these tools.

From the repository root:

```bash
python3 -B tools/jetson_audit/test_tools.py -v
python3 -B tools/jetson_audit/check_package.py --help
python3 -B tools/jetson_audit/check_image_tar.py --help
python3 -B tools/jetson_audit/test_admission_ast.py --help
```

The eight regression tests generate small temporary packages and Docker-save
archives. They cover malformed/non-finite CSV data, manifest traversal,
checksum/syntax failure, whiteouts, opaque directories, hardlink contents,
extra/missing CSV row fields, file-to-directory replacement, preservation of
children under ordinary directory entries, parent replacement and uncompressed
layer digest corruption. The synthetic
`.pth`, `.whl` and image files contain placeholder bytes.

## Inspect an extracted package

```bash
python3 -B tools/jetson_audit/check_package.py /path/to/package-v2 \
  --rows 9 --output /tmp/package-audit.json

python3 -B tools/jetson_audit/check_package.py /path/to/package-v2 \
  --baseline-manifest /path/to/package-v1/manifest.json \
  --require-core-unchanged --rows 9 --output /tmp/package-comparison.json
```

The supported manifest has `algorithm: "sha256"`, `package_root` matching the
directory name, and `files` entries with `path`, `size` and `sha256`. Every file
except the manifest itself must be covered. The submission layout is specific
to this review: a runner, explainer, README, Dockerfile, dependency list,
checkpoint, organizer wheel, scorer, `submission.csv` and a `docker/*.tar` image.
Paths use forward slashes and package symlinks are rejected.

`--require-core-unchanged` compares the explainer, checkpoint, CSV, wheel,
scorer and dependency list to the baseline's recorded digests. A baseline
manifest must already be trusted; this flag does not authenticate its author.

Python files are compiled for syntax without executing them. CSV checks enforce
the exact ordered columns `Id,Faithfulness,Max_Sensitivity,Sparsity`, exactly
four fields in every row, IDs 1..N,
finite values, Faithfulness in [-1,1], nonnegative sensitivity and Sparsity in
[0,1]. The formula score is arithmetic from existing CSV values. It does not
recompute explanations or establish an official result.

## Inspect a saved image

```bash
python3 -B tools/jetson_audit/check_image_tar.py /path/to/image.tar \
  /path/to/extracted-package --output /tmp/image-audit.json
```

The inspector handles Docker-save and OCI layouts, SHA-256 descriptors and
uncompressed `rootfs.diff_ids`. It applies layer whiteouts, parent replacements
and file-to-directory replacements, preserves selected hardlink contents at the
layer where they appear, and compares selected effective `/app` artifacts against the extracted
package. It records declared platform and installed package METADATA without
importing those packages. A caller must examine the declared architecture; a
PASS does not enforce `arm64` or certify device compatibility.

This is a selected-file view. A link whose target lies outside selected files
cannot be resolved; the report fails that comparison. Full filesystem semantics,
Docker loading, ABI compatibility, CUDA execution, performance and numerical
model output require separate runtime checks. Archives are read without
extraction. Large archives still require time and memory for metadata inspection.

## Compare reviewed admission and scoring functions

First manually review both `strict_admission` functions, `EXPECTED_COLUMNS`,
`METRIC_RANGES`, and the scorer's `score` function. Then:

```bash
python3 -B tools/jetson_audit/test_admission_ast.py \
  --v1-runner /path/to/package-v1/run_submission.py \
  --v2-runner /path/to/package-v2/run_submission.py \
  --scorer /path/to/package-v2/script/metrix.py \
  --reviewed-source --output /tmp/admission-comparison.json
```

The tool selects the actual reviewed definitions with AST parsing, checks that
the admission function bodies are unchanged, and runs 34 boundary/schema/ID
cases against both versions plus five scorer cases. It expects the original
range fix: v1 rejects finite sensitivity above 2; v2 accepts it. A different
change needs new expectations. No runner import, Torch import or model loading
is needed. **AST selection is not a security sandbox:** a selected function can
execute arbitrary code when called, so never run this tool on unreviewed source.

Every CLI uses exit code 0 for PASS and 1 for audit failure; argparse uses 2 for
invalid arguments. Write JSON outside the inspected package. Generated evidence
may include local paths and image commands/tags; review it before publication.
The curated public case study contains only summarized evidence.

The original one-off 7z verification/extraction script depended on a local
7zip binary and package-specific paths. It is represented by the historical
integrity result in the case study; this toolkit begins with an extracted
package or image tar.
