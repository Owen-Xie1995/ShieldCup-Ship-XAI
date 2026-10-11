#!/usr/bin/env python3
"""Inspect Docker-save/OCI tar contents without extraction or image execution."""

import argparse
import bz2
import email.parser
import gzip
import hashlib
import io
import json
import lzma
import posixpath
import re
import tarfile
from pathlib import Path


CHUNK = 1024 * 1024
SHA_NAME = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = (
    "app/custom_explain.py", "app/run_submission.py", "app/README.md",
    "app/script/metrix.py",
)


def virtual_path(value, absolute_ok=False):
    if value.startswith("/") and not absolute_ok:
        raise ValueError(f"Absolute archive reference: {value!r}")
    value = value.lstrip("/") if absolute_ok else value
    while value.startswith("./"):
        value = value[2:]
    if ".." in value.split("/"):
        raise ValueError(f"Traversal archive member: {value!r}")
    return posixpath.normpath(value) if value not in ("", ".") else ""


def named_sha(path):
    parts = path.split("/")
    if len(parts) >= 3 and parts[-3:-1] == ["blobs", "sha256"] and SHA_NAME.fullmatch(parts[-1]):
        return parts[-1]
    # Classic Docker config names are their content digest. Layer-directory
    # names in legacy saves are NOT necessarily hashes of layer.tar bytes.
    name = parts[-1].removesuffix(".json")
    return name if parts[-1].endswith(".json") and SHA_NAME.fullmatch(name) else None


def wanted(path):
    return (
        path in REQUIRED
        or (path.startswith("app/checkpoints/") and path.endswith((".pth", ".pt")))
        or (path.startswith("app/dist/") and path.endswith(".whl"))
        or (posixpath.dirname(path) == "app" and posixpath.basename(path).startswith("requirements"))
        or path.endswith(".dist-info/METADATA")
    )


class HashReader(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self.source.read(size)
        if data:
            self.digest.update(data)
            self.count += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def drain(self):
        while self.read(CHUNK):
            pass


class Inspector:
    def __init__(self, archive, package):
        self.archive = archive.resolve()
        self.package = package.resolve()
        self.tar = tarfile.open(archive, mode="r:*")
        self.members = {}
        self.issues = []
        self.warnings = []
        self.metadata_checks = {}
        self.layer_cache = {}
        for member in self.tar.getmembers():
            try:
                name = virtual_path(member.name)
            except ValueError as exc:
                self.issues.append(str(exc))
                continue
            if name in self.members:
                self.issues.append(f"Duplicate outer tar member: {name}")
            self.members[name] = member

    def open_member(self, path):
        path = virtual_path(path)
        member = self.members.get(path)
        if member is None or not member.isfile():
            raise ValueError(f"Missing/nonregular referenced blob: {path}")
        return self.tar.extractfile(member)

    def json_blob(self, path, expected=None):
        path = virtual_path(path)
        member = self.members.get(path)
        if member is None or member.size > 16 * CHUNK:
            raise ValueError(f"Missing/oversized JSON metadata: {path}")
        with self.open_member(path) as stream:
            data = stream.read()
        actual = hashlib.sha256(data).hexdigest()
        inferred = named_sha(path)
        expected = expected.removeprefix("sha256:") if expected else inferred
        matched = actual == expected if expected else None
        self.metadata_checks[path] = {
            "path": path, "bytes": len(data), "sha256": actual,
            "expected_sha256": expected, "digest_match": matched,
        }
        if matched is False:
            self.issues.append(f"Metadata digest mismatch: {path}")
        return json.loads(data)

    def descriptor_path(self, descriptor):
        digest = descriptor["digest"]
        algorithm, value = digest.split(":", 1)
        if algorithm != "sha256" or not SHA_NAME.fullmatch(value):
            raise ValueError(f"Unsupported OCI digest: {digest}")
        path = f"blobs/sha256/{value}"
        member = self.members.get(path)
        if member is not None and descriptor.get("size") not in (None, member.size):
            self.issues.append(f"Descriptor size mismatch: {path}")
        return path

    def image_specs(self):
        specs = []
        if "manifest.json" in self.members:
            manifest = self.json_blob("manifest.json")
            if isinstance(manifest, list):
                for entry in manifest:
                    specs.append({
                        "config": entry["Config"], "layers": entry["Layers"],
                        "tags": entry.get("RepoTags") or [], "layout": "docker-save",
                    })

        def visit(descriptor, inherited_tags=(), depth=0):
            if depth > 8:
                raise ValueError("OCI index nesting exceeds 8 levels")
            path = self.descriptor_path(descriptor)
            document = self.json_blob(path, descriptor["digest"])
            tag = descriptor.get("annotations", {}).get("org.opencontainers.image.ref.name")
            tags = list(inherited_tags) + ([tag] if tag else [])
            if "manifests" in document:
                for child in document["manifests"]:
                    visit(child, tags, depth + 1)
            elif "config" in document and "layers" in document:
                specs.append({
                    "config": self.descriptor_path(document["config"]),
                    "config_digest": document["config"]["digest"],
                    "layers": [self.descriptor_path(layer) for layer in document["layers"]],
                    "layer_digests": [layer["digest"] for layer in document["layers"]],
                    "tags": tags, "platform_descriptor": descriptor.get("platform"),
                    "layout": "oci", "manifest": path,
                })
            else:
                self.warnings.append(f"Skipped non-image OCI descriptor: {path}")

        if "index.json" in self.members:
            index = self.json_blob("index.json")
            for descriptor in index.get("manifests", []):
                visit(descriptor)
        # docker save may provide both interfaces to the very same image.
        unique = {}
        for spec in specs:
            key = (spec["config"], tuple(spec["layers"]))
            if key in unique:
                prior = unique[key]
                prior["tags"] = sorted(set(prior["tags"] + spec["tags"]))
                prior["layout"] = "docker-save+oci"
                for field in ("config_digest", "layer_digests", "platform_descriptor", "manifest"):
                    if field in spec:
                        prior[field] = spec[field]
            else:
                unique[key] = spec
        if not unique:
            raise ValueError("No Docker-save or OCI image manifests found")
        return list(unique.values())

    def layer(self, path):
        path = virtual_path(path)
        if path in self.layer_cache:
            return self.layer_cache[path]
        result = {"path": path, "changes": {}, "deletions": [], "opaque_dirs": [],
                  "replacements": set(), "directories": set()}
        with self.open_member(path) as source:
            magic = source.read(6)
            source.seek(0)
            raw = HashReader(source)
            decoder = raw
            try:
                if magic.startswith(b"\x1f\x8b"):
                    decoder = gzip.GzipFile(fileobj=raw)
                    result["compression"] = "gzip"
                elif magic.startswith(b"BZh"):
                    decoder = bz2.BZ2File(raw)
                    result["compression"] = "bzip2"
                elif magic.startswith(b"\xfd7zXZ\x00"):
                    decoder = lzma.LZMAFile(raw)
                    result["compression"] = "xz"
                elif magic.startswith(b"\x28\xb5\x2f\xfd"):
                    import zstandard  # Optional; never installed by this helper.
                    decoder = zstandard.ZstdDecompressor().stream_reader(raw, closefd=False)
                    result["compression"] = "zstd"
                else:
                    result["compression"] = "none"
                expanded = HashReader(decoder)
                with tarfile.open(fileobj=expanded, mode="r|") as layer_tar:
                    for member in layer_tar:
                        name = virtual_path(member.name, absolute_ok=True)
                        base = posixpath.basename(name)
                        parent = posixpath.dirname(name)
                        if base == ".wh..wh..opq":
                            result["opaque_dirs"].append(parent)
                            continue
                        if base.startswith(".wh."):
                            result["deletions"].append(posixpath.join(parent, base[4:]))
                            continue
                        if member.ischr() and member.devmajor == member.devminor == 0:
                            result["deletions"].append(name)
                            continue
                        if member.isdir():
                            # A directory may replace a lower regular file or
                            # link at this exact path. Ordinary directory entries
                            # keep lower descendants; opaque markers remove them.
                            result["directories"].add(name)
                            result["changes"].pop(name, None)
                            continue
                        result["directories"].discard(name)
                        result["replacements"].add(name)
                        if not wanted(name):
                            continue
                        record = {"path": "/" + name, "layer": path}
                        if member.isfile():
                            digest = hashlib.sha256()
                            capture = name.endswith(".dist-info/METADATA") or name == "app/README.md"
                            text_bytes = bytearray()
                            count = 0
                            with layer_tar.extractfile(member) as contents:
                                while data := contents.read(CHUNK):
                                    digest.update(data)
                                    count += len(data)
                                    if capture and len(text_bytes) < CHUNK:
                                        text_bytes.extend(data[:CHUNK - len(text_bytes)])
                            record.update(type="file", bytes=count, sha256=digest.hexdigest())
                            if count != member.size:
                                self.issues.append(f"Layer file size mismatch: {name}")
                            if capture:
                                record["text"] = text_bytes.decode("utf-8", errors="replace")
                        elif member.issym() or member.islnk():
                            record.update(type="symlink" if member.issym() else "hardlink", target=member.linkname)
                        else:
                            record["type"] = "special"
                        result["changes"][name] = record
                # tar iteration can finish before padding/trailing bytes.
                expanded.drain()
                result["uncompressed_bytes"] = expanded.count
                result["diff_id"] = "sha256:" + expanded.digest.hexdigest()
                result["walk_complete"] = True
            except Exception as exc:
                result["walk_complete"] = False
                result["error"] = f"{type(exc).__name__}: {exc}"
                self.issues.append(f"Layer inspection failed {path}: {result['error']}")
            finally:
                if decoder is not raw:
                    decoder.close()
                raw.drain()
                result["bytes"] = raw.count
                result["sha256"] = raw.digest.hexdigest()
        result["expected_sha256"] = named_sha(path)
        result["digest_match"] = (result["sha256"] == result["expected_sha256"]
                                  if result["expected_sha256"] else None)
        if result["digest_match"] is False:
            self.issues.append(f"Layer blob digest mismatch: {path}")
        self.layer_cache[path] = result
        return result

    @staticmethod
    def under(path, directory):
        return not directory or path == directory or path.startswith(directory + "/")

    def resolve(self, name, state, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in state:
            return None
        seen.add(name)
        record = state[name]
        if record["type"] == "file":
            return record
        if record["type"] not in ("symlink", "hardlink"):
            return None
        target = record["target"]
        if record["type"] == "symlink" and not target.startswith("/"):
            target = posixpath.join(posixpath.dirname(name), target)
        target = posixpath.normpath(target.lstrip("/"))
        if target == ".." or target.startswith("../"):
            return None
        return self.resolve(target, state, seen)

    def compare_files(self, state):
        files = []
        for name in sorted(path for path in state if path.startswith("app/")):
            original = state[name]
            resolved = self.resolve(name, state)
            record = {key: value for key, value in original.items() if key != "text"}
            record["outer_path"] = name.removeprefix("app/")
            outer = self.package / record["outer_path"]
            if resolved is None:
                record["comparison"] = "UNRESOLVED_IMAGE_LINK_OR_SPECIAL_FILE"
                self.issues.append(f"Cannot resolve selected image file: /{name}")
            elif not outer.is_file() or not outer.resolve().is_relative_to(self.package):
                record["comparison"] = "OUTER_FILE_MISSING"
                self.issues.append(f"Outer counterpart missing: {outer}")
            else:
                outer_digest = hashlib.sha256()
                with outer.open("rb") as source:
                    while data := source.read(CHUNK):
                        outer_digest.update(data)
                record.update(image_sha256=resolved["sha256"], image_bytes=resolved["bytes"],
                              outer_sha256=outer_digest.hexdigest(), outer_bytes=outer.stat().st_size)
                match = record["image_sha256"] == record["outer_sha256"] and record["image_bytes"] == record["outer_bytes"]
                record["comparison"] = "MATCH" if match else "MISMATCH"
                if not match:
                    self.issues.append(f"Image/outer mismatch: /{name}")
            files.append(record)
        missing = [name for name in REQUIRED if name not in state]
        if not any(path.startswith("app/checkpoints/") for path in state):
            missing.append("app/checkpoints/<checkpoint>")
        if not any(path.startswith("app/dist/") and path.endswith(".whl") for path in state):
            missing.append("app/dist/<wheel>")
        if not any(posixpath.dirname(path) == "app" and posixpath.basename(path).startswith("requirements") for path in state):
            missing.append("app/requirements*")
        for name in missing:
            self.issues.append(f"Effective selected file missing: /{name}")
        return files, missing

    def image(self, spec):
        config = self.json_blob(spec["config"], spec.get("config_digest"))
        runtime = config.get("config") or {}
        expected_diff_ids = config.get("rootfs", {}).get("diff_ids", [])
        state, layer_reports = {}, []
        if expected_diff_ids and len(expected_diff_ids) != len(spec["layers"]):
            self.issues.append(f"rootfs diff_id/layer count mismatch: {spec['config']}")
        for index, path in enumerate(spec["layers"]):
            layer = self.layer(path)
            report = {key: value for key, value in layer.items()
                      if key not in ("changes", "deletions", "opaque_dirs", "replacements", "directories")}
            explicit = spec.get("layer_digests", [])
            if index < len(explicit):
                report["descriptor_digest_match"] = "sha256:" + layer["sha256"] == explicit[index]
                if not report["descriptor_digest_match"]:
                    self.issues.append(f"Layer descriptor digest mismatch: {path}")
            if index < len(expected_diff_ids):
                report["expected_diff_id"] = expected_diff_ids[index]
                report["diff_id_match"] = layer.get("diff_id") == expected_diff_ids[index]
                if not report["diff_id_match"]:
                    self.issues.append(f"Uncompressed rootfs diff_id mismatch: {path}")
            report.update(selected_files=len(layer["changes"]), whiteouts=len(layer["deletions"]),
                          opaque_dirs=len(layer["opaque_dirs"]))
            layer_reports.append(report)
            # Apply whiteouts to the lower filesystem before same-layer files.
            for directory in layer["deletions"] + layer["opaque_dirs"]:
                state = {name: entry for name, entry in state.items() if not self.under(name, directory)}
            # Any non-directory parent replacement hides lower descendants,
            # even when that parent's contents are not selected for hashing.
            state = {
                path: entry for path, entry in state.items()
                if path not in layer["directories"]
                and not any("/".join(path.split("/")[:part]) in layer["replacements"]
                            for part in range(1, len(path.split("/"))))
            }
            state.update(layer["changes"])
            # Hardlinks preserve the target inode's contents when a later
            # layer replaces the target path; do not resolve them only after
            # applying the entire image. Symlinks remain path references.
            for name, entry in layer["changes"].items():
                if entry["type"] != "hardlink":
                    continue
                resolved = self.resolve(name, state)
                if resolved is not None:
                    state[name] = dict(entry, type="file", original_type="hardlink",
                                       bytes=resolved["bytes"], sha256=resolved["sha256"])
                    if "text" in resolved:
                        state[name]["text"] = resolved["text"]
        files, missing = self.compare_files(state)
        versions = {}
        for path in sorted(state):
            if not path.endswith(".dist-info/METADATA"):
                continue
            entry = self.resolve(path, state)
            if entry is None or "text" not in entry:
                self.warnings.append(f"Unresolved package metadata: /{path}")
                continue
            headers = email.parser.Parser().parsestr(entry["text"], headersonly=True)
            if headers.get("Name") and headers.get("Version"):
                versions.setdefault(headers["Name"], []).append({"version": headers["Version"], "path": "/" + path})
        return {
            "layout": spec["layout"], "config": spec["config"], "tags": spec["tags"],
            "platform": {key: config.get(key) for key in ("os", "architecture", "variant") if config.get(key)},
            "platform_descriptor": spec.get("platform_descriptor"),
            "entrypoint": runtime.get("Entrypoint"), "command": runtime.get("Cmd"),
            "working_directory": runtime.get("WorkingDir"),
            "layers": layer_reports,
            "effective_app_files": files, "missing_app_files": missing,
            "package_versions": versions,
            "scope": "Selected /app files and dist-info METADATA; no image execution or installed-code imports",
        }

    def run(self):
        images = []
        try:
            for spec in self.image_specs():
                images.append(self.image(spec))
        except Exception as exc:
            self.issues.append(f"Inspection failed: {type(exc).__name__}: {exc}")
        finally:
            self.tar.close()
        return {
            "status": "PASS" if not self.issues else "FAIL",
            "image_tar": str(self.archive), "image_tar_bytes": self.archive.stat().st_size,
            "package_root": str(self.package), "images": images,
            "metadata_digest_checks": list(self.metadata_checks.values()),
            "issues": sorted(set(self.issues)), "warnings": sorted(set(self.warnings)),
            "boundary": "Read-only tar inspection. Does not verify Docker load, ABI, CUDA, GPU execution, or numeric scoring.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_tar", type=Path)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON evidence destination")
    args = parser.parse_args()
    archive, package = args.image_tar.resolve(), args.package_root.resolve()
    if not archive.is_file() or not package.is_dir():
        parser.error("image_tar must be a file and package_root must be an extracted package directory")
    output = args.output.resolve()
    if output == archive or output == package or package in output.parents:
        parser.error("JSON output must be outside the read-only package and must not overwrite image_tar")
    try:
        result = Inspector(archive, package).run()
    except Exception as exc:
        result = {"status": "FAIL", "issues": [f"{type(exc).__name__}: {exc}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "images": len(result.get("images", [])),
                      "issues": result["issues"], "output": str(args.output.resolve())}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
