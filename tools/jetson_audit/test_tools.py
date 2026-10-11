"""Small synthetic regression checks; no supplied submission or image needed."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import check_image_tar
import check_package
import test_admission_ast


def tar_bytes(files, links=()):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for name, target in links:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = tarfile.LNKTYPE, target
            archive.addfile(member)
    return stream.getvalue()


class AuditToolsTests(unittest.TestCase):
    def test_package_checks_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "sample"
            package.mkdir()
            for name in check_package.REQUIRED:
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                # A valid .py that would fail immediately if imported.
                path.write_bytes(b'raise RuntimeError("must not execute")\n' if name.endswith(".py") else b"synthetic")
            (package / "docker").mkdir()
            (package / "docker/image.tar").write_bytes(b"synthetic image placeholder")
            (package / "submission.csv").write_text("Id,Faithfulness,Max_Sensitivity,Sparsity\n1,0.5,3,0.9\n")
            entries = [{"path": str(path.relative_to(package)), "size": path.stat().st_size,
                        "sha256": check_package.sha256(path)} for path in package.rglob("*") if path.is_file()]
            (package / "manifest.json").write_text(json.dumps({"algorithm": "sha256", "package_root": "sample", "files": entries}))
            result = check_package.audit(package, 1)
            self.assertEqual(result["status"], "PASS", result["errors"])
            self.assertAlmostEqual(result["csv"]["attached_csv_formula_score"], 57.0)
            (package / "run_submission.py").write_text("invalid syntax here !")
            result = check_package.audit(package, 1)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("Manifest mismatch" in error for error in result["errors"]))
            self.assertTrue(any("Python syntax failed" in error for error in result["errors"]))

    def test_csv_rejects_malformed_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            for content in ("Id,Faithfulness,Max_Sensitivity,Sparsity\n",
                            "Id,Faithfulness,Max_Sensitivity,Sparsity\n1,0.5,NaN,0.9\n",
                            "Id,Faithfulness,Max_Sensitivity,Sparsity\n1,0.5,invalid,0.9\n",
                            "Id,Faithfulness,Max_Sensitivity,Sparsity\n1.5,0.5,1,0.9\n"):
                with self.subTest(content=content):
                    path.write_text(content)
                    self.assertFalse(check_package.check_csv(path, 1)["passed"])

    def test_manifest_paths_reject_traversal(self):
        for name in ("../escape", "/absolute", "app/../escape", "app\\escape", "", "./README.md"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_package.manifest_path(name)

    def test_csv_rejects_extra_and_missing_row_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            for row in ("1,0.5,3,0.9,unexpected", "1,0.5,3,0.9,", "1,0.5,3"):
                with self.subTest(row=row):
                    path.write_text("Id,Faithfulness,Max_Sensitivity,Sparsity\n" + row + "\n")
                    result = check_package.check_csv(path, 1)
                    self.assertFalse(result["passed"])
                    self.assertIn("CSV rows must have exactly four fields", result["errors"])

    def make_image(self, root, parent_replacement=False, corrupt_diff_id=False, directory_entries=()):
        old_readme, new_readme = b"old readme", b"new readme"
        layers = [tar_bytes({
            "app/README.md": old_readme, "app/run_submission.py": b"old runner",
            "app/script/metrix.py": b"synthetic scorer", "app/requirements-test.txt": b"",
            "app/checkpoints/old.pth": b"old synthetic checkpoint", "app/dist/demo.whl": b"synthetic wheel",
        }, [("app/custom_explain.py", "app/README.md")]), tar_bytes({
            "app/.wh.run_submission.py": b"", "app/run_submission.py": b"new runner",
            "app/README.md": new_readme, "app/checkpoints/.wh..wh..opq": b"",
            "app/checkpoints/new.pth": b"new synthetic checkpoint",
        })]
        if parent_replacement:
            layers.append(tar_bytes({"app/dist": b"a file replaces directory"}))
        if directory_entries:
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as directory_layer:
                for name in directory_entries:
                    member = tarfile.TarInfo(name)
                    member.type = tarfile.DIRTYPE
                    directory_layer.addfile(member)
            layers.append(stream.getvalue())
        diff_ids = ["sha256:" + hashlib.sha256(data).hexdigest() for data in layers]
        if corrupt_diff_id:
            diff_ids[0] = "sha256:" + "0" * 64
        config = json.dumps({"os": "linux", "architecture": "arm64", "rootfs": {"diff_ids": diff_ids}, "config": {}}).encode()
        config_name = hashlib.sha256(config).hexdigest() + ".json"
        layer_names = [f"layer-{index}/layer.tar" for index in range(len(layers))]
        members = {config_name: config, "manifest.json": json.dumps([{"Config": config_name, "Layers": layer_names, "RepoTags": ["synthetic:test"]}]).encode()}
        members.update(zip(layer_names, layers))
        archive = root / "image.tar"
        archive.write_bytes(tar_bytes(members))
        package = root / "package"
        package.mkdir()
        for name, data in {"README.md": new_readme, "run_submission.py": b"new runner",
                           "custom_explain.py": old_readme, "script/metrix.py": b"synthetic scorer",
                           "requirements-test.txt": b"", "checkpoints/new.pth": b"new synthetic checkpoint",
                           "dist/demo.whl": b"synthetic wheel"}.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return archive, package

    def test_image_whiteouts_and_hardlink_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, package = self.make_image(Path(directory))
            result = check_image_tar.Inspector(archive, package).run()
            self.assertEqual(result["status"], "PASS", result["issues"])
            files = result["images"][0]["effective_app_files"]
            self.assertEqual(len(files), 7)
            self.assertTrue(all(item["comparison"] == "MATCH" for item in files))
            self.assertFalse(any(item["path"].endswith("old.pth") for item in files))
            custom = next(item for item in files if item["path"] == "/app/custom_explain.py")
            self.assertEqual(custom["original_type"], "hardlink")

    def test_image_parent_replacement_hides_lower_files(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, package = self.make_image(Path(directory), parent_replacement=True)
            result = check_image_tar.Inspector(archive, package).run()
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("app/dist/<wheel>", result["images"][0]["missing_app_files"])

    def test_image_diff_id_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, package = self.make_image(Path(directory), corrupt_diff_id=True)
            result = check_image_tar.Inspector(archive, package).run()
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("diff_id mismatch" in issue for issue in result["issues"]))

    def test_image_directory_entries_replace_files_and_preserve_children(self):
        for replaces_file in (False, True):
            with self.subTest(replaces_file=replaces_file), tempfile.TemporaryDirectory() as directory:
                entries = ["app", "app/checkpoints", "app/dist"]
                if replaces_file:
                    entries.append("app/custom_explain.py")
                archive, package = self.make_image(Path(directory), directory_entries=entries)
                result = check_image_tar.Inspector(archive, package).run()
                self.assertEqual(result["status"], "FAIL" if replaces_file else "PASS", result["issues"])
                image = result["images"][0]
                files = {item["path"] for item in image["effective_app_files"]}
                self.assertIn("/app/checkpoints/new.pth", files)
                self.assertIn("/app/dist/demo.whl", files)
                if replaces_file:
                    self.assertNotIn("/app/custom_explain.py", files)
                    self.assertIn("app/custom_explain.py", image["missing_app_files"])
                else:
                    self.assertIn("/app/custom_explain.py", files)


if __name__ == "__main__":
    unittest.main()
