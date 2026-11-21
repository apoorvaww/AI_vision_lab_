"""
tests/test_security_and_hardening.py
-----------------------------------
Security and hardening verification tests for AI VisionLab:
1. Job ID validation & path resolution
2. Secure ZIP archive extraction & ZIP bomb mitigations
3. Persistent job recovery & state validation
4. Route parameter validation & exception handling
5. Health check probe behavior (liveness vs. readiness)
"""

import io
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from app import (
    app,
    is_valid_job_id,
    _resolve_result_path,
    _validate_archive_member,
    _extract_dataset,
    _job,
    JOBS,
    JOBS_LOCK,
)
import config


class TestJobIDValidation(unittest.TestCase):
    """Test strict job identifier format and safe path resolution."""

    def test_valid_job_ids(self):
        valid_ids = [
            "0123456789ab",
            "abcdef012345",
            "deadbeefcafe",
            "1a2b3c4d5e6f",
        ]
        for jid in valid_ids:
            self.assertTrue(is_valid_job_id(jid), f"Expected '{jid}' to be valid.")

    def test_invalid_job_ids(self):
        invalid_ids = [
            "ABCDEF012345",         # uppercase hex
            "0123456789a",          # 11 characters (too short)
            "0123456789abc",        # 13 characters (too long)
            "g123456789ab",         # non-hex character 'g'
            "../etc/passwd",        # directory traversal
            "../../app012",         # directory traversal
            "012345/../../",        # traversal inside
            "job-12345678",         # hyphens
            "0123456789ab\x00",     # null byte
            "",                     # empty string
            None,                   # non-string
            123456789012,           # int
            ["0123456789ab"],       # list
        ]
        for jid in invalid_ids:
            self.assertFalse(is_valid_job_id(jid), f"Expected '{jid}' to be invalid.")

    def test_resolve_result_path_containment(self):
        valid_id = "0123456789ab"
        resolved = _resolve_result_path(valid_id)
        self.assertIsNotNone(resolved)
        real_results = os.path.realpath(config.RESULTS_DIR)
        self.assertEqual(os.path.dirname(resolved), real_results)
        self.assertTrue(resolved.endswith(f"{valid_id}.json"))

    def test_resolve_result_path_rejects_traversal(self):
        bad_ids = [
            "../secrets",
            "../../etc/passwd",
            "012345/../evil",
        ]
        for bad_id in bad_ids:
            self.assertIsNone(_resolve_result_path(bad_id))


class TestSecureZipExtraction(unittest.TestCase):
    """Test robust archive member validation, path traversal prevention, and extraction limits."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="visionlab_test_zip_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_archive_member_valid(self):
        zinfo = zipfile.ZipInfo("dataset/cats/cat_1.jpg")
        resolved = _validate_archive_member(zinfo, self.temp_dir)
        real_temp = os.path.realpath(self.temp_dir)
        self.assertTrue(resolved.startswith(real_temp))
        self.assertTrue(resolved.endswith(os.path.join("dataset", "cats", "cat_1.jpg")))

    def test_validate_archive_member_posix_traversal(self):
        traversals = [
            "../evil.txt",
            "../../evil.txt",
            "cats/../../evil.txt",
            "cats/./../../evil.txt",
        ]
        for path in traversals:
            zinfo = zipfile.ZipInfo(path)
            with self.assertRaises(ValueError, msg=f"Should reject traversal: {path}"):
                _validate_archive_member(zinfo, self.temp_dir)

    def test_validate_archive_member_windows_backslash_traversal(self):
        traversals = [
            "..\\evil.txt",
            "..\\..\\evil.txt",
            "cats\\..\\..\\evil.txt",
        ]
        for path in traversals:
            zinfo = zipfile.ZipInfo(path)
            with self.assertRaises(ValueError, msg=f"Should reject backslash traversal: {path}"):
                _validate_archive_member(zinfo, self.temp_dir)

    def test_validate_archive_member_absolute_paths(self):
        abs_paths = [
            "/etc/passwd",
            "/tmp/evil.txt",
        ]
        for path in abs_paths:
            zinfo = zipfile.ZipInfo(path)
            with self.assertRaises(ValueError, msg=f"Should reject absolute path: {path}"):
                _validate_archive_member(zinfo, self.temp_dir)

    def test_validate_archive_member_windows_drive_letter(self):
        drive_paths = [
            "C:/windows/system32/calc.exe",
            "D:\\data\\hack.py",
            "c:/evil.txt",
        ]
        for path in drive_paths:
            zinfo = zipfile.ZipInfo(path)
            with self.assertRaises(ValueError, msg=f"Should reject drive letter: {path}"):
                _validate_archive_member(zinfo, self.temp_dir)

    def test_validate_archive_member_symlink(self):
        zinfo = zipfile.ZipInfo("symlink_target")
        # 0o120000 indicates a symbolic link in POSIX file attributes
        zinfo.external_attr = 0o120777 << 16
        with self.assertRaises(ValueError):
            _validate_archive_member(zinfo, self.temp_dir)

    def test_extract_dataset_cleanup_macos_metadata(self):
        # Create a zip containing valid dataset and macOS metadata (.DS_Store, __MACOSX, ._*)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dataset/class_a/img1.txt", "sample image data")
            zf.writestr("dataset/class_b/img2.txt", "sample image data")
            zf.writestr("dataset/.DS_Store", "dummy metadata")
            zf.writestr("dataset/class_a/._img1.txt", "dummy apple double")
            zf.writestr("__MACOSX/dataset/._class_a", "dummy resource fork")

        zip_buf.seek(0)
        zip_path = os.path.join(self.temp_dir, "test.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

        extract_dest = os.path.join(self.temp_dir, "extracted")
        root = _extract_dataset(zip_path, extract_dest)

        self.assertTrue(os.path.isdir(root))
        # Verify macOS metadata was purged
        for cur, dirs, files in os.walk(extract_dest):
            for d in dirs:
                self.assertNotEqual(d.lower(), "__macosx")
            for f in files:
                self.assertFalse(f.startswith("._"))
                self.assertNotEqual(f, ".DS_Store")

    def test_extract_dataset_rejects_malicious_archive_and_cleans_destination(self):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("normal.txt", "safe file")
            zf.writestr("../traversal.txt", "malicious payload")

        zip_buf.seek(0)
        zip_path = os.path.join(self.temp_dir, "malicious.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

        extract_dest = os.path.join(self.temp_dir, "extracted_fail")
        with self.assertRaises(ValueError):
            _extract_dataset(zip_path, extract_dest)

        # Ensure failed extraction cleans up directory
        self.assertFalse(os.path.exists(extract_dest))

    def test_extract_dataset_file_count_limit(self):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(15):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        zip_buf.seek(0)
        zip_path = os.path.join(self.temp_dir, "too_many_files.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

        extract_dest = os.path.join(self.temp_dir, "extracted_limit")
        with patch.object(config, "MAX_EXTRACT_FILES", 10):
            with self.assertRaises(ValueError) as ctx:
                _extract_dataset(zip_path, extract_dest)
            self.assertIn("maximum allowable file count", str(ctx.exception).lower())

    def test_extract_dataset_byte_limit(self):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("large.txt", "A" * 1024)

        zip_buf.seek(0)
        zip_path = os.path.join(self.temp_dir, "too_large.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

        extract_dest = os.path.join(self.temp_dir, "extracted_bytes")
        with patch.object(config, "MAX_EXTRACT_BYTES", 500):
            with self.assertRaises(ValueError) as ctx:
                _extract_dataset(zip_path, extract_dest)
            self.assertIn("maximum allowable uncompressed size", str(ctx.exception).lower())


class TestPersistentJobRecovery(unittest.TestCase):
    """Test job recovery from disk, state field population, and corrupted file handling."""

    def setUp(self):
        self.temp_results_dir = tempfile.mkdtemp(prefix="visionlab_results_")
        self.orig_results_dir = config.RESULTS_DIR
        config.RESULTS_DIR = self.temp_results_dir
        with JOBS_LOCK:
            JOBS.clear()

    def tearDown(self):
        config.RESULTS_DIR = self.orig_results_dir
        shutil.rmtree(self.temp_results_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS.clear()

    def test_job_restores_complete_fields_from_valid_result(self):
        job_id = "0123456789ab"
        sample_result = {
            "best_config": {
                "model_name": "Small CNN (from scratch)",
                "split_label": "80% train / 20% test",
                "split_ratio": 0.8,
                "expected_accuracy": 0.95,
                "expected_f1_macro": 0.94,
                "params": {"epochs": 5, "batch_size": 16, "learning_rate": 0.001, "image_size": 96},
                "xai_enabled": True,
                "xai_method": "gradcam",
                "confusion_matrix": [[10, 0], [1, 9]],
                "class_names": ["cat", "dog"],
                "prediction_sample": [],
            },
            "leaderboard": [
                {
                    "model_name": "Small CNN (from scratch)",
                    "model_family": "dl",
                    "split_label": "80% train / 20% test",
                    "accuracy": 0.95,
                    "f1_macro": 0.94,
                    "train_seconds": 12.3,
                    "params": {"epochs": 5},
                }
            ],
            "dataset_summary": {
                "root": "/tmp/test_dataset",
                "num_valid_images": 20,
                "num_classes": 2,
                "class_counts": {"cat": 10, "dog": 10},
            },
            "characteristics": {"avg_width": 96, "avg_height": 96},
            "recommendations": {},
            "search_metric": "f1_macro",
            "num_trials_run": 1,
        }

        result_path = os.path.join(self.temp_results_dir, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(sample_result, f)

        recovered = _job(job_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["status"], "done")
        self.assertEqual(recovered["pct"], 100)
        self.assertIn("result", recovered)
        self.assertEqual(recovered["dataset_name"], "test_dataset")
        self.assertEqual(recovered["selected_models"], ["Small CNN (from scratch)"])
        self.assertIn("small_cnn", recovered["selected_keys"])
        self.assertEqual(recovered["epochs"], 5)
        self.assertEqual(recovered["batch_size"], 16)
        self.assertEqual(recovered["learning_rate"], 0.001)
        self.assertEqual(recovered["image_size"], 96)
        self.assertEqual(recovered["metric"], "f1_macro")
        self.assertEqual(recovered["split_ratios"], [0.8])
        self.assertTrue(recovered["xai_enabled"])

    def test_job_restores_cancelled_state(self):
        job_id = "0123456789ac"
        cancelled_result = {
            "cancelled": True,
            "dataset_summary": {"root": "/tmp/cancelled_ds"},
        }
        result_path = os.path.join(self.temp_results_dir, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(cancelled_result, f)

        recovered = _job(job_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["status"], "cancelled")
        self.assertTrue(recovered["cancel_requested"])

    def test_job_rejects_corrupted_json(self):
        job_id = "0123456789ad"
        result_path = os.path.join(self.temp_results_dir, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content ...")

        recovered = _job(job_id)
        self.assertIsNone(recovered)

    def test_job_rejects_incomplete_schema(self):
        job_id = "0123456789ae"
        incomplete_result = {
            "some_key": "some_value"
            # Missing best_config, leaderboard, dataset_summary
        }
        result_path = os.path.join(self.temp_results_dir, f"{job_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(incomplete_result, f)

        recovered = _job(job_id)
        self.assertIsNone(recovered)

    def test_job_does_not_claim_interrupted_run_resumes(self):
        # A non-existent file on disk returns None
        job_id = "0123456789af"
        recovered = _job(job_id)
        self.assertIsNone(recovered)


class TestRouteSecurity(unittest.TestCase):
    """Test HTTP routes for invalid job_id rejection and path traversal protection."""

    def setUp(self):
        self.client = app.test_client()

    def test_api_progress_invalid_job_id(self):
        resp = self.client.get("/api/progress/invalid_id")
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "invalid_id")

    def test_api_progress_traversal_job_id(self):
        # Path traversal normalized away from route returns 404
        resp = self.client.get("/api/progress/../../etc/passwd")
        self.assertEqual(resp.status_code, 404)

        # URL-encoded or path component traversal reaching view returns 400
        resp_encoded = self.client.get("/api/progress/..%2F..%2Fetc%2Fpasswd")
        self.assertIn(resp_encoded.status_code, [400, 404])
        if resp_encoded.status_code == 400:
            self.assertEqual(resp_encoded.get_json().get("status"), "invalid_id")

    def test_api_cancel_invalid_job_id(self):
        resp = self.client.post("/api/cancel/invalid_id")
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("ok"))

    def test_dataset_image_invalid_job_id(self):
        resp = self.client.get("/dataset-image/invalid_id/test.png")
        self.assertEqual(resp.status_code, 404)

    def test_preview_transform_invalid_job_id(self):
        resp = self.client.get("/preview-transform/invalid_id/test.png")
        self.assertEqual(resp.status_code, 404)

    def test_download_routes_invalid_job_id(self):
        for ext in ["csv", "notebook", "pdf"]:
            resp = self.client.get(f"/download/invalid_id/{ext}")
            self.assertEqual(resp.status_code, 404, f"Failed for {ext}")

    def test_results_page_invalid_job_id_redirects_safely(self):
        resp = self.client.get("/results/invalid_id")
        # Redirects to /upload with flash message
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/upload", resp.headers.get("Location", ""))


class TestUploadSecurity(unittest.TestCase):
    """Test upload route validation, malicious archive rejection, and error message sanitization."""

    def setUp(self):
        self.client = app.test_client()
        self.temp_upload_dir = tempfile.mkdtemp(prefix="visionlab_upload_")
        self.orig_upload_dir = config.UPLOAD_DIR
        config.UPLOAD_DIR = self.temp_upload_dir

    def tearDown(self):
        config.UPLOAD_DIR = self.orig_upload_dir
        shutil.rmtree(self.temp_upload_dir, ignore_errors=True)

    def test_upload_no_file_rejected(self):
        resp = self.client.post("/start", data={}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Please choose a dataset .zip file to upload", resp.data)

    def test_upload_non_zip_rejected(self):
        data = {"dataset": (io.BytesIO(b"hello world"), "test.txt")}
        resp = self.client.post("/start", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Dataset must be uploaded as a .zip file", resp.data)

    def test_upload_corrupt_zip_rejected(self):
        data = {"dataset": (io.BytesIO(b"not a real zip content"), "bad.zip")}
        resp = self.client.post("/start", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"That file doesn&#39;t look like a valid .zip archive", resp.data)

    def test_upload_zip_with_traversal_rejected_without_path_disclosure(self):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("../../evil.txt", "payload")
        zip_buf.seek(0)

        data = {"dataset": (zip_buf, "traversal.zip")}
        resp = self.client.post("/start", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Archive validation rejected", resp.data)
        # Ensure server host paths are not leaked in the flash message
        self.assertNotIn(b"/Users/", resp.data)
        self.assertNotIn(b"/app/webapp_data", resp.data)

    def test_upload_valid_zip_initializes_job(self):
        from PIL import Image

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            # Create a 2-class dataset structure with unique image contents
            for c_idx, cls in enumerate(["cat", "dog"]):
                for i in range(2):
                    img = Image.new("RGB", (32, 32), color=(50 + c_idx * 60, 80 + i * 40, 120 + (c_idx + i) * 15))
                    # Draw a distinct pixel so hashes differ
                    img.putpixel((i, c_idx), (255, 255, 255))
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="JPEG")
                    zf.writestr(f"my_dataset/{cls}/img_{i}.jpg", img_buf.getvalue())
        zip_buf.seek(0)

        data = {"dataset": (zip_buf, "test_dataset.zip"), "dataset_name": "Test Pets"}
        resp = self.client.post("/start", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 302)
        loc = resp.headers.get("Location", "")
        self.assertIn("/preview/", loc)

        job_id = loc.split("/preview/")[1]
        self.assertTrue(is_valid_job_id(job_id))
        self.assertIn(job_id, JOBS)
        self.assertEqual(JOBS[job_id]["dataset_name"], "Test Pets")


class TestHealthCheckProbe(unittest.TestCase):
    """Test health check readiness and liveness probe implementations."""

    def setUp(self):
        self.client = app.test_client()

    def test_liveness_probe(self):
        resp = self.client.get("/api/health?probe=liveness")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("probe"), "liveness")

    def test_readiness_probe_healthy(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("probe"), "readiness")
        self.assertEqual(data.get("storage"), "writable")
        self.assertGreater(data.get("available_models", 0), 0)

    def test_readiness_probe_unhealthy_when_storage_unwritable(self):
        with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
            resp = self.client.get("/api/health")
            self.assertEqual(resp.status_code, 503)
            data = resp.get_json()
            self.assertEqual(data.get("status"), "unhealthy")
            self.assertIn("errors", data)
            # Ensure raw paths are NOT exposed to caller
            for err in data["errors"]:
                self.assertNotIn("/Users/", err)
                self.assertNotIn("/app/", err)


if __name__ == "__main__":
    unittest.main()
