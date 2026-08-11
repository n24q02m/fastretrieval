"""Tests for model management utility functions."""

import base64
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.model_management import ModelManagement

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def make_repo_file(path: str, size: int = 100, oid: str = "abc123") -> RepoFile:
    """Create a RepoFile instance for testing."""
    return RepoFile(path=path, size=size, oid=oid)


def make_tar_gz(tmp_path: Path, inner_name: str = "model.onnx") -> Path:
    """Create a minimal valid .tar.gz archive and return its path."""
    tar_path = tmp_path / "model.tar.gz"
    inner_file = tmp_path / inner_name
    inner_file.write_bytes(b"dummy content")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(inner_file), arcname=inner_name)
    tar_path.write_bytes(buf.getvalue())
    return tar_path


def make_model_description(
    model: str = "test/model",
    hf: str | None = "org/repo",
    url: str | None = None,
    model_file: str = "model.onnx",
    additional_files: list[str] | None = None,
) -> BaseModelDescription:
    """Build a minimal BaseModelDescription for testing."""
    source = ModelSource(hf=hf, url=url, _deprecated_tar_struct=False)
    return BaseModelDescription(
        model=model,
        sources=source,
        model_file=model_file,
        description="Test model",
        license="MIT",
        size_in_GB=0.1,
        additional_files=additional_files or [],
    )


class ConcreteModelManagement(ModelManagement):
    """Concrete subclass for testing abstract base class methods."""

    _models: list[BaseModelDescription] = []

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return cls._models

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return [{"model": m.model} for m in cls._models]

    @classmethod
    def add_custom_model(cls, *args: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# TestAbstractMethods
# ---------------------------------------------------------------------------


class TestAbstractMethods:
    """Tests for methods that raise NotImplementedError in the base class."""

    def test_list_supported_models_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement.list_supported_models()

    def test_add_custom_model_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement.add_custom_model()

    def test_list_supported_models_private_raises(self):
        with pytest.raises(NotImplementedError):
            ModelManagement._list_supported_models()


# ---------------------------------------------------------------------------
# TestGetModelDescription
# ---------------------------------------------------------------------------


class TestGetModelDescription:
    """Tests for _get_model_description."""

    def setup_method(self):
        ConcreteModelManagement._models = [
            make_model_description(model="Alpha"),
            make_model_description(model="Beta"),
        ]

    def test_found_exact_case(self):
        result = ConcreteModelManagement._get_model_description("Alpha")
        assert result.model == "Alpha"

    def test_found_case_insensitive(self):
        result = ConcreteModelManagement._get_model_description("BETA")
        assert result.model == "Beta"

    def test_not_found_raises_value_error(self):
        with pytest.raises(ValueError, match="Gamma is not supported"):
            ConcreteModelManagement._get_model_description("Gamma")


# ---------------------------------------------------------------------------
# TestDownloadFileFromGcs
# ---------------------------------------------------------------------------


class TestDownloadFileFromGcs:
    """Tests for download_file_from_gcs."""

    GCS_URL = "https://storage.googleapis.com"

    def test_download_file_from_gcs_invalid_url(self, tmp_path):
        """Verify that an invalid URL or HTTP URL raises a ValueError."""
        with pytest.raises(ValueError, match="Only URLs from Google Cloud Storage are allowed"):
            ModelManagement.download_file_from_gcs(
                "http://storage.googleapis.com/test", str(tmp_path / "out")
            )

    def test_download_file_from_gcs_ssrf_bypass(self, tmp_path):
        """Verify that an SSRF bypass URL with auth syntax raises a ValueError."""
        with pytest.raises(ValueError, match="Only URLs from Google Cloud Storage are allowed"):
            ModelManagement.download_file_from_gcs(
                "https://storage.googleapis.com@127.0.0.1/test", str(tmp_path / "out")
            )

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_download_file_from_gcs_uses_large_chunk_size(self, mock_get_session, tmp_path):
        """Verify that iter_content is called with the optimized chunk_size."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        response = Mock()
        response.status_code = 200
        response.headers = {
            "content-length": "4",
            "x-goog-hash": "md5="
            + base64.b64encode(hashlib.md5(b"data", usedforsecurity=False).digest()).decode(),
        }
        response.iter_content.return_value = [b"data"]
        mock_session.get.return_value = response

        output = tmp_path / "test.file"
        ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/test.file", str(output))

        # Check if iter_content was called with chunk_size=1MB (1048576)
        response.iter_content.assert_called_with(chunk_size=1024 * 1024)

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_requests_get_uses_verify_true(self, mock_get_session, tmp_path):
        """requests.get MUST be called with verify=True to prevent accidental bypass."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "0"}
        response.iter_content.return_value = []
        mock_get.return_value = response

        output = tmp_path / "test.onnx"
        ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/test.onnx", str(output))

        # Assert that verify=True was passed in the call to requests.get
        args, kwargs = mock_get.call_args
        assert kwargs.get("verify") is True

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_download_file_from_gcs_404_raises_error(self, mock_get_session, tmp_path):
        """Non-403 HTTP errors should be caught by raise_for_status."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 404
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = response

        output = tmp_path / "missing.onnx"
        with pytest.raises(requests.exceptions.HTTPError, match="404 Client Error"):
            ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/missing.onnx", str(output))

    def test_invalid_scheme_raises_value_error(self, tmp_path):
        """Non-HTTP(S) schemes must be rejected."""
        output = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="Invalid URL"):
            ModelManagement.download_file_from_gcs("file:///etc/passwd", str(output))

    def test_ssrf_payloads_rejected(self, tmp_path):
        """SSRF payloads attempting to bypass host checks must be rejected."""
        output = tmp_path / "model.onnx"
        payloads = [
            "https://storage.googleapis.com@127.0.0.1/",
            "https://127.0.0.1#@storage.googleapis.com/",
            "https://storage.googleapis.com.evil.com/",
            "http://127.0.0.1:80@storage.googleapis.com/",
        ]
        for payload in payloads:
            with pytest.raises(ValueError, match="Invalid URL"):
                ModelManagement.download_file_from_gcs(payload, str(output))

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_ssrf_redirects_rejected(self, mock_get_session, tmp_path):
        """SSRF payloads attempting to bypass via open redirects must be rejected."""
        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 302
        mock_get.return_value = response

        output = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="SSRF Prevention: Redirects are not allowed."):
            ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/test.onnx", str(output))

        args, kwargs = mock_get.call_args
        assert kwargs.get("allow_redirects") is False

    def test_invalid_hostname_raises_value_error(self, tmp_path):
        """Non-GCS hostnames must be rejected."""
        output = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="Invalid URL"):
            ModelManagement.download_file_from_gcs(
                "http://169.254.169.254/latest/meta-data/", str(output)
            )
        with pytest.raises(ValueError, match="Invalid URL"):
            ModelManagement.download_file_from_gcs("https://example.com/x.onnx", str(output))

    def test_returns_existing_file(self, tmp_path):
        """If the file already exists, return immediately without HTTP request."""
        existing = tmp_path / "model.onnx"
        existing.write_bytes(b"data")
        result = ModelManagement.download_file_from_gcs(
            f"{self.GCS_URL}/model.onnx", str(existing)
        )
        assert result == str(existing)

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_403_raises_permission_error(self, mock_get_session, tmp_path):
        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 403
        mock_get.return_value = response

        output = tmp_path / "new_model.onnx"
        with pytest.raises(PermissionError, match="Authentication Error"):
            ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/x.onnx", str(output))

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_missing_content_length_logs_warning(self, mock_get_session, tmp_path):
        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "0"}
        response.iter_content.return_value = [b"hello"]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        with patch("qwen3_embed.common.model_management.logger.warning") as mock_warning:
            result = ModelManagement.download_file_from_gcs(
                f"{self.GCS_URL}/out.onnx", str(output), show_progress=False
            )
            assert result == str(output)
            mock_warning.assert_called_once()
            assert "Content-length header is missing" in mock_warning.call_args[0][0]

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_downloads_file_with_content_length(self, mock_get_session, tmp_path):
        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        chunk = b"A" * 1024
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": str(len(chunk))}
        response.iter_content.return_value = [chunk]
        mock_get.return_value = response

        output = tmp_path / "model.onnx"
        result = ModelManagement.download_file_from_gcs(
            f"{self.GCS_URL}/model.onnx", str(output), show_progress=True
        )
        assert result == str(output)
        assert output.read_bytes() == chunk

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_skips_keepalive_empty_chunks(self, mock_get_session, tmp_path):
        """Empty chunks must be filtered out (keep-alive frames)."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "5"}
        response.iter_content.return_value = [b"", b"hello", b""]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/out.onnx", str(output))
        assert output.read_bytes() == b"hello"

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_show_progress_false_when_no_content_length(self, mock_get_session, tmp_path):
        """Progress bar disabled when content-length is missing."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.iter_content.return_value = [b"data"]
        mock_get.return_value = response

        output = tmp_path / "out.onnx"
        # Should not raise even with show_progress=True but no content-length
        ModelManagement.download_file_from_gcs(
            f"{self.GCS_URL}/out.onnx", str(output), show_progress=True
        )
        assert output.exists()

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_hash_mismatch_raises_value_error(self, mock_get_session, tmp_path):
        """MD5 mismatch between header and downloaded content raises ValueError."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        chunk = b"actual content"
        wrong_md5 = base64.b64encode(
            hashlib.md5(b"different content", usedforsecurity=False).digest()
        ).decode()  # SECURITY: MD5 is used solely for non-cryptographic file integrity checking.

        response = Mock()
        response.status_code = 200
        response.headers = {
            "content-length": str(len(chunk)),
            "x-goog-hash": f"md5={wrong_md5}",
        }
        response.iter_content.return_value = [chunk]
        mock_get.return_value = response

        output = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="File integrity check failed"):
            ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/model.onnx", str(output))
        assert not output.exists()

    @patch("qwen3_embed.common.model_management.ModelManagement._get_session")
    def test_hash_match_succeeds(self, mock_get_session, tmp_path):
        """Matching MD5 in x-goog-hash header allows download to complete."""

        mock_session = MagicMock()
        mock_get = mock_session.get
        mock_get_session.return_value = mock_session
        chunk = b"verified content"
        correct_md5 = base64.b64encode(
            hashlib.md5(chunk, usedforsecurity=False).digest()
        ).decode()  # SECURITY: MD5 is used solely for non-cryptographic file integrity checking.

        response = Mock()
        response.status_code = 200
        response.headers = {
            "content-length": str(len(chunk)),
            "x-goog-hash": f"crc32c=abc123, md5={correct_md5}",
        }
        response.iter_content.return_value = [chunk]
        mock_get.return_value = response

        output = tmp_path / "model.onnx"
        result = ModelManagement.download_file_from_gcs(f"{self.GCS_URL}/model.onnx", str(output))
        assert result == str(output)
        assert output.read_bytes() == chunk


# ---------------------------------------------------------------------------
# TestDecompressToCache
# ---------------------------------------------------------------------------


class TestDecompressToCache:
    """Tests for decompress_to_cache method."""

    def test_decompress_nonexistent_file(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.tar.gz"
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(nonexistent_file), str(tmp_path))

    def test_decompress_invalid_extension(self, tmp_path):
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.touch()
        with pytest.raises(ValueError, match="is not a .tar.gz file"):
            ModelManagement.decompress_to_cache(str(invalid_file), str(tmp_path))

    def test_decompress_directory(self, tmp_path):
        directory = tmp_path / "directory.tar.gz"
        directory.mkdir()
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(directory), str(tmp_path))

    def test_decompress_corrupted_tar_gz(self, tmp_path):
        corrupted_file = tmp_path / "corrupted.tar.gz"
        corrupted_file.write_text("not a tar file")

        cache_dir = tmp_path / "cache_tmp"
        cache_dir.mkdir()

        with pytest.raises(tarfile.TarError, match="not a gzip file"):
            ModelManagement.decompress_to_cache(str(corrupted_file), str(cache_dir))

    def test_decompress_success(self, tmp_path):
        """Valid .tar.gz extracts successfully and returns cache_dir."""
        tar_path = make_tar_gz(tmp_path, inner_name="model.onnx")
        cache_dir = tmp_path / "out"
        cache_dir.mkdir()

        result = ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))
        assert result == str(cache_dir)
        assert (cache_dir / "model.onnx").exists()

    def test_decompress_extraction_failure_removes_cache_dir(self, tmp_path):
        """If tarfile extraction fails mid-way, the cache directory is completely removed."""
        cache_dir = tmp_path / "cache_dir"
        cache_dir.mkdir()

        # Create a valid tar structure but mock extractall to fail
        tar_path = make_tar_gz(tmp_path, inner_name="model.onnx")

        def mock_extractall(*args, **kwargs):
            raise tarfile.TarError("Simulated extraction failure")

        with (
            patch("tarfile.TarFile.extractall", side_effect=mock_extractall),
            pytest.raises(tarfile.TarError, match="Simulated extraction failure"),
        ):
            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        # Ensure the cache dir was removed due to the error
        assert not cache_dir.exists()

    def test_decompress_tar_slip_prevention(self, tmp_path):
        """Tar slip via malicious path raises TarError and cache dir is removed."""
        cache_dir = tmp_path / "tmp_cache_dir"
        cache_dir.mkdir()

        malicious_tar = tmp_path / "malicious.tar.gz"
        with tarfile.open(malicious_tar, "w:gz") as tar:
            with open("test.txt", "w") as f:
                f.write("test")
            info = tarfile.TarInfo(name="../evil.txt")
            info.size = 4
            with open("test.txt", "rb") as fileobj:
                tar.addfile(info, fileobj=fileobj)
            import os

            os.remove("test.txt")

        with pytest.raises(tarfile.TarError):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_absolute_path_traversal_prevention(self, tmp_path):
        """Tar slip via absolute path raises TarError and cache dir is removed."""
        cache_dir = tmp_path / "tmp_cache_dir_abs"
        cache_dir.mkdir()

        malicious_tar = tmp_path / "malicious_abs.tar.gz"
        with tarfile.open(malicious_tar, "w:gz") as tar:
            data = b"test"
            info = tarfile.TarInfo(name="/etc/passwd")
            info.size = len(data)
            fileobj = io.BytesIO(data)
            tar.addfile(info, fileobj=fileobj)

        with pytest.raises(tarfile.TarError):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_symlink_slip_prevention(self, tmp_path):
        """Tar slip via symlink traversal raises TarError and cache dir is removed."""
        cache_dir = tmp_path / "tmp_cache_dir_symlink"
        cache_dir.mkdir()

        malicious_tar = tmp_path / "malicious_symlink.tar.gz"
        with tarfile.open(malicious_tar, "w:gz") as tar:
            info = tarfile.TarInfo(name="evil_symlink")
            info.type = tarfile.SYMTYPE
            info.linkname = "../evil.txt"
            tar.addfile(info)

        with pytest.raises(tarfile.TarError):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_symlink_absolute_prevention(self, tmp_path):
        """Tar slip via absolute symlink raises TarError and cache dir is removed."""
        cache_dir = tmp_path / "tmp_cache_dir_symlink_abs"
        cache_dir.mkdir()

        malicious_tar = tmp_path / "malicious_symlink_abs.tar.gz"
        with tarfile.open(malicious_tar, "w:gz") as tar:
            info = tarfile.TarInfo(name="evil_symlink_abs")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        with pytest.raises(tarfile.TarError):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_hardlink_traversal_prevention(self, tmp_path):
        """Tar slip via hardlink resolved relative to extraction root is caught.

        Hardlinks in tar archives resolve linkname relative to the archive root
        (extraction directory), NOT relative to the containing directory like
        symlinks. A nested hardlink with '../evil.txt' linkname must be detected
        as escaping the target directory.
        """
        cache_dir = tmp_path / "tmp_cache_dir_hardlink_nested"
        cache_dir.mkdir()

        malicious_tar = tmp_path / "malicious_hardlink_nested.tar.gz"
        with tarfile.open(malicious_tar, "w:gz") as tar:
            info = tarfile.TarInfo(name="sub/nested/evil_hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "../evil.txt"
            tar.addfile(info)

        with pytest.raises(tarfile.TarError):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_tar_bomb_prevention(self, tmp_path):
        """Tar bomb prevention: extraction raises TarError if uncompressed size > 20GB."""
        cache_dir = tmp_path / "tmp_cache_dir_bomb"
        cache_dir.mkdir()
        fake_tar_gz = tmp_path / "fake.tar.gz"
        fake_tar_gz.write_text("fake tar")

        mock_member = MagicMock()
        mock_member.name = "huge_file.txt"
        mock_member.issym.return_value = False
        mock_member.islnk.return_value = False
        mock_member.size = 20 * 1024 * 1024 * 1024 + 1  # 20 GB + 1 byte

        with patch("tarfile.open") as mock_tar_open:
            mock_tar = MagicMock()
            mock_tar.__iter__.return_value = iter([mock_member])
            mock_tar.extractall.side_effect = lambda path, members, filter=None: list(members)
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            with pytest.raises(tarfile.TarError, match="Decompression bomb detected"):
                ModelManagement.decompress_to_cache(str(fake_tar_gz), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_unsupported_file_type(self, tmp_path):
        """Unsupported file type (e.g. FIFO) raises TarError and cache dir is removed."""
        cache_dir = tmp_path / "tmp_cache_dir_unsupported"
        cache_dir.mkdir()

        unsupported_tar = tmp_path / "unsupported.tar.gz"
        with tarfile.open(unsupported_tar, "w:gz") as tar:
            info = tarfile.TarInfo(name="fifo")
            info.type = tarfile.FIFOTYPE
            tar.addfile(info)

        with pytest.raises(tarfile.TarError, match="Unsupported file type in tar file"):
            ModelManagement.decompress_to_cache(str(unsupported_tar), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_mid_extraction_failure(self, tmp_path):
        """Mid-extraction TarError is re-raised and cache dir is removed."""
        tar_path = make_tar_gz(tmp_path, inner_name="model.onnx")
        cache_dir = tmp_path / "tmp_out"
        cache_dir.mkdir()

        def fake_extractall(*args, **kwargs):
            raise tarfile.TarError("Mid-extraction error")

        with (
            patch("tarfile.TarFile.extractall", side_effect=fake_extractall),
            pytest.raises(tarfile.TarError, match="Mid-extraction error"),
        ):
            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_tar_bomb_cumulative(self, tmp_path):
        """Tar bomb prevention: multiple small files exceeding 20GB limit."""
        cache_dir = tmp_path / "tmp_cache_dir_bomb_cumulative"
        cache_dir.mkdir()
        fake_tar_gz = tmp_path / "fake.tar.gz"
        fake_tar_gz.write_text("fake tar")

        m1 = MagicMock(spec=tarfile.TarInfo)
        m1.name = "f1.txt"
        m1.size = 15 * 1024 * 1024 * 1024
        m1.issym.return_value = False
        m1.islnk.return_value = False
        m1.isdir.return_value = False
        m1.isreg.return_value = True

        m2 = MagicMock(spec=tarfile.TarInfo)
        m2.name = "f2.txt"
        m2.size = 6 * 1024 * 1024 * 1024
        m2.issym.return_value = False
        m2.islnk.return_value = False
        m2.isdir.return_value = False
        m2.isreg.return_value = True

        with patch("tarfile.open") as mock_tar_open:
            mock_tar = MagicMock()
            mock_tar.__iter__.return_value = iter([m1, m2])
            # Simulate extractall consuming the members generator
            mock_tar.extractall.side_effect = lambda path, members, filter=None: list(members)
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            with pytest.raises(tarfile.TarError, match="Decompression bomb detected"):
                ModelManagement.decompress_to_cache(str(fake_tar_gz), str(cache_dir))

        assert not cache_dir.exists()

    def test_decompress_tar_bomb_fallback_path(self, tmp_path):
        """Tar bomb prevention in fallback path (no data_filter)."""
        cache_dir = tmp_path / "tmp_cache_dir_bomb_fallback"
        cache_dir.mkdir()
        fake_tar_gz = tmp_path / "fake.tar.gz"
        fake_tar_gz.write_text("fake tar")

        mock_member = MagicMock(spec=tarfile.TarInfo)
        mock_member.name = "huge_file.txt"
        mock_member.size = 21 * 1024 * 1024 * 1024
        mock_member.issym.return_value = False
        mock_member.islnk.return_value = False
        mock_member.isdir.return_value = False
        mock_member.isreg.return_value = True

        with patch("qwen3_embed.common.model_management.tarfile") as mock_tarfile_mod:
            # Mock tarfile.TarError since it is used in "except" and "raise"
            mock_tarfile_mod.TarError = tarfile.TarError
            # Mock tarfile.open to return a mock_tar
            mock_tar = MagicMock()
            mock_tar.__iter__.return_value = iter([mock_member])
            mock_tarfile_mod.open.return_value.__enter__.return_value = mock_tar

            # Ensure it DOES NOT have data_filter
            if hasattr(mock_tarfile_mod, "data_filter"):
                del mock_tarfile_mod.data_filter

            with pytest.raises(tarfile.TarError, match="Decompression bomb detected"):
                ModelManagement.decompress_to_cache(str(fake_tar_gz), str(cache_dir))

        assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# TestDownloadFilesFromHuggingFace
# ---------------------------------------------------------------------------


class TestDownloadFilesFromHuggingFace:
    """Tests for download_files_from_huggingface."""

    def _make_repo_files(self) -> list[RepoFile]:
        return [
            make_repo_file("model.onnx", size=500, oid="aaa"),
            make_repo_file("sub/config.json", size=100, oid="bbb"),
        ]

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_no_metadata(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with no metadata file just calls snapshot_download."""
        mock_snap.return_value = str(tmp_path / "result")

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_disable.assert_called_once()
        mock_snap.assert_called_once()
        assert result == str(tmp_path / "result")

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_with_valid_metadata(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with valid metadata: snapshot_download is called."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        metadata = {
            "model.onnx": {"size": 500, "blob_id": "aaa"},
        }
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_snap.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_with_invalid_metadata_warns(self, mock_snap, mock_disable, tmp_path):
        """local_files_only=True with mismatched file sizes logs a warning but continues."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 200)  # size mismatch

        metadata = {"model.onnx": {"size": 500, "blob_id": "aaa"}}
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        mock_snap.return_value = str(snapshot_dir)

        # Should NOT raise, just warn
        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        mock_snap.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_passes_cached_revision(self, mock_snap, mock_disable, tmp_path):
        """local_files_only resolves the cached SHA and pins ``revision``.

        The online path pins an explicit commit SHA, so HF never writes a
        ``refs/<branch>`` pointer. Omitting ``revision`` offline would default
        to ``"main"`` and raise ``LocalEntryNotFoundError`` despite a complete
        cache; the cached SHA must be passed through instead.
        """
        snapshot_dir = tmp_path / "models--org--repo"
        (snapshot_dir / "snapshots" / "deadbeefsha").mkdir(parents=True)
        mock_snap.return_value = str(snapshot_dir)

        ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )
        assert mock_snap.call_args.kwargs["revision"] == "deadbeefsha"

    @patch("qwen3_embed.common.model_management.disable_progress_bars")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_local_files_only_explicit_revision_not_overwritten(
        self, mock_snap, mock_disable, tmp_path
    ):
        """An explicit ``revision`` kwarg takes precedence over the cached SHA."""
        snapshot_dir = tmp_path / "models--org--repo"
        (snapshot_dir / "snapshots" / "deadbeefsha").mkdir(parents=True)
        mock_snap.return_value = str(snapshot_dir)

        ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
            local_files_only=True,
            revision="explicit-rev",
        )
        assert mock_snap.call_args.kwargs["revision"] == "explicit-rev"

    def test_resolve_cached_revision_returns_newest(self, tmp_path):
        """Picks the most recently modified snapshot directory."""
        snapshots = tmp_path / "models--org--repo" / "snapshots"
        (snapshots / "old").mkdir(parents=True)
        (snapshots / "new").mkdir(parents=True)
        os.utime(snapshots / "old", (1000, 1000))
        os.utime(snapshots / "new", (2000, 2000))
        assert ModelManagement._resolve_cached_revision(tmp_path / "models--org--repo") == "new"

    def test_resolve_cached_revision_none_when_absent(self, tmp_path):
        """Returns None when there is no snapshots directory or it is empty."""
        assert ModelManagement._resolve_cached_revision(tmp_path / "missing") is None
        empty = tmp_path / "models--org--repo"
        (empty / "snapshots").mkdir(parents=True)
        assert ModelManagement._resolve_cached_revision(empty) is None

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_no_cached_metadata(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode: no prior metadata -> downloads, collects metadata, verifies."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.onnx").write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)
        meta_file = snapshot_dir / ModelManagement.METADATA_FILE
        assert meta_file.exists()

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_cached_metadata_verified(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode with valid cached metadata: disables progress bars."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        metadata = {"model.onnx": {"size": 500, "blob_id": "aaa"}}
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))

        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        with patch("qwen3_embed.common.model_management.disable_progress_bars") as mock_dis:
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        mock_dis.assert_called_once()
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_empty_repo_tree(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Online mode with empty repo tree: treats repo_files as empty list."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)

        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = []
        mock_snap.return_value = str(snapshot_dir)

        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_online_download_failure_raises_value_error(
        self, mock_snap, mock_info, mock_tree, tmp_path
    ):
        """If post-download offline verification fails (size mismatch), raises ValueError."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        # File exists but with wrong size -> offline verify will fail
        (snapshot_dir / "model.onnx").write_bytes(b"x" * 100)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        with pytest.raises(ValueError, match="Files have been corrupted"):
            ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_metadata_save_mkdir_oserror_swallowed(
        self, mock_snap, mock_info, mock_tree, tmp_path
    ):
        """OSError while creating metadata directory is logged but does not raise."""
        snapshot_dir = tmp_path / "models--org--repo"
        # We DO NOT create snapshot_dir so that .exists() is False

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # Patch Path.mkdir to raise OSError
        original_mkdir = Path.mkdir

        def patched_mkdir(self, *args, **kwargs):
            if "models--org--repo" in str(self):
                raise OSError("permission denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", patched_mkdir):
            # Should not raise
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_metadata_save_mkdir_valueerror_swallowed(
        self, mock_snap, mock_info, mock_tree, tmp_path
    ):
        """ValueError while creating metadata directory is logged but does not raise."""
        snapshot_dir = tmp_path / "models--org--repo"
        # We DO NOT create snapshot_dir so that .exists() is False

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # Patch Path.mkdir to raise ValueError
        original_mkdir = Path.mkdir

        def patched_mkdir(self, *args, **kwargs):
            if "models--org--repo" in str(self):
                raise ValueError("invalid path")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", patched_mkdir):
            # Should not raise
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_metadata_save_oserror_is_swallowed(self, mock_snap, mock_info, mock_tree, tmp_path):
        """OSError while saving metadata is logged but does not raise."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # Patch Path.write_text to raise OSError on metadata write
        original_write_text = Path.write_text

        def patched_write_text(self, data, *args, **kwargs):
            if self.name == ModelManagement.METADATA_FILE:
                raise OSError("disk full")
            return original_write_text(self, data, *args, **kwargs)

        with patch.object(Path, "write_text", patched_write_text):
            # Should not raise
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        assert result == str(snapshot_dir)

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_verify_files_keyerror_returns_false(self, mock_snap, mock_info, mock_tree, tmp_path):
        """KeyError in metadata causes _verify_files_from_metadata to return False."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)

        # metadata with broken entry (missing 'size' key) -> KeyError in verify
        metadata = {"model.onnx": {"blob_id": "aaa"}}  # 'size' key missing
        (snapshot_dir / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # With broken metadata, verified_metadata=False -> re-collects + re-verifies
        # Since the actual file matches (size 500), final offline verify passes
        result = ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )
        assert result == str(snapshot_dir)

    # -----------------------------------------------------------------------

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_metadata_save_valueerror_is_swallowed(
        self, mock_snap, mock_info, mock_tree, tmp_path
    ):
        """ValueError while saving metadata logs an exception and warning but does not raise."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)
        inner_file = snapshot_dir / "model.onnx"
        inner_file.write_bytes(b"x" * 500)

        repo_files = [make_repo_file("model.onnx", size=500, oid="aaa")]
        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        # Patch json.dumps to raise ValueError on metadata write
        def patched_dumps(data, *args, **kwargs):
            raise ValueError("metadata serialization error")

        with (
            patch("qwen3_embed.common.model_management.json.dumps", patched_dumps),
            patch("qwen3_embed.common.model_management.logger.exception") as mock_exception,
            patch("qwen3_embed.common.model_management.logger.warning") as mock_warning,
        ):
            # Should not raise
            result = ModelManagement.download_files_from_huggingface(
                hf_source_repo="org/repo",
                cache_dir=str(tmp_path),
                extra_patterns=["model.onnx"],
            )
        assert result == str(snapshot_dir)
        mock_exception.assert_called_once()
        # Verify ValueError is passed to exception
        args, _ = mock_exception.call_args
        assert isinstance(args[0], ValueError)
        assert str(args[0]) == "metadata serialization error"

        mock_warning.assert_called_once_with(
            "Failed to save metadata file. Next load may take longer to verify."
        )

    # ---------------------------------------------------------------------------
    # TestRetrieveModelGcs
    # ---------------------------------------------------------------------------

    @patch("qwen3_embed.common.model_management.list_repo_tree")
    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.snapshot_download")
    def test_collect_file_metadata_logic(self, mock_snap, mock_info, mock_tree, tmp_path):
        """Tests the internal _collect_file_metadata logic by verifying the saved metadata.json."""
        snapshot_dir = tmp_path / "models--org--repo"
        snapshot_dir.mkdir(parents=True)

        # Create some files:
        # 1. A normal file in the root
        (snapshot_dir / "model.onnx").write_bytes(b"x" * 500)

        # 2. A file in a subdirectory
        sub_dir = snapshot_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "config.json").write_bytes(b"x" * 100)

        # 3. A file that is not in repo_files
        (snapshot_dir / "extra.txt").write_bytes(b"x" * 50)

        # 4. The metadata file itself (should be ignored)
        # removed to force re-collection

        repo_files = [
            make_repo_file("model.onnx", size=500, oid="aaa"),
            make_repo_file("sub/config.json", size=100, oid="bbb"),
        ]

        mock_info.return_value = Mock(sha="rev123")
        mock_tree.return_value = repo_files
        mock_snap.return_value = str(snapshot_dir)

        ModelManagement.download_files_from_huggingface(
            hf_source_repo="org/repo",
            cache_dir=str(tmp_path),
            extra_patterns=["model.onnx"],
        )

        meta_file = snapshot_dir / ModelManagement.METADATA_FILE
        assert meta_file.exists()

        metadata = json.loads(meta_file.read_text())

        # Verify metadata dictionary contents
        assert "model.onnx" in metadata
        assert metadata["model.onnx"] == {"size": 500, "blob_id": "aaa"}

        sub_key = "sub/config.json"
        assert sub_key in metadata
        assert metadata[sub_key] == {"size": 100, "blob_id": "bbb"}

        assert "extra.txt" not in metadata
        assert ModelManagement.METADATA_FILE not in metadata


# # ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestDownloadFromHF
# ---------------------------------------------------------------------------


class TestDownloadFromHF:
    """Tests for _download_from_hf method."""

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_download_success(self, mock_hf, mock_enable, tmp_path):
        """Verify successful download returns Path and enables progress bars."""
        expected_path = tmp_path / "model_dir"
        mock_hf.return_value = str(expected_path)

        model = make_model_description(hf="org/repo")
        result = ModelManagement._download_from_hf(model=model, cache_dir=str(tmp_path))

        assert result == expected_path
        mock_enable.assert_called_once()

    @pytest.mark.parametrize("exception_cls", [OSError, RepositoryNotFoundError, ValueError])
    @patch("qwen3_embed.common.model_management.logger.error")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_download_exceptions_logging(
        self, mock_hf, mock_enable, mock_logger, exception_cls, tmp_path
    ):
        """Verify exceptions are caught, logged, and return None when local_files_only is False."""
        if exception_cls == RepositoryNotFoundError:
            mock_hf.side_effect = RepositoryNotFoundError("Not found", response=MagicMock())
        else:
            mock_hf.side_effect = exception_cls("Error message")

        model = make_model_description(hf="org/repo")
        result = ModelManagement._download_from_hf(
            model=model,
            cache_dir=str(tmp_path),
            local_files_only=False,
        )

        assert result is None
        mock_logger.assert_called_once()
        assert "Could not download model from HuggingFace" in mock_logger.call_args[0][0]
        mock_enable.assert_called_once()

    @patch("qwen3_embed.common.model_management.logger.error")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_download_exception_no_logging_local_files_only(
        self, mock_hf, mock_enable, mock_logger, tmp_path
    ):
        """Verify exceptions return None without logging when local_files_only is True."""
        mock_hf.side_effect = OSError("Error message")

        model = make_model_description(hf="org/repo")
        result = ModelManagement._download_from_hf(
            model=model,
            cache_dir=str(tmp_path),
            local_files_only=True,
        )

        assert result is None
        mock_logger.assert_not_called()
        mock_enable.assert_called_once()


class TestRetrieveModelGcs:
    """Tests for retrieve_model_gcs."""

    def test_returns_existing_populated_model_dir(self, tmp_path):
        """If model_dir already exists and has files, return it without downloading."""
        model_dir = tmp_path / "name"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"data")

        with patch.object(ModelManagement, "download_file_from_gcs") as mock_dl:
            result = ModelManagement.retrieve_model_gcs(
                model_name="model/name",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
            )
        mock_dl.assert_not_called()
        assert result == model_dir

    def test_local_files_only_raises_value_error(self, tmp_path):
        """local_files_only=True when model dir is absent raises ValueError."""
        with pytest.raises(ValueError, match="local_files_only=True"):
            ModelManagement.retrieve_model_gcs(
                model_name="test/model",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
                local_files_only=True,
            )

    def test_removes_stale_tmp_dir_before_download(self, tmp_path):
        """Pre-existing model_tmp_dir is removed before download attempt."""
        model_tmp_dir = tmp_path / "tmp" / "model"
        model_tmp_dir.mkdir(parents=True)
        (model_tmp_dir / "stale.onnx").write_bytes(b"stale")

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            # Simulate extraction creating model_tmp_dir again
            out = Path(cache_dir) / "model"
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"fresh")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name="model",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(tmp_path),
            )
        assert result == tmp_path / "model"

    def test_deletes_tar_gz_after_extraction(self, tmp_path):
        """The downloaded .tar.gz is removed after extraction."""
        model_name = "mymodel"

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            out = Path(cache_dir) / model_name
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"data")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name=model_name,
                source_url="http://example.com/mymodel.tar.gz",
                cache_dir=str(tmp_path),
            )
        tar_gz = tmp_path / f"{model_name}.tar.gz"
        assert not tar_gz.exists()
        assert result == tmp_path / model_name

    def test_raises_if_tmp_dir_missing_after_extraction(self, tmp_path):
        """If model_tmp_dir is absent after decompress, raises ValueError."""
        with (
            patch.object(ModelManagement, "download_file_from_gcs"),
            patch.object(
                ModelManagement, "decompress_to_cache", return_value=str(tmp_path / "tmp")
            ),
            pytest.raises(ValueError, match="Could not find"),
        ):
            ModelManagement.retrieve_model_gcs(
                model_name="missing",
                source_url="http://example.com/missing.tar.gz",
                cache_dir=str(tmp_path),
            )

    def test_deprecated_tar_struct_prefixes_fast(self, tmp_path):
        """deprecated_tar_struct=True prepends 'fast-' to the model dir name."""
        model_name = "org/mymodel"

        def fake_download(url: str, output_path: str, **kwargs) -> str:
            Path(output_path).write_bytes(b"fake tar")
            return output_path

        def fake_decompress(targz_path: str, cache_dir: str) -> str:
            out = Path(cache_dir) / "fast-mymodel"
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.onnx").write_bytes(b"data")
            return cache_dir

        with (
            patch.object(ModelManagement, "download_file_from_gcs", side_effect=fake_download),
            patch.object(ModelManagement, "decompress_to_cache", side_effect=fake_decompress),
        ):
            result = ModelManagement.retrieve_model_gcs(
                model_name=model_name,
                source_url="http://example.com/fast-mymodel.tar.gz",
                cache_dir=str(tmp_path),
                deprecated_tar_struct=True,
            )
        assert result.name == "fast-mymodel"


# ---------------------------------------------------------------------------
# TestDownloadFromGcs
# ---------------------------------------------------------------------------


class TestDownloadFromGcs:
    """Tests for _download_from_gcs method."""

    @patch("qwen3_embed.common.model_management.logger.error")
    def test_download_from_gcs_returns_none_on_exception(self, mock_logger, tmp_path):
        """If retrieve_model_gcs raises an exception, return None and log error."""
        with patch.object(
            ModelManagement, "retrieve_model_gcs", side_effect=ValueError("GCS Error")
        ):
            model = make_model_description(url="http://example.com/model.tar.gz")
            result = ModelManagement._download_from_gcs(
                model=model,
                cache_dir=str(tmp_path),
                local_files_only=False,
            )

        assert result is None
        mock_logger.assert_called_once_with(
            "Could not download model from url: http://example.com/model.tar.gz"
        )

    @patch("qwen3_embed.common.model_management.logger.error")
    def test_download_from_gcs_no_logger_on_local_files_only(self, mock_logger, tmp_path):
        """If local_files_only is True, do not log error on exception."""
        with patch.object(
            ModelManagement, "retrieve_model_gcs", side_effect=ValueError("GCS Error")
        ):
            model = make_model_description(url="http://example.com/model.tar.gz")
            result = ModelManagement._download_from_gcs(
                model=model,
                cache_dir=str(tmp_path),
                local_files_only=True,
            )

        assert result is None
        mock_logger.assert_not_called()


# ---------------------------------------------------------------------------
# TestDownloadModel
# ---------------------------------------------------------------------------


class TestDownloadModel:
    """Tests for download_model."""

    def test_specific_model_path_returns_immediately(self, tmp_path):
        """specific_model_path kwarg bypasses all download logic."""
        model = make_model_description()
        result = ModelManagement.download_model(
            model, cache_dir=str(tmp_path), specific_model_path="/custom/path"
        )
        assert result == Path("/custom/path")

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_source_cache_hit_returns_path(self, mock_enable, tmp_path):
        """HF local cache hit returns immediately without retry loop."""
        model = make_model_description(hf="org/repo")
        cached_dir = tmp_path / "cached"
        cached_dir.mkdir()
        (cached_dir / "model.onnx").touch()

        with patch.object(
            ModelManagement,
            "download_files_from_huggingface",
            return_value=str(cached_dir),
        ) as mock_hf:
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path))

        mock_hf.assert_called_once()
        assert result == Path(cached_dir)
        mock_enable.assert_called()

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_source_cache_miss_falls_through_to_retry(self, mock_enable, tmp_path):
        """HF local cache miss leads to online retry, then success."""
        model = make_model_description(hf="org/repo")
        online_path = str(tmp_path / "online")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("local_files_only"):
                raise OSError("not in cache")
            return online_path

        with patch.object(
            ModelManagement, "download_files_from_huggingface", side_effect=side_effect
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        assert result == Path(online_path)
        assert call_count == 2

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_hf_online_fails_falls_back_to_url(self, mock_enable, tmp_path):
        """HF online failure falls back to GCS/URL source."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        def hf_side_effect(*args, **kwargs):
            raise OSError("network error")

        with (
            patch.object(
                ModelManagement, "download_files_from_huggingface", side_effect=hf_side_effect
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", return_value=gcs_path) as mock_gcs,
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_url_only_source_uses_gcs(self, mock_enable, tmp_path):
        """Model with only url source uses retrieve_model_gcs directly."""
        model = make_model_description(hf=None, url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        with patch.object(
            ModelManagement, "retrieve_model_gcs", return_value=gcs_path
        ) as mock_gcs:
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.time.sleep")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_all_sources_fail_raises_value_error(self, mock_enable, mock_sleep, tmp_path):
        """All retry attempts exhausted -> ValueError."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=OSError("network error"),
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", side_effect=OSError("gcs error")),
            pytest.raises(ValueError, match="Could not load model"),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=2)

    @patch("qwen3_embed.common.model_management.time.sleep")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_sleep_called_between_retries(self, mock_enable, mock_sleep, tmp_path):
        """time.sleep is called between failed retry attempts."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=OSError("fail"),
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", side_effect=OSError("fail")),
            pytest.raises(ValueError),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=2)

        assert mock_sleep.call_count >= 1

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_local_files_only_breaks_on_failure(self, mock_enable, tmp_path):
        """local_files_only=True does not retry and raises ValueError."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")

        with (
            patch.object(
                ModelManagement,
                "download_files_from_huggingface",
                side_effect=OSError("not cached"),
            ),
            patch.object(
                ModelManagement,
                "retrieve_model_gcs",
                side_effect=OSError("not cached"),
            ),
            pytest.raises(ValueError, match="Could not load model"),
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path), local_files_only=True)

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_repository_not_found_error_handled(self, mock_enable, tmp_path):
        """RepositoryNotFoundError in HF online -> falls back to GCS."""
        model = make_model_description(hf="org/repo", url="http://example.com/model.tar.gz")
        gcs_path = tmp_path / "gcs_model"

        def hf_side_effect(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            raise RepositoryNotFoundError("not found", response=MagicMock())

        with (
            patch.object(
                ModelManagement, "download_files_from_huggingface", side_effect=hf_side_effect
            ),
            patch.object(ModelManagement, "retrieve_model_gcs", return_value=gcs_path) as mock_gcs,
        ):
            result = ModelManagement.download_model(model, cache_dir=str(tmp_path), retries=1)

        mock_gcs.assert_called_once()
        assert result == gcs_path

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    def test_extra_patterns_built_from_model_file_and_additional_files(
        self, mock_enable, tmp_path
    ):
        """extra_patterns includes model_file and additional_files."""
        model = make_model_description(
            hf="org/repo", model_file="weights.onnx", additional_files=["vocab.txt"]
        )
        captured_patterns: list[list[str]] = []

        def capture_hf(*args, **kwargs):
            captured_patterns.append(kwargs.get("extra_patterns", []))
            return str(tmp_path / "cached")

        with patch.object(
            ModelManagement, "download_files_from_huggingface", side_effect=capture_hf
        ):
            ModelManagement.download_model(model, cache_dir=str(tmp_path))

        patterns = captured_patterns[0]
        assert "weights.onnx" in patterns
        assert "vocab.txt" in patterns


# ---------------------------------------------------------------------------
# TestCheckHFCache
# ---------------------------------------------------------------------------


class TestCheckHFCache:
    """Tests for _check_hf_cache method."""

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_returns_path_if_model_exists(self, mock_download, mock_enable, tmp_path):
        """If the model file exists in the downloaded snapshot, return the path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        # Create dummy model file
        model_file = "model.onnx"
        (snapshot_dir / model_file).write_bytes(b"data")

        mock_download.return_value = str(snapshot_dir)

        model = make_model_description(hf="org/repo")
        result = ModelManagement._check_hf_cache(
            model=model,
            cache_dir=str(cache_dir),
        )

        assert result == snapshot_dir
        mock_enable.assert_called_once()
        mock_download.assert_called_once()
        # Verify local_files_only was passed as True
        assert mock_download.call_args[1].get("local_files_only") is True

    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_returns_none_if_model_missing(self, mock_download, mock_enable, tmp_path):
        """If the snapshot directory is downloaded but missing the required file, return None."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        # We purposely do not create the model file

        mock_download.return_value = str(snapshot_dir)

        model = make_model_description(hf="org/repo")
        result = ModelManagement._check_hf_cache(
            model=model,
            cache_dir=str(cache_dir),
        )

        assert result is None
        mock_enable.assert_called_once()

    @patch("qwen3_embed.common.model_management.logger.debug")
    @patch("qwen3_embed.common.model_management.enable_progress_bars")
    @patch.object(ModelManagement, "download_files_from_huggingface")
    def test_returns_none_on_exception(self, mock_download, mock_enable, mock_logger, tmp_path):
        """If an OSError is raised during the cache check (e.g. not found), it is caught and None is returned."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        mock_download.side_effect = OSError("Not found in cache")

        model = make_model_description(hf="org/repo")
        result = ModelManagement._check_hf_cache(
            model=model,
            cache_dir=str(cache_dir),
        )

        assert result is None
        mock_enable.assert_called_once()
        mock_logger.assert_called_once_with("Model not found in cache, will attempt download")


# ---------------------------------------------------------------------------
# TestSaveFileMetadata
# ---------------------------------------------------------------------------


class TestSaveFileMetadata:
    """Tests for _save_file_metadata method."""

    @patch("qwen3_embed.common.model_management.logger")
    def test_save_file_metadata_handles_oserror(self, mock_logger, tmp_path):
        """Verify that OSError during file write is caught and logged."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        meta = {"file.txt": {"size": 100, "blob_id": "abc"}}

        # Mock write_text to raise OSError
        with patch("pathlib.Path.write_text", side_effect=OSError("Disk full")):
            ModelManagement._save_file_metadata(model_dir, meta)

        mock_logger.exception.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "Failed to save metadata file. Next load may take longer to verify."
        )

    @patch("qwen3_embed.common.model_management.logger")
    def test_save_file_metadata_handles_valueerror(self, mock_logger, tmp_path):
        """Verify that ValueError during json.dumps is caught and logged."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        meta = {"file.txt": {"size": 100, "blob_id": "abc"}}

        # Mock json.dumps to raise ValueError
        with patch(
            "qwen3_embed.common.model_management.json.dumps",
            side_effect=ValueError("Invalid JSON"),
        ):
            ModelManagement._save_file_metadata(model_dir, meta)

        mock_logger.exception.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "Failed to save metadata file. Next load may take longer to verify."
        )

    @patch("qwen3_embed.common.model_management.logger")
    def test_save_file_metadata_handles_mkdir_oserror(self, mock_logger, tmp_path):
        """Verify that OSError during directory creation is caught and logged."""
        model_dir = tmp_path / "model_no_exist"
        meta = {"file.txt": {"size": 100, "blob_id": "abc"}}

        # Mock mkdir to raise OSError
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            ModelManagement._save_file_metadata(model_dir, meta)

        mock_logger.exception.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "Failed to save metadata file. Next load may take longer to verify."
        )


class TestVerifyFilesFromMetadata:
    """Tests for direct verification of the _verify_files_from_metadata method."""

    def test_verify_files_oserror_returns_false(self, tmp_path):
        """Verify that OSError in _verify_files_from_metadata is caught and returns False."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        metadata = {"file.txt": {"size": 100, "blob_id": "abc"}}

        # Mock Path.exists to raise OSError
        with patch("pathlib.Path.exists", side_effect=OSError("Permission denied")):
            result = ModelManagement._verify_files_from_metadata(
                model_dir=model_dir, stored_metadata=metadata, repo_files=[]
            )

        assert result is False


class TestVerifyLocalMetadata:
    """Tests for _verify_local_metadata method."""

    def test_verify_local_metadata_success(self, tmp_path):
        """Verify successful metadata parsing and verification."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        metadata_file = snapshot_dir / ".metadata.json"
        metadata = {"file.txt": {"size": 100, "blob_id": "abc"}}
        metadata_file.write_text(json.dumps(metadata))

        with patch.object(ModelManagement, "_verify_files_from_metadata") as mock_verify:
            mock_verify.return_value = True
            result = ModelManagement._verify_local_metadata(
                snapshot_dir, metadata_file, repo_files=[]
            )

        assert result is True
        mock_verify.assert_called_once_with(snapshot_dir, metadata, [])

    @patch("qwen3_embed.common.model_management.logger")
    def test_verify_local_metadata_json_decode_error(self, mock_logger, tmp_path):
        """Verify handling of invalid JSON in metadata file."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        metadata_file = snapshot_dir / ".metadata.json"
        metadata_file.write_text("invalid json")

        result = ModelManagement._verify_local_metadata(snapshot_dir, metadata_file, repo_files=[])

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Failed to read or parse metadata file" in mock_logger.warning.call_args[0][0]

    @patch("qwen3_embed.common.model_management.logger")
    def test_verify_local_metadata_os_error(self, mock_logger, tmp_path):
        """Verify handling of OSError during metadata reading."""
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        metadata_file = snapshot_dir / ".metadata.json"
        metadata_file.write_text("{}")

        with patch("pathlib.Path.read_text", side_effect=OSError("Read error")):
            result = ModelManagement._verify_local_metadata(
                snapshot_dir, metadata_file, repo_files=[]
            )

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Failed to read or parse metadata file" in mock_logger.warning.call_args[0][0]

    def test_verify_local_metadata_not_exists(self, tmp_path):
        """Verify return False if snapshot_dir or metadata_file does not exist."""
        snapshot_dir = tmp_path / "snapshot"
        metadata_file = snapshot_dir / ".metadata.json"

        # Case 1: snapshot_dir doesn't exist
        assert ModelManagement._verify_local_metadata(snapshot_dir, metadata_file, []) is False

        # Case 2: snapshot_dir exists but metadata_file doesn't
        snapshot_dir.mkdir()
        assert ModelManagement._verify_local_metadata(snapshot_dir, metadata_file, []) is False


# ---------------------------------------------------------------------------
# TestGetExpectedHashes
# ---------------------------------------------------------------------------


class TestGetExpectedHashes:
    """Tests for _get_expected_hashes method."""

    def test_get_expected_md5_no_header(self):
        """None should be returned if x-goog-hash header is missing."""
        assert ModelManagement._get_expected_hashes({}) == (None, None)

    def test_get_expected_md5_no_md5(self):
        """None should be returned if x-goog-hash header is present but missing md5."""
        headers = {"x-goog-hash": "crc32c=n9f4Sg=="}
        assert ModelManagement._get_expected_hashes(headers) == (None, None)

    def test_get_expected_md5_success(self):
        """Correct hex MD5 should be returned if md5 is present in x-goog-hash."""
        # "test" md5 is 098f6bcd4621d373cade4e832627b4f6
        # base64 encoded: CY9rzUYh03PK3k6DJie09g==
        headers = {"x-goog-hash": "md5=CY9rzUYh03PK3k6DJie09g=="}
        assert ModelManagement._get_expected_hashes(headers) == (
            "098f6bcd4621d373cade4e832627b4f6",
            None,
        )

    def test_get_expected_md5_multi_part(self):
        """Correct hex MD5 should be returned if multiple hashes are present."""
        headers = {
            "x-goog-hash": "crc32c=n9f4Sg==, md5=CY9rzUYh03PK3k6DJie09g==, sha256=uU0nuZNNPgilLlLX2n2r+sSE7+N6U4DukIj3rOLvzek="
        }
        assert ModelManagement._get_expected_hashes(headers) == (
            "098f6bcd4621d373cade4e832627b4f6",
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )

    def test_get_expected_md5_whitespace(self):
        """Correct hex MD5 should be returned even with extra whitespace."""
        headers = {"x-goog-hash": " crc32c=n9f4Sg== , md5=CY9rzUYh03PK3k6DJie09g== "}
        assert ModelManagement._get_expected_hashes(headers) == (
            "098f6bcd4621d373cade4e832627b4f6",
            None,
        )


# ---------------------------------------------------------------------------
# TestFetchRepoFiles
# ---------------------------------------------------------------------------


class TestFetchRepoFiles:
    """Tests for _fetch_repo_files method."""

    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.list_repo_tree")
    def test_fetch_repo_files_success(self, mock_list_tree, mock_model_info):
        """Verify SHA retrieval and file filtering by extension."""
        mock_model_info.return_value = MagicMock(sha="rev123")

        files = [
            make_repo_file("config.json"),
            make_repo_file("model.onnx"),
            make_repo_file("weights.gguf"),
            make_repo_file("vocab.txt"),
            make_repo_file("README.md"),  # Should be filtered out
            make_repo_file("script.py"),  # Should be filtered out
        ]
        mock_list_tree.return_value = files

        sha, repo_files = ModelManagement._fetch_repo_files("org/repo")

        assert sha == "rev123"
        assert len(repo_files) == 4
        file_paths = {f.path for f in repo_files}
        assert "config.json" in file_paths
        assert "model.onnx" in file_paths
        assert "weights.gguf" in file_paths
        assert "vocab.txt" in file_paths
        assert "README.md" not in file_paths
        assert "script.py" not in file_paths

    @patch("qwen3_embed.common.model_management.model_info")
    def test_fetch_repo_files_no_sha_raises_error(self, mock_model_info):
        """Verify ValueError is raised if SHA is None."""
        mock_model_info.return_value = MagicMock(sha=None)

        with pytest.raises(ValueError, match="Could not determine revision sha"):
            ModelManagement._fetch_repo_files("org/repo")

    @patch("qwen3_embed.common.model_management.model_info")
    @patch("qwen3_embed.common.model_management.list_repo_tree")
    def test_fetch_repo_files_empty_tree(self, mock_list_tree, mock_model_info):
        """Verify empty list is returned if repo tree is empty."""
        mock_model_info.return_value = MagicMock(sha="rev123")
        mock_list_tree.return_value = []

        sha, repo_files = ModelManagement._fetch_repo_files("org/repo")

        assert sha == "rev123"
        assert repo_files == []


class TestValidateTarMember:
    """Tests for _validate_tar_member: must allow safe relative members while
    blocking absolute paths and ``..`` traversal (cross-platform)."""

    def _member(
        self,
        name: str,
        *,
        is_reg: bool = True,
        is_dir: bool = False,
        is_sym: bool = False,
        is_lnk: bool = False,
        linkname: str = "",
    ) -> MagicMock:
        member = MagicMock(spec=tarfile.TarInfo)
        member.name = name
        member.linkname = linkname
        member.isreg.return_value = is_reg
        member.isdir.return_value = is_dir
        member.issym.return_value = is_sym
        member.islnk.return_value = is_lnk
        return member

    # --- safe members must be ALLOWED (regression for previous validation issue) ---

    def test_allows_plain_file(self, tmp_path):
        ModelManagement._validate_tar_member(self._member("file.txt"), str(tmp_path))

    def test_allows_dot_prefixed_file(self, tmp_path):
        # Path normalization should handle the leading dot.
        ModelManagement._validate_tar_member(self._member("./file.txt"), str(tmp_path))

    def test_allows_nested_relative_file(self, tmp_path):
        # Forward slashes are the canonical tar separator and must be accepted.
        ModelManagement._validate_tar_member(self._member("sub/nested/model.onnx"), str(tmp_path))

    def test_allows_directory(self, tmp_path):
        ModelManagement._validate_tar_member(
            self._member("subdir", is_reg=False, is_dir=True), str(tmp_path)
        )

    def test_allows_safe_parent_reference(self, tmp_path):
        # A path containing ".." is safe as long as it stays within the cache directory.
        ModelManagement._validate_tar_member(self._member("sub/../file.txt"), str(tmp_path))

    def test_allows_member_when_cache_dir_has_trailing_separator(self, tmp_path):
        # A trailing separator on cache_dir previously broke the naive
        # ``startswith(cache_dir + os.sep)`` check and wrongly rejected the file.
        cache_dir = str(tmp_path) + os.sep
        ModelManagement._validate_tar_member(self._member("file.txt"), cache_dir)

    # --- real traversal attempts must be BLOCKED ---

    def test_blocks_symlink_even_with_relative_target(self, tmp_path):
        # A target that reads as safe is still refused: it can only be resolved
        # against the filesystem the link will live on, which does not exist yet.
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(
                self._member("dir/link", is_reg=False, is_sym=True, linkname="../file.txt"),
                str(tmp_path),
            )

    def test_blocks_hardlink_even_with_relative_target(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(
                self._member("link", is_reg=False, is_lnk=True, linkname="file.txt"),
                str(tmp_path),
            )

    def test_blocks_parent_traversal_name(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
            ModelManagement._validate_tar_member(self._member("../evil.txt"), str(tmp_path))

    def test_blocks_nested_parent_traversal_name(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
            ModelManagement._validate_tar_member(self._member("sub/../../evil.txt"), str(tmp_path))

    def test_blocks_absolute_posix_name(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
            ModelManagement._validate_tar_member(self._member("/etc/passwd"), str(tmp_path))

    def test_blocks_backslash_rooted_name(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
            ModelManagement._validate_tar_member(
                self._member("\\windows\\system32"), str(tmp_path)
            )

    def test_blocks_unsupported_file_type(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(self._member("dev", is_reg=False), str(tmp_path))

    def test_blocks_absolute_symlink_target(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(
                self._member("link", is_reg=False, is_sym=True, linkname="/etc/passwd"),
                str(tmp_path),
            )

    def test_blocks_parent_traversal_symlink_target(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(
                self._member("link", is_reg=False, is_sym=True, linkname="../../../etc/passwd"),
                str(tmp_path),
            )

    def test_blocks_parent_traversal_hardlink_target(self, tmp_path):
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(
                self._member("link", is_reg=False, is_lnk=True, linkname="../outside.txt"),
                str(tmp_path),
            )


class TestIsWithinDir:
    """Tests for _is_within_dir static method."""

    def test_is_within_dir_absolute_paths_success(self):
        """Test _is_within_dir with absolute paths that resolve correctly."""
        base = "/tmp/base"
        candidate = "/tmp/base/sub/file.txt"
        assert ModelManagement._is_within_dir(base, candidate) is True

    def test_is_within_dir_absolute_paths_failure(self):
        """Test _is_within_dir with absolute paths that are outside the base."""
        base = "/tmp/base"
        candidate = "/tmp/other/file.txt"
        assert ModelManagement._is_within_dir(base, candidate) is False

    def test_is_within_dir_relative_paths_success(self):
        """Test _is_within_dir with relative paths that resolve inside the base."""
        # abspath will resolve relative to the current working directory
        base = "cache"
        candidate = "cache/sub/file.txt"
        assert ModelManagement._is_within_dir(base, candidate) is True

    def test_is_within_dir_relative_paths_failure(self):
        """Test _is_within_dir with relative paths that resolve outside the base."""
        base = "cache"
        candidate = "cache/../../evil.txt"
        assert ModelManagement._is_within_dir(base, candidate) is False

    def test_is_within_dir_sibling_directories(self):
        """Test _is_within_dir with sibling directories sharing a prefix."""
        base = "/tmp/cache"
        candidate = "/tmp/cache-evil/file.txt"
        assert ModelManagement._is_within_dir(base, candidate) is False

    def test_is_within_dir_trailing_separators(self):
        """Test _is_within_dir with trailing separators."""
        base = "/tmp/cache/"
        candidate = "/tmp/cache/file.txt"
        assert ModelManagement._is_within_dir(base, candidate) is True

    def test_is_within_dir_same_path(self):
        """Test _is_within_dir when base and candidate are the same."""
        path = "/tmp/cache"
        assert ModelManagement._is_within_dir(path, path) is True

    def test_is_within_dir_different_drives(self):
        """Test _is_within_dir when paths are on different drives (Windows)."""
        with patch("os.path.commonpath", side_effect=ValueError):
            assert ModelManagement._is_within_dir("C:\\base", "D:\\base") is False
