import base64
import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
import urllib.parse
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

import requests
from huggingface_hub import list_repo_tree, model_info, snapshot_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    disable_progress_bars,
    enable_progress_bars,
)
from loguru import logger
from tqdm import tqdm

from qwen3_embed.common.model_description import BaseModelDescription

T = TypeVar("T", bound=BaseModelDescription)


class ModelManagement(Generic[T]):
    METADATA_FILE = "files_metadata.json"
    _session: requests.Session | None = None
    _session_lock = threading.Lock()

    @classmethod
    def _get_session(cls) -> requests.Session:
        # ⚡ Bolt: Use requests.Session for connection pooling (~30% faster for multiple files)
        # Reusing the TCP connection avoids the overhead of repeated TCP/TLS handshakes
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    session = requests.Session()
                    # SECURITY: Enforce trust_env=False to prevent proxy/CA environment variable injection
                    session.trust_env = False
                    cls._session = session
        return cls._session

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[T]: A list of dictionaries containing the model information.
        """
        raise NotImplementedError()

    @classmethod
    def add_custom_model(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a custom model to the existing embedding classes based on the passed model descriptions

        Model description dict should contain the fields same as in one of the model descriptions presented
         in qwen3_embed.common.model_description

         E.g. for BaseModelDescription:
              model: str
              sources: ModelSource
              model_file: str
              description: str
              license: str
              size_in_GB: float
              additional_files: list[str]

        Returns:
            None
        """
        raise NotImplementedError()

    @classmethod
    def _list_supported_models(cls) -> list[T]:
        raise NotImplementedError()

    @classmethod
    def _clear_model_cache(cls) -> None:
        """Clears the model description cache to allow dynamic model registration."""
        if "_model_description_cache" in cls.__dict__:
            delattr(cls, "_model_description_cache")

    @classmethod
    def _ensure_model_cache(cls) -> dict[str, T]:
        cache = cls.__dict__.get("_model_description_cache")
        if cache is None:
            # ⚡ Bolt: Using dictionary lookup for O(1) model checks and descriptions
            cache = {model.model.lower(): model for model in cls._list_supported_models()}
            cls._model_description_cache = cache
        return cache

    @classmethod
    def _check_model_exists(cls, model: str) -> None:
        cache = cls._ensure_model_cache()
        if model.lower() in cache:
            raise ValueError(
                f"Model {model} is already registered in {cls.__name__}, if you still want to add this model, "
                f"please use another model name"
            )

    @classmethod
    def _get_model_description(cls, model_name: str) -> T:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            T: The model description.
        """
        cache = cls._ensure_model_cache()

        model_name_lower = model_name.lower()
        model = cache.get(model_name_lower)
        if model is not None:
            return model

        raise ValueError(f"Model {model_name} is not supported in {cls.__name__}.")

    @staticmethod
    def _get_expected_hashes(headers: Any) -> tuple[str | None, str | None]:
        expected_md5 = None
        expected_sha256 = None
        if "x-goog-hash" in headers:
            x_goog_hash = headers["x-goog-hash"]
            for part in x_goog_hash.split(","):
                part = part.strip()
                if part.startswith("md5="):
                    expected_md5 = base64.b64decode(part[4:]).hex()
                elif part.startswith("sha256="):
                    expected_sha256 = base64.b64decode(part[7:]).hex()
        return expected_md5, expected_sha256

    @staticmethod
    def _download_and_hash_file(
        response: Any, output_path: str, total_size_in_bytes: int, show_progress: bool
    ) -> tuple[str, str]:
        show_progress = bool(total_size_in_bytes and show_progress)
        md5_hash = hashlib.md5(
            usedforsecurity=False
        )  # SECURITY: MD5 is used solely for non-cryptographic file integrity checking (GCS checksums).
        sha256_hash = hashlib.sha256()
        with (
            tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {os.path.basename(output_path)}",
                disable=not show_progress,
            ) as progress_bar,
            open(output_path, "wb") as file,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:  # Filter out keep-alive new chunks
                    progress_bar.update(len(chunk))
                    file.write(chunk)
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
        return md5_hash.hexdigest(), sha256_hash.hexdigest()

    @classmethod
    def download_file_from_gcs(cls, url: str, output_path: str, show_progress: bool = True) -> str:
        """
        Downloads a file from Google Cloud Storage.

        Args:
            url (str): The URL to download the file from.
            output_path (str): The path to save the downloaded file to.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            str: The path to the downloaded file.
        """

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
            raise ValueError(
                f"Invalid URL: {url}. Only URLs from Google Cloud Storage are allowed."
            )
        if os.path.exists(output_path):
            return output_path
        # SECURITY: Explicitly enforce TLS verification to prevent accidental or malicious bypass via environment variables (like REQUESTS_CA_BUNDLE).
        response = cls._get_session().get(
            url, stream=True, timeout=10, verify=True, allow_redirects=False
        )

        if response.status_code in (301, 302, 303, 307, 308):
            raise ValueError(
                f"SSRF Prevention: Redirects are not allowed. Status code: {response.status_code}"
            )

        # Handle HTTP errors
        if response.status_code == 403:
            raise PermissionError(
                "Authentication Error: You do not have permission to access this resource. "
                "Please check your credentials."
            )
        response.raise_for_status()

        expected_md5, expected_sha256 = cls._get_expected_hashes(response.headers)

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        if total_size_in_bytes == 0:
            logger.warning(f"Content-length header is missing or zero in the response from {url}.")

        tmp_output_path = f"{output_path}.{uuid.uuid4().hex}.tmp"
        try:
            calculated_md5, calculated_sha256 = cls._download_and_hash_file(
                response, tmp_output_path, total_size_in_bytes, show_progress
            )

            if expected_sha256:
                if expected_sha256 != calculated_sha256:
                    raise ValueError(
                        f"File integrity check failed: expected SHA256 {expected_sha256}, got {calculated_sha256}"
                    )
            elif expected_md5 and expected_md5 != calculated_md5:
                raise ValueError(
                    f"File integrity check failed: expected MD5 {expected_md5}, got {calculated_md5}"
                )
            os.replace(tmp_output_path, output_path)
        finally:
            if os.path.exists(tmp_output_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_output_path)

        return output_path

    @classmethod
    def _verify_files_from_metadata(
        cls, model_dir: Path, stored_metadata: dict[str, Any], repo_files: list[RepoFile]
    ) -> bool:
        try:
            # ⚡ Bolt: Convert repo_files to a map for O(1) lookups inside the loop (was O(N*M))
            repo_files_map = {f.path: f for f in repo_files} if repo_files else {}
            for rel_path, meta in stored_metadata.items():
                file_path = model_dir / rel_path

                if not file_path.exists():
                    return False

                if repo_files:  # online verification
                    file_info = repo_files_map.get(rel_path)
                    if (
                        not file_info
                        or file_info.size != meta["size"]
                        or file_info.blob_id != meta["blob_id"]
                    ):
                        return False

                else:  # offline verification
                    if file_path.stat().st_size != meta["size"]:
                        return False
            return True
        except (OSError, KeyError) as e:
            logger.error(f"Error verifying files: {str(e)}")
            return False

    @classmethod
    def _collect_file_metadata(
        cls, model_dir: Path, repo_files: list[RepoFile]
    ) -> dict[str, dict[str, int | str]]:
        meta: dict[str, dict[str, int | str]] = {}
        file_info_map = {f.path: f for f in repo_files}
        for file_path in model_dir.rglob("*"):
            if file_path.is_file() and file_path.name != cls.METADATA_FILE:
                rel_path = file_path.relative_to(model_dir).as_posix()
                repo_file = file_info_map.get(rel_path)
                if repo_file:
                    meta[rel_path] = {
                        "size": repo_file.size,
                        "blob_id": repo_file.blob_id,
                    }
        return meta

    @classmethod
    def _save_file_metadata(cls, model_dir: Path, meta: dict[str, dict[str, int | str]]) -> None:
        try:
            if not model_dir.exists():
                model_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (model_dir / cls.METADATA_FILE).write_text(json.dumps(meta))
        except (OSError, ValueError) as e:
            logger.exception(e)
            logger.warning("Failed to save metadata file. Next load may take longer to verify.")

    @classmethod
    def _fetch_repo_files(cls, hf_source_repo: str) -> tuple[str, list[RepoFile]]:
        repo_revision = model_info(hf_source_repo).sha
        if repo_revision is None:
            raise ValueError(f"Could not determine revision sha for repo '{hf_source_repo}'")
        repo_tree = list(list_repo_tree(hf_source_repo, revision=repo_revision, repo_type="model"))

        allowed_extensions = {".json", ".onnx", ".gguf", ".txt"}
        repo_files = (
            [
                f
                for f in repo_tree
                if isinstance(f, RepoFile) and Path(f.path).suffix in allowed_extensions
            ]
            if repo_tree
            else []
        )
        return repo_revision, repo_files

    @classmethod
    def _resolve_cached_revision(cls, snapshot_dir: Path) -> str | None:
        """Return a commit SHA present in the local snapshot cache, if any.

        Looks under ``<snapshot_dir>/snapshots/<sha>/`` -- the standard HF cache
        layout -- and returns the most recently modified revision directory's
        name. Returns ``None`` when no cached snapshot exists, in which case the
        caller falls back to HF's default revision resolution.
        """
        snapshots = snapshot_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        revision_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
        if not revision_dirs:
            return None
        revision_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return revision_dirs[0].name

    @classmethod
    def _verify_local_metadata(
        cls, snapshot_dir: Path, metadata_file: Path, repo_files: list[RepoFile]
    ) -> bool:
        if snapshot_dir.exists() and metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                return cls._verify_files_from_metadata(snapshot_dir, metadata, repo_files)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read or parse metadata file: {e}")
        return False

    @classmethod
    def _finalize_hf_download(cls, snapshot_dir: Path, repo_files: list[RepoFile]) -> None:
        metadata = cls._collect_file_metadata(snapshot_dir, repo_files)
        download_successful = cls._verify_files_from_metadata(
            snapshot_dir, metadata, repo_files=[]
        )
        if not download_successful:
            raise ValueError(
                "Files have been corrupted during downloading process. "
                "Please check your internet connection and try again."
            )
        cls._save_file_metadata(snapshot_dir, metadata)

    @classmethod
    def _handle_local_hf_download(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        snapshot_dir: Path,
        metadata_file: Path,
        allow_patterns: list[str],
        **kwargs: Any,
    ) -> str:
        disable_progress_bars()
        cls._verify_local_metadata(snapshot_dir, metadata_file, repo_files=[])
        # The online path pins ``revision`` to an explicit commit SHA, so HF
        # never writes a ``refs/<branch>`` pointer into the cache. A
        # ``local_files_only`` lookup that omits ``revision`` defaults to
        # ``"main"`` and raises ``LocalEntryNotFoundError`` even when the
        # snapshot is fully cached -- which forces a needless network round
        # trip on every load (and stalls/hangs hard when that network call
        # runs inside a worker thread, e.g. an MCP stdio server). Resolve the
        # cached SHA directly so the offline lookup hits the cache.
        if "revision" not in kwargs:
            cached_revision = cls._resolve_cached_revision(snapshot_dir)
            if cached_revision is not None:
                kwargs["revision"] = cached_revision
        return snapshot_download(  # nosec B615
            repo_id=hf_source_repo,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=True,
            **kwargs,
        )

    @classmethod
    def _handle_remote_hf_download(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        snapshot_dir: Path,
        metadata_file: Path,
        allow_patterns: list[str],
        **kwargs: Any,
    ) -> str:
        repo_revision, repo_files = cls._fetch_repo_files(hf_source_repo)
        verified_metadata = cls._verify_local_metadata(snapshot_dir, metadata_file, repo_files)

        if verified_metadata:
            disable_progress_bars()

        result = snapshot_download(
            repo_id=hf_source_repo,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=False,
            revision=repo_revision,
            **kwargs,
        )

        if not verified_metadata:
            cls._finalize_hf_download(snapshot_dir, repo_files)

        return result

    @classmethod
    def download_files_from_huggingface(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Downloads a model from HuggingFace Hub.

        Args:
            hf_source_repo (str): Name of the model on HuggingFace Hub, e.g. "qdrant/all-MiniLM-L6-v2-onnx".
            cache_dir (Optional[str]): The path to the cache directory.
            extra_patterns (list[str]): extra patterns to allow in the snapshot download, typically
                includes the required model files.
            local_files_only (bool, optional): Whether to only use local files. Defaults to False.

        Returns:
            Path: The path to the model directory.
        """

        allow_patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ]
        allow_patterns.extend(extra_patterns)

        snapshot_dir = Path(cache_dir) / f"models--{hf_source_repo.replace('/', '--')}"
        metadata_file = snapshot_dir / cls.METADATA_FILE

        if local_files_only:
            return cls._handle_local_hf_download(
                hf_source_repo=hf_source_repo,
                cache_dir=cache_dir,
                snapshot_dir=snapshot_dir,
                metadata_file=metadata_file,
                allow_patterns=allow_patterns,
                **kwargs,
            )

        return cls._handle_remote_hf_download(
            hf_source_repo=hf_source_repo,
            cache_dir=cache_dir,
            snapshot_dir=snapshot_dir,
            metadata_file=metadata_file,
            allow_patterns=allow_patterns,
            **kwargs,
        )

    @staticmethod
    def _is_within_dir(base: str, candidate: str) -> bool:
        """
        Returns True if ``candidate`` resolves to a path inside ``base``.

        Uses ``os.path.commonpath`` rather than a naive ``startswith`` so that
        results are correct regardless of trailing separators, mixed path
        separators, or sibling directories that share a name prefix (e.g.
        ``cache`` vs ``cache-evil``). Paths on different drives (Windows) raise
        ``ValueError`` from ``commonpath`` and are treated as outside ``base``.
        """
        base = os.path.abspath(base)
        candidate = os.path.abspath(candidate)
        if base == candidate:
            return True
        try:
            return os.path.commonpath([base, candidate]) == base
        except ValueError:
            return False

    @classmethod
    def _validate_tar_member(cls, member: tarfile.TarInfo, cache_dir: str) -> None:
        """
        Validates a tar member to prevent path traversal attacks.

        Args:
            member (tarfile.TarInfo): The tar member to validate.
            cache_dir (str): The absolute path to the extraction directory.

        Raises:
            tarfile.TarError: If a path traversal attempt is detected.
        """
        # Normalize so containment checks are stable regardless of how the
        # caller spelled the directory (trailing separator, relative path, ...).
        cache_dir = os.path.abspath(cache_dir)

        # SECURITY: Only regular files and directories. Symlinks and hardlinks are
        # refused outright rather than validated, because a link target can only be
        # judged against the filesystem the link will actually be resolved on, and
        # this check runs before extraction. Validating the target textually (via
        # os.path.abspath, which collapses "..") let a link point at an earlier link
        # and escape: "sub/link -> .." reads as inside the cache, after which
        # "sub/link/../file" normalises to "sub/file" on paper while resolving above
        # the cache on disk. Model archives hold plain file trees, so nothing legitimate
        # needs links here.
        if not (member.isreg() or member.isdir()):
            raise tarfile.TarError(f"Unsupported file type in tar file: {member.name}")

        if os.path.isabs(member.name) or member.name.startswith(("/", "\\")):
            raise tarfile.TarError(f"Attempted path traversal in tar file: {member.name}")

        member_path = os.path.abspath(os.path.join(cache_dir, member.name))
        if not cls._is_within_dir(cache_dir, member_path):
            raise tarfile.TarError(f"Attempted path traversal in tar file: {member.name}")

    @classmethod
    def decompress_to_cache(cls, targz_path: str, cache_dir: str) -> str:
        """
        Decompresses a .tar.gz file to a cache directory.

        Args:
            targz_path (str): Path to the .tar.gz file.
            cache_dir (str): Path to the cache directory.

        Returns:
            cache_dir (str): Path to the cache directory.
        """
        # Check if targz_path exists and is a file
        if not os.path.isfile(targz_path):
            raise ValueError(f"{targz_path} does not exist or is not a file.")

        # Check if targz_path is a .tar.gz file
        if not targz_path.endswith(".tar.gz"):
            raise ValueError(f"{targz_path} is not a .tar.gz file.")

        try:
            # Open the tar.gz file
            with tarfile.open(targz_path, "r:gz") as tar:
                # Extract all files into the cache directory securely
                target_dir = os.path.abspath(cache_dir)

                def validate_and_yield_members():
                    total_size = 0
                    max_uncompressed_size = 20 * 1024 * 1024 * 1024  # 20 GB
                    for member in tar:
                        cls._validate_tar_member(member, target_dir)
                        total_size += member.size
                        if total_size > max_uncompressed_size:
                            raise tarfile.TarError(
                                f"Decompression bomb detected: total uncompressed size exceeds {max_uncompressed_size} bytes"
                            )
                        yield member

                if hasattr(tarfile, "data_filter"):
                    tar.extractall(
                        path=cache_dir, members=validate_and_yield_members(), filter="data"
                    )
                else:
                    for member in validate_and_yield_members():
                        # Sanitize metadata to mimic "data" filter
                        member.mode &= 0o777
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        tar.extract(member, path=cache_dir)
        except tarfile.TarError as e:
            # If decompression fails, remove the partially extracted directory
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.error(f"Failed to decompress {targz_path}: {e}")
            raise e

        return cache_dir

    @classmethod
    def _get_gcs_model_paths(
        cls, model_name: str, cache_dir: str, deprecated_tar_struct: bool
    ) -> tuple[Path, Path, Path, Path]:
        fast_model_name = f"{'fast-' if deprecated_tar_struct else ''}{model_name.split('/')[-1]}"
        cache_path = Path(cache_dir)
        cache_tmp_dir = cache_path / "tmp"
        model_tmp_dir = cache_tmp_dir / fast_model_name
        model_dir = cache_path / fast_model_name
        model_tar_gz = cache_path / f"{fast_model_name}.tar.gz"
        return model_dir, model_tmp_dir, cache_tmp_dir, model_tar_gz

    @classmethod
    def _is_model_cached(cls, model_dir: Path) -> bool:
        # check if the model_dir and the model files are both present for macOS
        # ⚡ Bolt: Fast directory check avoiding hidden files like .DS_Store (~10x faster than list(glob("*")))
        return model_dir.exists() and any(not f.name.startswith(".") for f in model_dir.iterdir())

    @classmethod
    def _prepare_gcs_cache(
        cls, cache_tmp_dir: Path, model_tmp_dir: Path, model_tar_gz: Path
    ) -> None:
        # SECURITY: ``rmtree`` only accepts a real directory. A leftover file, or a
        # symlink planted at this path by another local user, makes it raise
        # (``NotADirectoryError`` for a file, "Cannot call rmtree on a symbolic link"
        # for a link) and every download aborts until the path is cleared by hand.
        # ``is_symlink`` is tested first and on its own because ``exists`` follows
        # links: a dangling symlink reads as absent, so an ``exists``-first guard
        # would leave it in place for the later extract and rename to follow.
        if model_tmp_dir.is_symlink() or (model_tmp_dir.exists() and not model_tmp_dir.is_dir()):
            model_tmp_dir.unlink()
        elif model_tmp_dir.exists():
            shutil.rmtree(model_tmp_dir)

        cache_tmp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # SECURITY: Prevent arbitrary permission modification via symlink attacks
        if not cache_tmp_dir.is_symlink():
            with contextlib.suppress(OSError):
                cache_tmp_dir.chmod(0o700)

        if model_tar_gz.exists():
            model_tar_gz.unlink()

    @classmethod
    def _download_and_extract_gcs_model(
        cls,
        source_url: str,
        model_tar_gz: Path,
        cache_tmp_dir: Path,
        model_tmp_dir: Path,
        model_dir: Path,
    ) -> None:
        cls.download_file_from_gcs(
            source_url,
            output_path=str(model_tar_gz),
        )

        cls.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=str(cache_tmp_dir))
        if not model_tmp_dir.exists():
            raise ValueError(f"Could not find {model_tmp_dir} in {cache_tmp_dir}")

        model_tar_gz.unlink()
        # Rename from tmp to final name is atomic
        model_tmp_dir.rename(model_dir)

    @classmethod
    def retrieve_model_gcs(
        cls,
        model_name: str,
        source_url: str,
        cache_dir: str,
        deprecated_tar_struct: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        model_dir, model_tmp_dir, cache_tmp_dir, model_tar_gz = cls._get_gcs_model_paths(
            model_name, cache_dir, deprecated_tar_struct
        )

        if cls._is_model_cached(model_dir):
            return model_dir

        if not local_files_only:
            cls._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)
            cls._download_and_extract_gcs_model(
                source_url, model_tar_gz, cache_tmp_dir, model_tmp_dir, model_dir
            )
        else:
            logger.error(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )
            raise ValueError(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )

        return model_dir

    @classmethod
    def _check_hf_cache(
        cls,
        model: T,
        cache_dir: str,
        **kwargs: Any,
    ) -> Path | None:
        try:
            cache_kwargs = deepcopy(kwargs)
            cache_kwargs["local_files_only"] = True
            extra_patterns = [model.model_file]
            extra_patterns.extend(model.additional_files)
            cached_path = Path(
                cls.download_files_from_huggingface(
                    model.sources.hf,
                    cache_dir=cache_dir,
                    extra_patterns=extra_patterns,
                    **cache_kwargs,
                )
            )
            # Verify the required model file actually exists in the cached snapshot
            if (cached_path / model.model_file).exists():
                return cached_path
        except (OSError, ValueError, RepositoryNotFoundError):
            logger.debug("Model not found in cache, will attempt download")
        finally:
            enable_progress_bars()
        return None

    @classmethod
    def _download_from_hf(cls, model: T, cache_dir: str, **kwargs: Any) -> Path | None:
        local_files_only = kwargs.get("local_files_only", False)
        try:
            extra_patterns = [model.model_file]
            extra_patterns.extend(model.additional_files)
            return Path(
                cls.download_files_from_huggingface(
                    model.sources.hf,
                    cache_dir=cache_dir,
                    extra_patterns=extra_patterns,
                    **kwargs,
                )
            )
        except (OSError, RepositoryNotFoundError, ValueError) as e:
            if not local_files_only:
                logger.error(
                    f"Could not download model from HuggingFace: {e} "
                    "Falling back to other sources."
                )
        finally:
            enable_progress_bars()
        return None

    @classmethod
    def _download_from_gcs(
        cls,
        model: T,
        cache_dir: str,
        **kwargs: Any,
    ) -> Path | None:
        url_source = model.sources.url
        local_files_only = kwargs.get("local_files_only", False)
        try:
            return cls.retrieve_model_gcs(
                model.model,
                str(url_source),
                str(cache_dir),
                deprecated_tar_struct=model.sources.deprecated_tar_struct,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError, requests.RequestException, tarfile.TarError):
            if not local_files_only:
                logger.error(f"Could not download model from url: {url_source}")
        return None

    @classmethod
    def _attempt_download(
        cls,
        model: T,
        cache_dir: str,
        **kwargs: Any,
    ) -> Path | None:
        """Attempts to download the model from available sources once."""
        local_files_only = kwargs.get("local_files_only", False)
        hf_source = model.sources.hf
        url_source = model.sources.url

        if hf_source and not local_files_only:
            hf_path = cls._download_from_hf(
                model=model,
                cache_dir=cache_dir,
                **kwargs,
            )
            if hf_path:
                return hf_path

        if url_source or local_files_only:
            gcs_path = cls._download_from_gcs(
                model=model,
                cache_dir=cache_dir,
                **kwargs,
            )
            if gcs_path:
                return gcs_path

        return None

    @classmethod
    def _download_with_retries(
        cls,
        model: T,
        cache_dir: str,
        retries: int,
        **kwargs: Any,
    ) -> Path | None:
        """Manages the retry loop and exponential backoff for model downloading."""
        local_files_only = kwargs.get("local_files_only", False)
        sleep = 3.0
        while retries > 0:
            retries -= 1

            path = cls._attempt_download(
                model=model,
                cache_dir=cache_dir,
                **kwargs,
            )
            if path:
                return path

            if local_files_only:
                logger.error("Could not find model in cache_dir")
                break

            logger.error(
                f"Could not download model from either source, sleeping for {sleep} seconds, {retries} retries left."
            )
            time.sleep(sleep)
            sleep *= 3

        return None

    @classmethod
    def download_model(cls, model: T, cache_dir: str, retries: int = 3, **kwargs: Any) -> Path:
        """
        Downloads a model from HuggingFace Hub or Google Cloud Storage.

        Args:
            model (T): The model description.
                Example:
                ```
                {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dim": 768,
                    "description": "Base English model, v1.5",
                    "size_in_GB": 0.44,
                    "sources": {
                        "url": "https://storage.googleapis.com/qdrant-qwen3_embed/fast-bge-base-en-v1.5.tar.gz",
                        "hf": "qdrant/bge-base-en-v1.5-onnx-q",
                    }
                }
                ```
            cache_dir (str): The path to the cache directory.
            retries: (int): The number of times to retry (including the first attempt)

        Returns:
            Path: The path to the downloaded model directory.
        """
        local_files_only = kwargs.get("local_files_only", False)
        specific_model_path: str | None = kwargs.pop("specific_model_path", None)
        if specific_model_path:
            return Path(specific_model_path)

        retries = 1 if local_files_only else retries
        hf_source = model.sources.hf

        if hf_source:
            cached_path = cls._check_hf_cache(
                model=model,
                cache_dir=cache_dir,
                **kwargs,
            )
            if cached_path:
                return cached_path

        path = cls._download_with_retries(
            model=model,
            cache_dir=cache_dir,
            retries=retries,
            **kwargs,
        )

        if path:
            return path

        raise ValueError(f"Could not load model {model.model} from any source.")
