import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest
import requests
from loguru import logger

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagementExtra:
    """Extra tests to cover missing branches in model_management.py."""

    def test_get_session_init(self):
        """Test that _get_session initializes the session if it's None."""
        # Save original session to restore it later
        original_session = ModelManagement._session
        ModelManagement._session = None
        try:
            session = ModelManagement._get_session()
            assert isinstance(session, requests.Session)
            assert ModelManagement._session is session
            assert session.trust_env is False

            # Second call should return the same session
            session2 = ModelManagement._get_session()
            assert session2 is session
        finally:
            ModelManagement._session = original_session

    def test_decompress_absolute_path_mock(self, tmp_path):
        """Mock getmembers to return an absolute path to trigger line 371."""
        tar_path = tmp_path / "test.tar.gz"
        # Create a dummy tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="safe.txt")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Mock tarfile.open to return a tar with a malicious member
        mock_member = MagicMock()
        mock_member.name = "/absolute/path"

        with patch("tarfile.open") as mock_open:
            mock_tar = mock_open.return_value.__enter__.return_value
            mock_tar.__iter__.return_value = iter([mock_member])
            mock_tar.extractall.side_effect = lambda path, members, filter=None: list(members)

            with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
                ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

    def test_decompress_blocks_symlink_and_hardlink(self, tmp_path):
        """An archive carrying links is refused, whatever the links point at."""
        cache_dir = tmp_path / "cache_links"
        cache_dir.mkdir()

        tar_path = tmp_path / "links.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Regular file
            info = tarfile.TarInfo(name="file.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))

            # Symlink pointing next to itself -- still refused.
            sym_info = tarfile.TarInfo(name="symlink.txt")
            sym_info.type = tarfile.SYMTYPE
            sym_info.linkname = "file.txt"
            tar.addfile(sym_info)

            # Hardlink pointing next to itself -- still refused.
            hard_info = tarfile.TarInfo(name="hardlink.txt")
            hard_info.type = tarfile.LNKTYPE
            hard_info.linkname = "file.txt"
            tar.addfile(hard_info)

        with pytest.raises(
            tarfile.TarError, match="Unsupported file type in tar file: symlink.txt"
        ):
            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

    def test_validate_tar_member_refuses_a_real_symlink(self, tmp_path):
        """``_validate_tar_member`` must refuse a symlink read out of a real archive.

        The member is a genuine ``TarInfo`` parsed back from a genuine tar, not a
        stand-in: link handling is the part of ``tarfile`` under test here, so a
        stubbed member would only assert what the stub was told to say. The
        earlier check called a link safe whenever ``os.path.abspath`` of its
        target stayed inside the cache -- a purely textual reading, since
        ``abspath`` collapses ``..`` without consulting the filesystem.
        """
        tar_path = tmp_path / "link.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            sym_info = tarfile.TarInfo(name="link")
            sym_info.type = tarfile.SYMTYPE
            sym_info.linkname = ".."
            tar.addfile(sym_info)

        with tarfile.open(tar_path, "r:gz") as tar:
            member = next(iter(tar))

        assert member.issym()
        with pytest.raises(tarfile.TarError, match="Unsupported file type"):
            ModelManagement._validate_tar_member(member, str(tmp_path))

    def test_extraction_refuses_a_symlink_escaping_the_cache(self, tmp_path):
        """A symlink chain that reads as safe but escapes on disk must be refused.

        Three members are enough to walk out of the cache directory.
        ``sub/link -> ..`` is inside the cache under either reading. But
        ``sub/link/../pwned.txt`` normalises to ``sub/pwned.txt`` on paper, while
        on disk -- once ``sub/link`` exists -- it resolves one level *above* the
        cache. Extracting this archive wrote ``pwned.txt`` outside the cache
        directory on Linux before the type check was tightened.

        The error must name the *link* as an unsupported type, which is this
        package refusing the archive. On Python 3.11.4+ ``tarfile``'s own "data"
        filter would also stop this one, a step later and with a different
        message -- but ``requires-python = ">=3.11"`` also admits 3.11.0-3.11.3,
        where ``decompress_to_cache`` falls back to a manual loop with no such
        backstop. That fallback is where the escape actually landed, so the
        refusal has to come from here rather than from the standard library.

        The archive is real: the escape is a property of how the filesystem
        resolves a real symlink, and a stubbed ``tarfile`` could not exhibit it.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        tar_path = tmp_path / "evil.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            sub_info = tarfile.TarInfo(name="sub")
            sub_info.type = tarfile.DIRTYPE
            tar.addfile(sub_info)

            link_info = tarfile.TarInfo(name="sub/link")
            link_info.type = tarfile.SYMTYPE
            link_info.linkname = ".."
            tar.addfile(link_info)

            payload_info = tarfile.TarInfo(name="sub/link/../pwned.txt")
            payload_info.size = 5
            tar.addfile(payload_info, io.BytesIO(b"PWNED"))

        with pytest.raises(tarfile.TarError, match="Unsupported file type in tar file: sub/link"):
            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        assert not (tmp_path / "pwned.txt").exists()

    def test_decompress_no_data_filter(self, tmp_path):
        """Cover fallback by mocking tarfile to lack data_filter and verify manual extraction loop."""
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="file.txt")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        cache_dir = tmp_path / "cache_no_filter"
        cache_dir.mkdir()

        # We patch the tarfile module in the model_management namespace
        with patch("qwen3_embed.common.model_management.tarfile") as mock_tarfile_mod:
            # Setup mock_tar
            mock_tar = MagicMock()
            mock_tarfile_mod.open.return_value.__enter__.return_value = mock_tar

            # Mock getmembers to return a list of members
            member = MagicMock()
            member.name = "file.txt"
            member.isreg.return_value = True
            member.isdir.return_value = False
            member.issym.return_value = False
            member.islnk.return_value = False
            member.size = 0
            mock_tar.__iter__.return_value = iter([member])
            mock_tarfile_mod.TarError = tarfile.TarError

            # Ensure hasattr(tarfile, 'data_filter') returns False
            del mock_tarfile_mod.data_filter

            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

            # Verify extract was called for the member (since extractall is no longer called in fallback)
            mock_tar.extract.assert_called_once_with(member, path=str(cache_dir))
            # Verify metadata sanitization
            # Since member.mode was a MagicMock, &= operation on it might be tricky to verify this way
            # In our code: member.mode &= 0o777 (which is member.mode = member.mode.__and__(0o777))
            # But member is also a MagicMock, so we can check attributes directly if we set them before
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""

    def test_decompress_logging_on_error(self, tmp_path):
        """Verify logger.error is called on TarError."""
        tar_path = tmp_path / "corrupt.tar.gz"
        tar_path.write_text("not a tar file")

        cache_dir = tmp_path / "cache_err"
        cache_dir.mkdir()

        with patch.object(logger, "error") as mock_log_error:
            with pytest.raises(tarfile.TarError):
                ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

            mock_log_error.assert_called()
            # Verify the log message contains the filename
            args, _ = mock_log_error.call_args
            assert str(tar_path) in args[0]

    @patch("qwen3_embed.common.model_management.model_info")
    def test_fetch_repo_files_no_sha_raises_value_error(self, mock_model_info):
        """Verify ValueError is raised if repo revision sha is None (line 289)."""
        mock_model_info.return_value = MagicMock(sha=None)
        repo = "org/repo"
        with pytest.raises(
            ValueError, match=f"Could not determine revision sha for repo '{repo}'"
        ):
            ModelManagement._fetch_repo_files(repo)

    def test_is_within_dir_same_path(self):
        """Test _is_within_dir with the same path for base and candidate."""
        path = "/tmp/cache"
        assert ModelManagement._is_within_dir(path, path) is True

    def test_is_within_dir_value_error(self):
        """Test _is_within_dir when commonpath raises ValueError."""
        with patch("os.path.commonpath", side_effect=ValueError):
            assert ModelManagement._is_within_dir("/tmp/base", "/tmp/candidate") is False

    def test_validate_tar_member_current_dir(self, tmp_path):
        """Test _validate_tar_member with a member named '.' (current directory)."""
        member = MagicMock(spec=tarfile.TarInfo)
        member.name = "."
        member.isreg.return_value = False
        member.isdir.return_value = True
        member.issym.return_value = False
        member.islnk.return_value = False

        # This should not raise an error and will cover base == candidate in _is_within_dir
        ModelManagement._validate_tar_member(member, str(tmp_path))

    def test_blocks_sibling_directory_traversal(self, tmp_path):
        """Test that _validate_tar_member blocks traversal into a sibling directory."""
        # cache-evil is a sibling of cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        member = MagicMock(spec=tarfile.TarInfo)
        member.name = "../cache-evil/file.txt"
        member.isreg.return_value = True
        member.isdir.return_value = False
        member.issym.return_value = False
        member.islnk.return_value = False

        with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
            ModelManagement._validate_tar_member(member, str(cache_dir))


def _symlinks_available(tmp_path) -> bool:
    """Whether this process may create symlinks (Windows needs a privilege)."""
    probe = tmp_path / "_symlink_probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


class TestPrepareGcsCacheCleanup:
    """``_prepare_gcs_cache`` must clear the staging path whatever is sitting on it.

    The path is a fixed, predictable name under a shared cache root, so a local
    user can plant something there before a download starts. ``shutil.rmtree``
    accepts only a real directory: anything else made the call raise and left
    every subsequent download failing at the same spot.
    """

    def test_replaces_a_leftover_regular_file(self, tmp_path):
        """A file where the staging directory belongs is removed, not passed to rmtree."""
        cache_tmp_dir = tmp_path / "cache"
        model_tmp_dir = tmp_path / "model_tmp"
        model_tar_gz = tmp_path / "model.tar.gz"
        model_tmp_dir.write_text("leftover from an interrupted run")

        ModelManagement._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)

        assert not model_tmp_dir.exists()
        assert cache_tmp_dir.is_dir()

    def test_removes_a_symlink_without_touching_its_target(self, tmp_path):
        """The link is unlinked; the directory it pointed at keeps its contents.

        ``rmtree`` refuses a symlink outright, so before the guard this raised.
        The assertion on ``victim`` is the part that matters: whatever replaces
        the guard must never follow the link and delete through it.
        """
        if not _symlinks_available(tmp_path):
            pytest.skip("creating symlinks is not permitted in this environment")

        cache_tmp_dir = tmp_path / "cache"
        model_tmp_dir = tmp_path / "model_tmp"
        model_tar_gz = tmp_path / "model.tar.gz"

        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("must survive")
        model_tmp_dir.symlink_to(victim, target_is_directory=True)

        ModelManagement._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)

        assert not model_tmp_dir.is_symlink()
        assert not model_tmp_dir.exists()
        assert (victim / "keep.txt").read_text() == "must survive"

    def test_removes_a_dangling_symlink(self, tmp_path):
        """A broken link reads as absent through ``exists`` and must still be cleared.

        This is the case an ``exists``-first guard misses: the link survives
        ``_prepare_gcs_cache`` and the later extract and rename follow it.
        """
        if not _symlinks_available(tmp_path):
            pytest.skip("creating symlinks is not permitted in this environment")

        cache_tmp_dir = tmp_path / "cache"
        model_tmp_dir = tmp_path / "model_tmp"
        model_tar_gz = tmp_path / "model.tar.gz"
        model_tmp_dir.symlink_to(tmp_path / "does_not_exist", target_is_directory=True)

        assert not model_tmp_dir.exists()
        assert model_tmp_dir.is_symlink()

        ModelManagement._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)

        assert not model_tmp_dir.is_symlink()

    def test_still_removes_a_real_stale_directory(self, tmp_path):
        """The ordinary path keeps working: a real directory is deleted recursively."""
        cache_tmp_dir = tmp_path / "cache"
        model_tmp_dir = tmp_path / "model_tmp"
        model_tar_gz = tmp_path / "model.tar.gz"
        (model_tmp_dir / "nested").mkdir(parents=True)
        (model_tmp_dir / "nested" / "stale.bin").write_bytes(b"stale")

        ModelManagement._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)

        assert not model_tmp_dir.exists()

    def test_removes_a_stale_archive(self, tmp_path):
        """A leftover tarball is unlinked so the download cannot append to it."""
        cache_tmp_dir = tmp_path / "cache"
        model_tmp_dir = tmp_path / "model_tmp"
        model_tar_gz = tmp_path / "model.tar.gz"
        model_tar_gz.write_bytes(b"partial download")

        ModelManagement._prepare_gcs_cache(cache_tmp_dir, model_tmp_dir, model_tar_gz)

        assert not model_tar_gz.exists()
