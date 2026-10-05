"""Local provenance must reject changed sources or binaries before adding hashes."""
import json
import shutil
import subprocess

import pytest

from miniworld_engine.tools import record_anthropic_triattn_build as build


@pytest.fixture
def payload(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "scratch"
    kernel = source / build.PAYLOAD / "cuda_80"
    (kernel / "csrc").mkdir(parents=True)
    (kernel / "csrc/kernel.cu").write_text("original kernel")
    (kernel / "triattn_sm80.py").write_text("original builder")
    (source / build.NATIVE / "SHA256SUMS").write_text("upstream manifest\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(build, "REVISION", revision)
    shutil.copytree(source / "common", root / "common")
    directory = root / build.PAYLOAD / "prebuilt/test-abi"
    directory.mkdir(parents=True)
    (directory / f"{build.EXT}.so").write_bytes(b"local binary")
    rec = {"stack_tag": "test-abi", "ext": build.EXT, "arch": ["arch=compute_80,code=sm_80"],
           "source_sha256": {"kernel.cu": build.sha256(kernel / "csrc/kernel.cu")},
           "module_sha256": {"triattn_sm80.py": build.sha256(kernel / "triattn_sm80.py")},
           "so_sha256": build.sha256(directory / f"{build.EXT}.so")}
    (directory / f"{build.EXT}.json").write_text(json.dumps(rec))
    return source, root, directory


def test_local_record_preserves_upstream_and_is_repeatable(payload):
    source, root, directory = payload
    first = build.record_build(root, source, "test-abi")
    assert first == build.record_build(root, source, "test-abi")
    assert first["verified_upstream_files"] == 3
    assert (root / build.NATIVE / "SHA256SUMS").read_bytes() == (source / build.NATIVE / "SHA256SUMS").read_bytes()
    assert (directory / "SHA256SUMS").read_text() == f"{build.sha256(directory / f'{build.EXT}.so')}  {build.EXT}.so\n"


@pytest.mark.parametrize("changed", ["source", "binary", "builder", "release_manifest"])
def test_changed_payload_is_not_registered(payload, changed):
    source, root, directory = payload
    path = {"source": root / build.PAYLOAD / "cuda_80/csrc/kernel.cu",
            "binary": directory / f"{build.EXT}.so",
            "builder": root / build.PAYLOAD / "cuda_80/triattn_sm80.py",
            "release_manifest": root / build.NATIVE / "SHA256SUMS"}[changed]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"(match the build record|differs from pinned upstream)"):
        build.record_build(root, source, "test-abi")
    assert not (directory / "SHA256SUMS").exists()


def test_original_checkout_cannot_be_registered(payload):
    source, _, _ = payload
    with pytest.raises(ValueError, match="separate scratch"):
        build.record_build(source, source, "test-abi")


def test_existing_local_manifest_is_not_overwritten(payload):
    source, root, directory = payload
    manifest = directory / "SHA256SUMS"
    manifest.write_text("another local build\n")
    with pytest.raises(ValueError, match="different local manifest"):
        build.record_build(root, source, "test-abi")
    assert manifest.read_text() == "another local build\n"
