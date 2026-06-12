"""_auto_push_status must never commit anything except docs/status.json."""
import subprocess

from locus import config
from locus.core import export_status


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_auto_push_does_not_swallow_staged_work(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "docs" / "status.json").write_text('{"v": 1}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")

    # status.json changes (the thing auto-push should commit) ...
    (repo / "docs" / "status.json").write_text('{"v": 2}')
    # ... while a human has unrelated work STAGED mid-task
    (repo / "important.py").write_text("work in progress")
    _git(repo, "add", "important.py")

    monkeypatch.setattr(export_status, "REPO_DIR", repo)
    monkeypatch.setattr(export_status, "_last_push_at", float("-inf"))
    monkeypatch.setattr(config, "AUTO_PUSH_STATUS", True)
    monkeypatch.setattr(config, "AUTO_PUSH_MIN_INTERVAL_SECONDS", 0.0)

    # push will fail (no remote) — _auto_push_status logs and swallows that;
    # the commit has already happened, which is what we're testing.
    export_status._auto_push_status()

    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert committed == ["docs/status.json"], f"commit swept in: {committed}"
    assert _git(repo, "log", "-1", "--format=%s") == "update dashboard data"

    # the staged unrelated file must still be staged, not committed
    assert _git(repo, "diff", "--cached", "--name-only") == "important.py"


def test_auto_push_noop_when_status_unchanged(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "docs" / "status.json").write_text('{"v": 1}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    head = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(export_status, "REPO_DIR", repo)
    monkeypatch.setattr(export_status, "_last_push_at", float("-inf"))
    monkeypatch.setattr(config, "AUTO_PUSH_STATUS", True)
    monkeypatch.setattr(config, "AUTO_PUSH_MIN_INTERVAL_SECONDS", 0.0)

    export_status._auto_push_status()
    assert _git(repo, "rev-parse", "HEAD") == head  # nothing committed
