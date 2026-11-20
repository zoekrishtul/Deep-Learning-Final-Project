"""Save experiment results to a shared Git branch while runs are in progress (Gaurav).

Kaggle stops a session after 12 hours and clears /kaggle/working between sessions.
This module keeps each person's result files on a shared branch (default
``final-results``) so that:

  * nothing has to be downloaded by hand: files are pushed every few minutes and
    after every finished job;
  * a session that is cut off can simply be run again: ``restore()`` copies the
    saved files back, and ``run()`` resumes from them;
  * several people can push to the same branch at once without overwriting each
    other: each instance only ever writes the files it owns (its own jobs), and a
    rejected push is retried after rebasing on the others' commits.

JSONL files are copied up to their last complete line, so a file caught mid-write
is never saved with a half-written record.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path


class GitError(RuntimeError):
    pass


class ResultsSync:
    def __init__(self, remote_url, branch, subdir, out_dir, owned_files, workdir,
                 author_name, author_email, token=None, interval_minutes=10, base_branch="main"):
        """
        remote_url:  repository URL (may contain a token; it is never printed)
        branch:      shared results branch, created from ``base_branch`` if missing
        subdir:      folder inside the repository that holds the results
        out_dir:     local folder the experiment writes to
        owned_files: file names (inside out_dir/subdir) this instance may write
        workdir:     where to keep the local clone of the results branch
        """
        self.remote_url = remote_url
        self.branch = branch
        self.base_branch = base_branch
        self.subdir = subdir
        self.out_dir = Path(out_dir)
        self.owned = sorted(set(owned_files))
        self.workdir = Path(workdir)
        self.author = (author_name, author_email)
        self.token = token
        self.interval = interval_minutes * 60
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.last_push = None

    # ---------------------------------------------------------------- git
    def _git(self, *args, cwd=None):
        result = subprocess.run(["git", *args], cwd=cwd or self.workdir,
                                capture_output=True, text=True)
        if result.returncode != 0:
            msg = result.stderr + result.stdout
            if self.token:
                msg = msg.replace(self.token, "***")
            raise GitError(f"git {args[0]} failed:\n{msg.strip()}")
        return result.stdout

    @property
    def results_dir(self):
        return self.workdir / self.subdir

    def connect(self):
        """Clone the results branch (creating it from the base branch if it doesn't exist)."""
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._git("clone", "-q", "--branch", self.branch, self.remote_url, str(self.workdir),
                      cwd=self.workdir.parent)
        except GitError as e:
            if "not found" not in str(e) and "Remote branch" not in str(e):
                raise
            self._git("clone", "-q", "--branch", self.base_branch, self.remote_url,
                      str(self.workdir), cwd=self.workdir.parent)
            self._git("checkout", "-q", "-b", self.branch)
        self._git("config", "user.name", self.author[0])
        self._git("config", "user.email", self.author[1])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ---------------------------------------------------------------- files
    def restore(self):
        """Copy this person's saved files back into out_dir. Returns the names restored."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        restored = []
        for name in self.owned:
            src = self.results_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / name)
                restored.append(name)
        return restored

    def _snapshot(self):
        """Copy owned files from out_dir into the clone (JSONL: complete lines only)."""
        for name in self.owned:
            src = self.out_dir / name
            if not src.exists():
                continue
            dst = self.results_dir / name
            if src.suffix == ".jsonl":
                data = src.read_bytes()
                dst.write_bytes(data[:data.rfind(b"\n") + 1])
            else:
                shutil.copy2(src, dst)

    def push(self, message, retries=5):
        """Snapshot, commit and push this person's files. Returns True if anything was pushed."""
        with self._lock:
            self._snapshot()
            paths = [str(Path(self.subdir) / n) for n in self.owned
                     if (self.results_dir / n).exists()]
            if not paths:
                return False
            self._git("add", "-f", "--", *paths)
            if not self._git("status", "--porcelain", "--", *paths).strip():
                return False
            self._git("commit", "-q", "-m", message)
            for attempt in range(retries):
                try:
                    self._git("push", "-q", "origin", f"HEAD:{self.branch}")
                    self.last_push = time.strftime("%H:%M:%S")
                    return True
                except GitError:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 + 3 * attempt)
                    try:   # someone else pushed first: put our commit on top of theirs
                        self._git("pull", "-q", "--rebase", "origin", self.branch)
                    except GitError:
                        pass   # the branch may not exist remotely yet; the next push creates it
        return False

    def pull(self):
        """Fetch everyone's latest results into the clone (read-only use, e.g. a team report)."""
        with self._lock:
            self._git("pull", "-q", "--rebase", "origin", self.branch)

    # ---------------------------------------------------------------- background
    def start(self):
        """Push every ``interval_minutes`` in a background thread until ``stop()``."""
        def loop():
            while not self._stop.wait(self.interval):
                try:
                    if self.push(f"{self.author[0]}: periodic save"):
                        print(f"\n[results saved to GitHub at {self.last_push}]")
                except Exception as e:   # never kill the experiment because of a sync error
                    print(f"\n[results sync failed, will retry: {e}]")
        self._stop.clear()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self, final_message=None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)
        if final_message:
            return self.push(final_message)
        return False
