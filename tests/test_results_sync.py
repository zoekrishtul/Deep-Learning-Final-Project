"""Tests for src/results_sync.py against a local bare Git repository (no network)."""

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from src.results_sync import ResultsSync

SUB = "final_results/2D_full"


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


class TestResultsSync(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # A bare "GitHub" with a main branch.
        self.remote = self.tmp / "remote.git"
        git("init", "-q", "--bare", "-b", "main", str(self.remote), cwd=self.tmp)
        seed = self.tmp / "seed"
        git("clone", "-q", str(self.remote), str(seed), cwd=self.tmp)
        (seed / "README.md").write_text("repo\n")
        (seed / ".gitignore").write_text("*.pt\n")
        git("add", ".", cwd=seed)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=seed)
        git("push", "-q", "origin", "main", cwd=seed)

    def make(self, person, owned):
        out = self.tmp / f"out_{person}"
        out.mkdir(exist_ok=True)
        s = ResultsSync(str(self.remote), "final-results", SUB, out, owned,
                        self.tmp / f"clone_{person}", person, f"{person}@x", interval_minutes=10)
        return s.connect(), out

    def remote_files(self):
        check = self.tmp / "check"
        if check.exists():
            subprocess.run(["rm", "-rf", str(check)])
        git("clone", "-q", "--branch", "final-results", str(self.remote), str(check), cwd=self.tmp)
        return check / SUB

    def test_creates_branch_and_pushes_only_owned_complete_lines(self):
        s, out = self.make("gaurav", ["direct_2D.jsonl", "direct_2D.jsonl.config.json"])
        (out / "direct_2D.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "half')  # mid-write
        (out / "direct_2D.jsonl.config.json").write_text("{}")
        (out / "someone_else.jsonl").write_text("x\n")                            # not owned
        self.assertTrue(s.push("save"))
        files = self.remote_files()
        self.assertEqual((files / "direct_2D.jsonl").read_text(), '{"a": 1}\n{"a": 2}\n')
        self.assertTrue((files / "direct_2D.jsonl.config.json").exists())
        self.assertFalse((files / "someone_else.jsonl").exists())
        self.assertFalse(s.push("nothing changed"))

    def test_restore_after_restart_resumes_from_saved_lines(self):
        s, out = self.make("gaurav", ["direct_2D.jsonl"])
        (out / "direct_2D.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
        s.push("save")
        # New Kaggle session: empty working folder, fresh clone.
        (out / "direct_2D.jsonl").unlink()
        s2, _ = self.make("gaurav", ["direct_2D.jsonl"])
        self.assertEqual(s2.restore(), ["direct_2D.jsonl"])
        self.assertEqual((out / "direct_2D.jsonl").read_text(), '{"a": 1}\n{"a": 2}\n')

    def test_three_people_push_at_once_without_overwriting(self):
        people = {"gaurav": "direct_2D.jsonl", "yujye": "direct+clip_2D.jsonl",
                  "zoe": "direct+scaffold_2D.jsonl"}
        syncs = {}
        for p, f in people.items():
            syncs[p] = self.make(p, [f])
            (syncs[p][1] / f).write_text(f'{{"by": "{p}"}}\n')
        errors = []

        def go(p):
            try:
                for i in range(3):
                    out = syncs[p][1] / people[p]
                    out.write_text(out.read_text() + f'{{"i": {i}}}\n')
                    syncs[p][0].push(f"{p} {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=go, args=(p,)) for p in people]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        files = self.remote_files()
        for p, f in people.items():
            lines = (files / f).read_text().splitlines()
            self.assertEqual(len(lines), 4, f)          # 1 initial + 3 appended, nothing lost
            self.assertIn(p, lines[0])

    def test_ignored_files_are_saved(self):
        s, out = self.make("yujye", ["grounding.pt"])
        (out / "grounding.pt").write_bytes(b"\x00\x01")
        self.assertTrue(s.push("checkpoint"))
        self.assertTrue((self.remote_files() / "grounding.pt").exists())

    def test_token_is_masked_in_errors(self):
        s = ResultsSync("https://x-access-token:SECRET123@invalid.invalid/r.git", "b", SUB,
                        self.tmp / "o", [], self.tmp / "c", "n", "e", token="SECRET123")
        with self.assertRaises(Exception) as ctx:
            s.connect()
        self.assertNotIn("SECRET123", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
