"""
ingestion/connectors/git_connector.py
Clones / pulls Git repositories and ingests text files matching allowed extensions.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import git  # GitPython

from ingestion.connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".json"}
_MAX_FILE_SIZE_BYTES = 500_000  # 500 KB safety cap


class GitConnector(BaseConnector):
    """
    Clones each repo into a temp directory, walks the working tree,
    and yields one Document per matching text file.

    Args:
        repos:            List of repository URLs (HTTPS or SSH).
        branch:           Branch to check out (default: "main").
        file_extensions:  Set of file extensions to include.
        github_token:     Optional PAT for private HTTPS repos.
    """

    @property
    def name(self) -> str:
        return "git"

    def __init__(
        self,
        repos: List[str],
        branch: str = "main",
        file_extensions: Optional[List[str]] = None,
        github_token: Optional[str] = None,
    ) -> None:
        self.repos = repos
        self.branch = branch
        self.file_extensions = set(file_extensions or _DEFAULT_EXTENSIONS)
        self.github_token = github_token

    # ── helpers ─────────────────────────────────────────────────────────

    def _inject_token(self, url: str) -> str:
        """Inject PAT into HTTPS URL for private repos."""
        if self.github_token and url.startswith("https://"):
            url = url.replace("https://", f"https://{self.github_token}@")
        return url

    def _clone_repo(self, url: str, dest: Path) -> Optional[git.Repo]:
        clone_url = self._inject_token(url)
        try:
            repo = git.Repo.clone_from(clone_url, str(dest), branch=self.branch, depth=1)
            logger.info("Cloned %s → %s", url, dest)
            return repo
        except git.GitCommandError as exc:
            logger.error("Failed to clone %s: %s", url, exc)
            return None

    def _repo_name(self, url: str) -> str:
        return url.rstrip("/").split("/")[-1].replace(".git", "")

    def _walk_repo(self, repo_root: Path, repo_url: str) -> Iterator[Document]:
        repo_name = self._repo_name(repo_url)

        for file_path in repo_root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.file_extensions:
                continue
            if file_path.stat().st_size > _MAX_FILE_SIZE_BYTES:
                logger.debug("Skipping large file: %s", file_path)
                continue

            # Skip hidden dirs (.git, .github, __pycache__, etc.)
            rel = file_path.relative_to(repo_root)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Cannot read %s: %s", file_path, exc)
                continue

            if not content.strip():
                continue

            rel_str = str(rel)
            source_id = hashlib.sha256(f"{repo_url}/{rel_str}".encode()).hexdigest()[:16]

            yield Document(
                source="git",
                source_id=source_id,
                title=f"{repo_name}/{rel_str}",
                content=content,
                url=f"{repo_url}/blob/{self.branch}/{rel_str}",
                file_path=rel_str,
                metadata={
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "branch": self.branch,
                    "file_extension": file_path.suffix,
                    "char_count": len(content),
                },
            )

    # ── public API ──────────────────────────────────────────────────────

    def fetch(self) -> Iterator[Document]:
        for repo_url in self.repos:
            tmp_dir = tempfile.mkdtemp(prefix="ragflow_git_")
            tmp_path = Path(tmp_dir)
            try:
                repo = self._clone_repo(repo_url, tmp_path)
                if repo is None:
                    continue
                yield from self._walk_repo(tmp_path, repo_url)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.debug("Cleaned up temp dir %s", tmp_dir)
