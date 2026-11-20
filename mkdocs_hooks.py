r"""Minimal MkDocs hooks for TPUMS transforms without common syntax or plugins.

Handles only what lacks a shared Markdown syntax across GitHub, g3doc, and
MkDocs or an existing Python-Markdown extension:
1. Registering root `README.md` as virtual `index.md` (`on_files`).
2. Rewriting relative repository links outside `docs/` to GitHub blob/tree URLs
   and `../README.md` links to `index.md` (`_rewrite_links`).
3. Normalizing single-line `$$...$$` inline formulas (required by g3doc, where
   `$...$` renders literally) into `\(...\)` for `pymdownx.arithmatex`
   (`_normalize_inline_math`).
"""

from collections.abc import Mapping, Sequence
import os
import posixpath
import re
from typing import Any

FENCE_RE = re.compile(r"(?m)^([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\2[ \t]*$")
_UNMASK_FENCE_RE = re.compile(r"\x00FENCE_(\d+)\x00")
_INLINE_CODE_RE = re.compile(r"(`+)([^`]*?)\1")
_UNMASK_INLINE_RE = re.compile(r"\x00INLINE_(\d+)\x00")
_LINK_RE = re.compile(r"(?<!!)(\[(?:[^\[\]]|\[[^\]]*\])*\])\(([^)\s]+)\)")
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$\$([^\n$]+?)\$\$(?!\$)")


def _mask_fences(markdown: str) -> tuple[str, list[str]]:
  """Replaces fenced code blocks with indexed sentinel tokens."""
  fences: list[str] = []

  def _store(match: re.Match[str]) -> str:
    idx = len(fences)
    fences.append(match.group(0))
    return f"\x00FENCE_{idx}\x00"

  return FENCE_RE.sub(_store, markdown), fences


def _unmask_fences(markdown: str, fences: Sequence[str]) -> str:
  """Restores fenced code blocks."""
  return _UNMASK_FENCE_RE.sub(lambda m: fences[int(m.group(1))], markdown)


def _mask_inline_code(markdown: str) -> tuple[str, list[str]]:
  """Replaces inline backtick code spans with indexed sentinel tokens."""
  spans: list[str] = []

  def _store(match: re.Match[str]) -> str:
    idx = len(spans)
    spans.append(match.group(0))
    return f"\x00INLINE_{idx}\x00"

  return _INLINE_CODE_RE.sub(_store, markdown), spans


def _unmask_inline_code(markdown: str, spans: Sequence[str]) -> str:
  """Restores inline backtick code spans."""
  return _UNMASK_INLINE_RE.sub(lambda m: spans[int(m.group(1))], markdown)


def on_files(files: Any, config: Mapping[str, Any]) -> Any:
  """Registers root README.md as virtual index.md for MkDocs builds."""
  if files.get_file_from_path("index.md") is not None:
    return files
  docs_dir = os.path.abspath(config["docs_dir"])
  readme_path = os.path.join(os.path.dirname(docs_dir), "README.md")
  if not os.path.isfile(readme_path):
    raise FileNotFoundError(f"Missing root README.md at {readme_path}")
  file_cls = type(next(iter(files)))
  files.append(file_cls.generated(config, "index.md", abs_src_path=readme_path))
  return files


def _rewrite_links(
    markdown: str,
    *,
    src_uri: str,
    docs_dir: str,
    repo_root: str,
    repo_url: str,
) -> str:
  """Rewrites relative repository links for MkDocs pages and GitHub files."""
  page_docs_dir = os.path.dirname(os.path.join(docs_dir, src_uri))
  source_dir = repo_root if src_uri == "index.md" else page_docs_dir
  readme_abs = os.path.join(repo_root, "README.md")

  def _replace(match: re.Match[str]) -> str:
    label, raw_target = match.group(1), match.group(2)
    if raw_target.startswith(("http://", "https://", "mailto:", "#")):
      return match.group(0)
    path_part, sep, fragment = raw_target.partition("#")
    anchor_suffix = f"#{fragment}" if sep else ""
    had_trailing_slash = path_part.endswith("/")
    target_abs = os.path.abspath(
        os.path.normpath(os.path.join(source_dir, path_part))
    )

    if target_abs == readme_abs:
      rel_index = posixpath.relpath(
          os.path.join(docs_dir, "index.md").replace(os.sep, "/"),
          page_docs_dir.replace(os.sep, "/"),
      )
      return f"{label}({rel_index}{anchor_suffix})"

    if os.path.commonpath([docs_dir, target_abs]) == docs_dir:
      rel_doc = posixpath.relpath(
          target_abs.replace(os.sep, "/"),
          page_docs_dir.replace(os.sep, "/"),
      )
      return f"{label}({rel_doc}{anchor_suffix})"

    if os.path.commonpath([repo_root, target_abs]) == repo_root:
      rel_repo = posixpath.relpath(
          target_abs.replace(os.sep, "/"),
          repo_root.replace(os.sep, "/"),
      )
      if os.path.isdir(target_abs) or had_trailing_slash:
        slash = "/" if had_trailing_slash else ""
        return f"{label}({repo_url}/tree/main/{rel_repo}{slash}{anchor_suffix})"
      return f"{label}({repo_url}/blob/main/{rel_repo}{anchor_suffix})"

    return match.group(0)

  return _LINK_RE.sub(_replace, markdown)


def _normalize_inline_math(markdown: str) -> str:
  r"""Converts single-line `$$...$$` inline formulas to `\(...\)`."""
  return _INLINE_MATH_RE.sub(lambda m: f"\\({m.group(1).strip()}\\)", markdown)


def on_page_markdown(
    markdown: str,
    page: Any,
    config: Mapping[str, Any],
    files: Any,
) -> str:
  """Rewrites relative repository links and single-line inline math."""
  del files
  docs_dir = os.path.abspath(config["docs_dir"])
  repo_root = os.path.dirname(docs_dir)
  repo_url = config["repo_url"].rstrip("/")
  src_uri = page.file.src_uri

  masked, fences = _mask_fences(markdown)
  masked, inlines = _mask_inline_code(masked)
  masked = _rewrite_links(
      masked,
      src_uri=src_uri,
      docs_dir=docs_dir,
      repo_root=repo_root,
      repo_url=repo_url,
  )
  masked = _normalize_inline_math(masked)
  unmasked = _unmask_inline_code(masked, inlines)
  return _unmask_fences(unmasked, fences)
