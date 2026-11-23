"""Unit tests for the TPUMS MkDocs documentation site and build hooks."""

import dataclasses
import os
import re
from typing import Any

from absl import flags
from absl.testing import absltest
import mkdocs_hooks
import yaml


class _MkDocsYamlLoader(yaml.SafeLoader):
  """Safe YAML loader supporting MkDocs python tag references as strings."""


_MkDocsYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: suffix,
)

_PUBLIC_DOCS = (
    "README.md",
    "docs/BENCHMARKS.md",
    "docs/GKE.md",
    "docs/AUTOMATION.md",
)


@dataclasses.dataclass(frozen=True)
class _FakeFile:
  """Minimal MkDocs File double."""

  src_uri: str
  abs_src_path: str | None = None

  @classmethod
  def generated(
      cls,
      config: Any,
      src_uri: str,
      *,
      abs_src_path: str,
  ) -> "_FakeFile":
    """Constructs a virtual _FakeFile backed by abs_src_path."""
    del config
    return cls(src_uri=src_uri, abs_src_path=abs_src_path)


class _FakeFiles:
  """Minimal MkDocs Files collection double for testing on_files."""

  def __init__(self, initial: list[_FakeFile]):
    self._files = list(initial)

  def __iter__(self):
    return iter(self._files)

  def __len__(self) -> int:
    return len(self._files)

  def get_file_from_path(self, path: str) -> _FakeFile | None:
    """Returns the _FakeFile matching path, or None if absent."""
    for f in self._files:
      if f.src_uri == path:
        return f
    return None

  def append(self, file_obj: _FakeFile) -> None:
    """Appends file_obj to the collection."""
    self._files.append(file_obj)


@dataclasses.dataclass(frozen=True)
class _FakePage:
  """Minimal MkDocs Page double."""

  file: _FakeFile


def _make_page(src_uri: str) -> _FakePage:
  return _FakePage(file=_FakeFile(src_uri=src_uri))


def _strip_fences(markdown: str) -> str:
  """Strips fenced code blocks while preserving headings and prose."""
  return mkdocs_hooks.FENCE_RE.sub("", markdown)


def _strip_code(markdown: str) -> str:
  """Strips fenced code blocks and inline code spans for link scans."""
  return re.sub(r"(`+)([^`]*?)\1", "", _strip_fences(markdown))


def _github_heading_slug(heading_text: str) -> str:
  """Computes a GitHub-compatible heading anchor slug."""
  cleaned = re.sub(r"<[^>]+>", "", heading_text)
  cleaned = cleaned.replace("`", "")
  cleaned = cleaned.strip().lower()
  cleaned = re.sub(r"[^\w\s-]", "", cleaned)
  return re.sub(r"\s", "-", cleaned)


def _collect_anchors(markdown: str) -> set[str]:
  """Collects explicit `<a id="...">` anchors and heading slugs in `markdown`."""
  anchors = set(re.findall(r'<a\s+id="([^"]+)"\s*>', markdown))
  stripped = _strip_fences(markdown)
  for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", stripped):
    anchors.add(_github_heading_slug(match.group(1)))
  return anchors


def _iter_nav_targets(nav_entries: list[Any]) -> list[str]:
  """Extracts Markdown relative paths from a MkDocs nav structure."""
  targets: list[str] = []
  for item in nav_entries:
    if isinstance(item, str):
      targets.append(item)
    elif isinstance(item, dict):
      for val in item.values():
        if isinstance(val, str):
          targets.append(val)
        elif isinstance(val, list):
          targets.extend(_iter_nav_targets(val))
  return targets


class DocsSiteTest(absltest.TestCase):
  """Tests Markdown link integrity, MkDocs hooks, and site configuration."""

  def setUp(self):
    super().setUp()
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()
    self.repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    self.docs_dir = os.path.join(self.repo_root, "docs")
    self.repo_url = (
        "https://github.com/AI-Hypercomputer/accelerator-microbenchmarks"
    )
    self.hook_config: dict[str, Any] = {
        "docs_dir": self.docs_dir,
        "repo_url": self.repo_url,
    }

  def _read_doc(self, rel_path: str) -> str:
    full_path = os.path.join(self.repo_root, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
      return f.read()

  def test_public_markdown_links_and_anchors_resolve(self):
    """Verifies every relative Markdown link and anchor in docs resolves."""
    doc_contents = {p: self._read_doc(p) for p in _PUBLIC_DOCS}
    doc_anchors = {
        os.path.abspath(os.path.join(self.repo_root, p)): _collect_anchors(text)
        for p, text in doc_contents.items()
    }

    link_re = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)\s]+)\)")
    for rel_path, content in doc_contents.items():
      abs_source = os.path.abspath(os.path.join(self.repo_root, rel_path))
      source_dir = os.path.dirname(abs_source)
      scannable = _strip_code(content)
      for match in link_re.finditer(scannable):
        raw_target = match.group(1)
        if raw_target.startswith(("http://", "https://", "mailto:")):
          continue
        path_part, sep, fragment = raw_target.partition("#")
        if not path_part:
          self.assertIn(
              fragment,
              doc_anchors[abs_source],
              f"Missing anchor #{fragment} in {rel_path}",
          )
          continue
        target_abs = os.path.abspath(os.path.join(source_dir, path_part))
        self.assertTrue(
            os.path.exists(target_abs),
            f"Broken relative link '{raw_target}' in {rel_path} ->"
            f" {target_abs}",
        )
        if sep and target_abs.endswith(".md"):
          if target_abs not in doc_anchors:
            with open(target_abs, "r", encoding="utf-8") as f:
              doc_anchors[target_abs] = _collect_anchors(f.read())
          self.assertIn(
              fragment,
              doc_anchors[target_abs],
              f"Missing target anchor #{fragment} for '{raw_target}' in"
              f" {rel_path}",
          )

  def test_public_docs_omit_manual_table_of_contents(self):
    """Verifies public docs omit manual Table of Contents sections."""
    for rel_path in _PUBLIC_DOCS:
      scannable = _strip_fences(self._read_doc(rel_path))
      self.assertNotIn(
          "## Table of Contents",
          scannable,
          f"Unexpected manual Table of Contents in {rel_path}",
      )

  def test_public_docs_use_four_space_list_indentation(self):
    """Verifies nested list items in public docs use 4-space steps."""
    list_item_re = re.compile(r"^([ ]*)(?:[-*+][ ]+|\d+\.[ ]+)")
    for rel_path in _PUBLIC_DOCS:
      stripped = mkdocs_hooks.FENCE_RE.sub(
          lambda m: "\n" * m.group(0).count("\n"),
          self._read_doc(rel_path),
      )
      for line_no, line in enumerate(stripped.splitlines(), 1):
        match = list_item_re.match(line)
        if match:
          indent = len(match.group(1))
          self.assertEqual(
              indent % 4,
              0,
              f"{rel_path}:{line_no} has {indent}-space list indent (expected"
              f" multiple of 4): {line!r}",
          )

  def test_index_page_populates_from_readme_and_rewrites_links(self):
    """Verifies index.md transforms root README.md and rewrites repo links."""
    tmpdir = self.create_tempdir().full_path
    docs_dir = os.path.join(tmpdir, "docs")
    os.makedirs(os.path.join(tmpdir, "configs", "tpu7x", "2x2x1"))
    os.makedirs(os.path.join(tmpdir, "templates"))
    os.makedirs(docs_dir)
    with open(
        os.path.join(tmpdir, "templates", "gke_job_template.yaml"),
        "w",
        encoding="utf-8",
    ) as f:
      f.write("kind: Job\n")
    readme_src = self._read_doc("README.md") + (
        "\n\nExtra links: [Template](templates/gke_job_template.yaml) and"
        " [Dir](configs/tpu7x/2x2x1/) and inline"
        " `[CodeSpan](docs/BENCHMARKS.md)`.\n\n```markdown\n[FenceLink](configs/tpu7x/2x2x1/)\n```\n"
    )

    rendered = mkdocs_hooks.on_page_markdown(
        readme_src,
        page=_make_page("index.md"),
        config={"docs_dir": docs_dir, "repo_url": self.repo_url},
        files=None,
    )

    self.assertIn("# TPUMS: TPU Microbenchmark Suite", rendered)
    self.assertIn("(BENCHMARKS.md#21-matrix-multiplication-gemm)", rendered)
    self.assertIn("(GKE.md)", rendered)
    self.assertIn("(AUTOMATION.md)", rendered)
    self.assertIn(
        f"({self.repo_url}/blob/main/templates/gke_job_template.yaml)",
        rendered,
    )
    self.assertIn(
        f"({self.repo_url}/tree/main/configs/tpu7x/2x2x1/)",
        rendered,
    )
    self.assertIn("`[CodeSpan](docs/BENCHMARKS.md)`", rendered)
    self.assertIn("[FenceLink](configs/tpu7x/2x2x1/)", rendered)

  def test_subpage_relative_links_rewrite_to_index_and_github(self):
    """Verifies docs/*.md pages rewrite relative links to index.md and GitHub."""
    benchmarks_md = self._read_doc("docs/BENCHMARKS.md")
    rendered_bench = mkdocs_hooks.on_page_markdown(
        benchmarks_md,
        page=_make_page("BENCHMARKS.md"),
        config=self.hook_config,
        files=None,
    )
    self.assertIn("(index.md)", rendered_bench)
    self.assertIn("(index.md#primary-metrics)", rendered_bench)
    self.assertIn(
        f"({self.repo_url}/blob/main/src/accelerator_microbenchmarks/core/dtypes.py)",
        rendered_bench,
    )

    gke_md = self._read_doc("docs/GKE.md")
    rendered_gke = mkdocs_hooks.on_page_markdown(
        gke_md,
        page=_make_page("GKE.md"),
        config=self.hook_config,
        files=None,
    )
    self.assertIn("(AUTOMATION.md)", rendered_gke)
    self.assertIn(
        f"({self.repo_url}/tree/main/configs/tpu7x/2x2x1/)",
        rendered_gke,
    )
    self.assertIn(
        f"({self.repo_url}/blob/main/templates/gke_job_template.yaml)",
        rendered_gke,
    )

  def test_on_page_markdown_preserves_boundary_links(self):
    """Verifies boundary #fragment, escaping, and dir/slash link rewrites."""
    tmpdir = self.create_tempdir().full_path
    repo_root = os.path.join(tmpdir, "repo")
    docs_dir = os.path.join(repo_root, "docs")
    os.makedirs(docs_dir)
    os.makedirs(os.path.join(repo_root, "configs"))

    sample = (
        "Same-page [Jump](#local-section), escaping"
        " [Outside](../../outside_repo/doc.md), existing dir without slash"
        " [Configs](../configs), and uncreated dir with slash"
        " [FutureDir](../future_dir/).\n"
    )
    rendered = mkdocs_hooks.on_page_markdown(
        sample,
        page=_make_page("BENCHMARKS.md"),
        config={"docs_dir": docs_dir, "repo_url": self.repo_url},
        files=None,
    )
    self.assertIn("[Jump](#local-section)", rendered)
    self.assertIn("[Outside](../../outside_repo/doc.md)", rendered)
    self.assertIn(f"[Configs]({self.repo_url}/tree/main/configs)", rendered)
    self.assertIn(
        f"[FutureDir]({self.repo_url}/tree/main/future_dir/)", rendered
    )

  def test_inline_math_normalizes_without_corrupting_latex_or_code(self):
    r"""Verifies inline $$...$$ formulas convert to \(...\) cleanly."""
    sample = (
        "Formula $$C = \\alpha (A \\times B) + \\beta C$$ in prose.\n\n| Op |"
        " Formula |\n| :--- | :--- |\n| AR | $$2 \\times S \\times"
        " \\frac{P}{R}$$ and $$C_{\\text{out}}$$ |\n\nInline code `$$x_1 +"
        " x_2$$` stays untouched.\n\n$$\nE = mc^2\n$$\n\n```tex\n$$a \\times"
        " b$$\n```\n"
    )
    rendered = mkdocs_hooks.on_page_markdown(
        sample,
        page=_make_page("BENCHMARKS.md"),
        config=self.hook_config,
        files=None,
    )
    self.assertIn(r"\(C = \alpha (A \times B) + \beta C\)", rendered)
    self.assertIn(
        r"| AR | \(2 \times S \times \frac{P}{R}\) and \(C_{\text{out}}\) |",
        rendered,
    )
    self.assertIn("`$$x_1 + x_2$$`", rendered)
    self.assertIn("$$\nE = mc^2\n$$", rendered)
    self.assertIn("```tex\n$$a \\times b$$\n```", rendered)

  def test_mkdocs_config_and_docs_requirements(self):
    """Verifies mkdocs.yml structural invariants and pinned requirements."""
    self.assertFalse(
        os.path.exists(os.path.join(self.docs_dir, "index.md")),
        "docs/index.md should not exist on disk (registered via on_files)",
    )
    mkdocs_path = os.path.join(self.repo_root, "mkdocs.yml")
    with open(mkdocs_path, "r", encoding="utf-8") as f:
      cfg = yaml.load(f, Loader=_MkDocsYamlLoader)
    self.assertIsInstance(cfg, dict)
    self.assertTrue(cfg.get("strict"))
    self.assertEqual(cfg.get("validation", {}).get("anchors"), "warn")
    self.assertIn("mkdocs_hooks.py", cfg.get("hooks", []))
    self.assertIn("requirements.txt", cfg.get("exclude_docs", ""))
    self.assertIn("github-callouts", cfg.get("markdown_extensions", []))

    nav_targets = _iter_nav_targets(cfg.get("nav", []))
    self.assertNotEmpty(nav_targets)
    for target in nav_targets:
      if target == "index.md":
        self.assertTrue(
            os.path.isfile(os.path.join(self.repo_root, "README.md"))
        )
      else:
        self.assertTrue(
            os.path.isfile(os.path.join(self.docs_dir, target)),
            f"Missing MkDocs nav target: {target}",
        )

    extra_js = cfg.get("extra_javascript", [])
    self.assertNotEmpty(extra_js)
    for entry in extra_js:
      script_ref = entry["path"] if isinstance(entry, dict) else entry
      if script_ref.startswith(("http://", "https://")):
        self.assertRegex(script_ref, r"@\d+\.\d+\.\d+/")
      else:
        self.assertTrue(
            os.path.isfile(os.path.join(self.docs_dir, script_ref)),
            f"Missing local extra_javascript asset: {script_ref}",
        )

    for css_ref in cfg.get("extra_css", []):
      if not css_ref.startswith(("http://", "https://")):
        self.assertTrue(
            os.path.isfile(os.path.join(self.docs_dir, css_ref)),
            f"Missing local extra_css asset: {css_ref}",
        )

    req_path = os.path.join(self.docs_dir, "requirements.txt")
    self.assertTrue(os.path.isfile(req_path))
    with open(req_path, "r", encoding="utf-8") as f:
      docs_deps = [
          line.strip()
          for line in f
          if line.strip() and not line.lstrip().startswith("#")
      ]
    self.assertNotEmpty(docs_deps)
    pinned_re = re.compile(r"^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.-]*$")
    for dep in docs_deps:
      self.assertRegex(dep, pinned_re)
    self.assertTrue(
        any(dep.startswith("markdown-callouts==") for dep in docs_deps)
    )

  def test_github_pages_workflow_schema(self):
    """Verifies .github/workflows/docs.yml security and least-privilege rules."""
    workflow_path = os.path.join(
        self.repo_root, ".github", "workflows", "docs.yml"
    )
    self.assertTrue(os.path.isfile(workflow_path))
    with open(workflow_path, "r", encoding="utf-8") as f:
      workflow = yaml.safe_load(f.read())
    self.assertIsInstance(workflow, dict)
    self.assertNotIn(True, workflow)
    self.assertIn("on", workflow)
    triggers = workflow["on"]
    self.assertIn("push", triggers)
    self.assertNotEmpty(triggers["push"].get("paths", []))
    self.assertIn("pull_request", triggers)
    self.assertIsNone(triggers["pull_request"])
    self.assertEqual(workflow.get("permissions"), {"contents": "read"})
    self.assertNotIn("concurrency", workflow)

    jobs = workflow["jobs"]
    build_job = jobs["build"]
    self.assertEqual(build_job.get("name"), "Build documentation")
    build_steps = build_job["steps"]
    deploy_job = jobs["deploy"]
    self.assertEqual(
        deploy_job.get("if"),
        "github.ref == 'refs/heads/main'",
    )
    self.assertEqual(deploy_job.get("needs"), "build")
    self.assertEqual(
        deploy_job.get("concurrency"),
        {"group": "pages", "cancel-in-progress": False},
    )
    self.assertEqual(
        deploy_job.get("permissions"),
        {"pages": "write", "id-token": "write"},
    )

    uses_values: list[str] = []
    for job in jobs.values():
      for step in job.get("steps", []):
        if "uses" in step:
          uses_values.append(step["uses"])
    self.assertNotEmpty(uses_values)
    sha_re = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
    for uses_ref in uses_values:
      self.assertRegex(uses_ref, sha_re)
      self.assertNotIn("configure-pages", uses_ref)

    checkout_steps = [
        s
        for s in build_steps
        if s.get("uses", "").startswith("actions/checkout@")
    ]
    self.assertLen(checkout_steps, 1)
    self.assertEqual(
        checkout_steps[0].get("with", {}).get("persist-credentials"), False
    )

    run_commands = "\n".join(s["run"] for s in build_steps if "run" in s)
    self.assertIn("pip install -r docs/requirements.txt", run_commands)
    self.assertIn("mkdocs build --strict", run_commands)

  def test_extra_css_hides_desktop_primary_sidebar(self):
    """Verifies extra.css is registered and hides primary sidebar on desktop."""
    mkdocs_path = os.path.join(self.repo_root, "mkdocs.yml")
    with open(mkdocs_path, "r", encoding="utf-8") as f:
      cfg = yaml.load(f, Loader=_MkDocsYamlLoader)
    self.assertEqual(cfg.get("extra_css"), ["stylesheets/extra.css"])

    css_path = os.path.join(self.docs_dir, "stylesheets", "extra.css")
    self.assertTrue(os.path.isfile(css_path))
    with open(css_path, "r", encoding="utf-8") as f:
      css = f.read()
    self.assertIn("@media screen and (min-width: 76.25em)", css)
    self.assertIn(".md-sidebar--primary", css)
    self.assertIn("display: none;", css)

  def test_on_files_registers_virtual_index_from_readme(self):
    """Verifies on_files registers root README.md as virtual index.md."""
    tmpdir = self.create_tempdir().full_path
    docs_dir = os.path.join(tmpdir, "docs")
    os.makedirs(docs_dir)
    readme_path = os.path.join(tmpdir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
      f.write("# TPUMS\n")

    files = _FakeFiles([_FakeFile(src_uri="BENCHMARKS.md")])
    returned = mkdocs_hooks.on_files(
        files, {"docs_dir": docs_dir, "repo_url": self.repo_url}
    )
    self.assertIs(returned, files)
    self.assertLen(files, 2)
    index_file = files.get_file_from_path("index.md")
    self.assertIsNotNone(index_file)
    self.assertEqual(index_file.abs_src_path, readme_path)

    # Calling on_files again when index.md already exists is idempotent.
    mkdocs_hooks.on_files(
        files, {"docs_dir": docs_dir, "repo_url": self.repo_url}
    )
    self.assertLen(files, 2)

    empty_tmp = self.create_tempdir().full_path
    empty_docs = os.path.join(empty_tmp, "docs")
    os.makedirs(empty_docs)
    missing_files = _FakeFiles([_FakeFile(src_uri="BENCHMARKS.md")])
    with self.assertRaises(FileNotFoundError):
      mkdocs_hooks.on_files(
          missing_files,
          {"docs_dir": empty_docs, "repo_url": self.repo_url},
      )

  def test_extra_css_configured_and_defines_experimental_badge(self):
    """Verifies mkdocs.yml enables md_in_html and styles experimental drawers."""
    mkdocs_path = os.path.join(self.repo_root, "mkdocs.yml")
    with open(mkdocs_path, "r", encoding="utf-8") as f:
      cfg = yaml.load(f, Loader=_MkDocsYamlLoader)
    self.assertEqual(cfg.get("extra_css"), ["stylesheets/extra.css"])
    self.assertIn("md_in_html", cfg.get("markdown_extensions", []))

    css_path = os.path.join(self.docs_dir, "stylesheets", "extra.css")
    self.assertTrue(os.path.isfile(css_path))
    with open(css_path, "r", encoding="utf-8") as f:
      css = f.read()
    self.assertIn(".md-typeset details.experimental", css)
    self.assertIn(".md-typeset details.experimental > summary > code", css)
    self.assertIn('[data-md-color-scheme="slate"]', css)

  def test_experimental_benchmarks_section_and_readme_pointer(self):
    """Verifies BENCHMARKS.md experimental drawers and README.md pointer."""
    readme_md = self._read_doc("README.md")
    self.assertIn("(docs/BENCHMARKS.md#3-experimental-benchmarks)", readme_md)
    self.assertGreaterEqual(
        readme_md.count("tpums benchmark list --include_experimental"),
        2,
    )

    benchmarks_md = self._read_doc("docs/BENCHMARKS.md")
    scannable = _strip_fences(benchmarks_md)
    self.assertIn("## 3. Experimental Benchmarks", scannable)
    self.assertIn("--include_experimental", scannable)
    self.assertEqual(
        scannable.count('<details class="experimental" markdown="1">'),
        9,
    )
    self.assertEqual(scannable.count("<code>[EXPERIMENTAL]</code>"), 9)
    for anchor_id in (
        "211-gemm-throttling-gemm_throttling-experimental",
        "212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental",
        "213-gemm-pulse-gemm_pulse-experimental",
        "214-quantized-matrix-multiplication-gemm_quantized-experimental",
        "251-reduce-scatter-reduce_scatter-experimental",
        "3-experimental-benchmarks",
    ):
      self.assertIn(f'<a id="{anchor_id}"></a>', benchmarks_md)

    config_links = re.findall(r"\]\((\.\./configs/[^)\s#]+)\)", scannable)
    self.assertIn("../configs/sample_configs/gemm_quantized.yaml", config_links)
    self.assertIn("../configs/sample_configs/rope.yaml", config_links)
    for broken_study_cfg in (
        "../configs/sample_configs/study_llm_200b/attention.yaml",
        "../configs/sample_configs/study_llm_200b/transformer_moe.yaml",
    ):
      self.assertNotIn(broken_study_cfg, config_links)
    for rel_cfg in config_links:
      cfg_abs = os.path.normpath(os.path.join(self.docs_dir, rel_cfg))
      self.assertTrue(
          os.path.isfile(cfg_abs),
          f"Missing sample config referenced in BENCHMARKS.md: {rel_cfg}",
      )


if __name__ == "__main__":
  absltest.main()
