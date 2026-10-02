"""Run the pytest verifiers against a vendor endpoint and write a Markdown report.

Each selected suite runs in its own pytest process (all suites in parallel,
each with its own pytest-xdist worker count) and writes a JUnit XML report.
The XML files are aggregated into a Markdown report with per-suite pass
rates, failures grouped by cause, and failures attributed to API features.
Features declared with --unsupported are reported as waived.

Usage:
    uv run python test_report.py \
        --base-url https://api.example.com/v1 \
        --model moonshotai/Kimi-K2.7-Code \
        --provider Tenstorrent \
        --workers 16 --unsupported vision structured_output

    # Rebuild the report from an existing run without calling the API again
    uv run python test_report.py --render-only reports/<run-dir>

The API key is read from KIMI_API_KEY and is never written to the report.
Exit code: 0 if acceptance passed, 1 if it failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

SUITES = {
    "params": "API parameter constraint pre-flight validation",
    "tool_call_json_schema": "Tool-call argument validation for walle-valid MFJS schemas",
    "k3_features": "K3 feature contracts: dynamic tools, response_format, tool_choice, thinking effort",
    "prompt_tokens": "Vendor-reported usage.prompt_tokens vs. expected constants",
}

# Suites that only apply to some models: suite -> pattern the model name must match.
K3_PATTERN = r"k3"
MODEL_SCOPED_SUITES = {"k3_features": K3_PATTERN}

FEATURES = {
    "structured_output": "Structured output (response_format, forced tool_choice)",
    "vision": "Vision / image input",
    "dynamic_tools": "Dynamic tools (messages[].tools)",
    "thinking": "Thinking / reasoning effort",
    "sampling_params": "Sampling parameter enforcement",
    "tool_calling": "Tool calling",
    "prompt_tokens": "Prompt token accounting",
    "other": "Other",
}

# (feature, suite pattern, test-id pattern, message pattern); first match wins.
FEATURE_RULES: list[tuple[str, str, str, str]] = [
    ("structured_output", "", "", r"does not implement: guided_decoding"),
    ("tool_calling", r"^tool_call_json_schema$", "", ""),
    ("sampling_params", r"^params$", "", ""),
    ("structured_output", "", r"response_format|json_object|json_schema|tool_choice_required", ""),
    ("vision", "", r"vision|image", ""),
    ("dynamic_tools", "", r"dynamic", ""),
    ("prompt_tokens", r"^prompt_tokens$", "", ""),
    ("thinking", "", r"thinking|effort", ""),
    ("tool_calling", "", r"tool", ""),
]

# Known failure causes, matched in order against the failure message.
# Anything unmatched is grouped by its normalised first line.
KNOWN_CAUSES: list[tuple[str, str]] = [
    (r"does not implement: guided_decoding",
     "Structured output (guided decoding) not implemented by the server"),
    (r"Internal server error",
     "HTTP 500 Internal server error"),
    (r"should be rejected, but request succeeded",
     "Immutable sampling parameter not enforced (expected HTTP 400)"),
    (r"tool_choice is \\?\"required\\?\" but tools is empty",
     "Dynamic tools (messages[].tools) ignored by the server"),
    (r"expected 400 .*got 200",
     "Invalid request accepted (expected HTTP 400)"),
    (r"missing field `content`",
     "Assistant message without `content` rejected (HTTP 400)"),
    (r"must be a valid JSON object string",
     "Malformed tool-call arguments in history rejected (HTTP 400)"),
    (r"`thinking\.type` must be",
     "`thinking` without `type` rejected (HTTP 400)"),
    (r"prompt_tokens outside accepted range|prompt_tokens=\d+, expected=\d+",
     "usage.prompt_tokens differs from expected value"),
    (r"did not include \w+ tool call; tool_calls=",
     "Tool call made with a wrong / corrupted function name"),
    (r"no tool_calls|did not include tool_calls",
     "No tool call returned (model answered in plain text or ran out of tokens)"),
    (r"arguments are not valid JSON",
     "Tool-call arguments are not valid JSON"),
    (r"arguments do not match schema",
     "Tool-call arguments violate the JSON schema"),
    (r"reasoning_effort should (be ignored|take effect)",
     "reasoning_effort has no effect on reasoning length"),
    (r"in effort report",
     "Thinking effort not reported in the response"),
    (r"KeyError: 'usage'",
     "Response is missing the `usage` block"),
]

STATUS_ICON = {"PASS": "✅", "FAIL": "❌", "NA": "🟨", "NOT GRADED": "⚪"}


@dataclass
class TestResult:
    suite: str
    classname: str
    name: str
    outcome: str  # passed | failed | error | skipped
    message: str = ""
    details: str = ""  # full pytest failure text (traceback + assertion)
    feature: str = ""
    waived: bool = False

    @property
    def is_failure(self) -> bool:
        return self.outcome in ("failed", "error")


@dataclass
class SuiteResult:
    name: str
    graded: bool = True
    tests: list[TestResult] = field(default_factory=list)
    duration: float = 0.0
    collection_error: str = ""

    def count(self, outcome: str) -> int:
        return sum(t.outcome == outcome for t in self.tests)

    @property
    def executed(self) -> int:
        return self.count("passed") + self.count("failed") + self.count("error")

    @property
    def failures(self) -> list[TestResult]:
        return [t for t in self.tests if t.is_failure]

    @property
    def waived(self) -> int:
        return sum(t.waived for t in self.failures)

    @property
    def pass_rate(self) -> float | None:
        return ratio(self.count("passed"), self.executed)

    @property
    def pass_rate_excl_unsupported(self) -> float | None:
        return ratio(self.count("passed"), self.executed - self.waived)

    @property
    def status(self) -> str:
        if not self.graded:
            return "NOT GRADED"
        if self.collection_error or any(not t.waived for t in self.failures):
            return "FAIL"
        return "PASS" if self.executed else "NA"


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.1f}%"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def classify(message: str) -> str:
    for pattern, cause in KNOWN_CAUSES:
        if re.search(pattern, message):
            return cause
    first = message.strip().splitlines()[0] if message.strip() else "No failure message"
    first = re.sub(r"chatcmpl-[\w-]+", "<id>", first)
    return first[:160] + ("…" if len(first) > 160 else "")


@lru_cache(maxsize=1)
def prompt_token_case_features() -> dict[str, str]:
    """Map prompt_tokens case ids to the feature their request exercises, if any."""
    features = {}
    for path in (REPO_ROOT / "testdata" / "prompt_token_cases").glob("*cases.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                case = json.loads(line)
            except json.JSONDecodeError:  # e.g. an un-pulled Git LFS pointer
                continue
            request = case.get("request", {})
            tool_choice = request.get("tool_choice")
            if request.get("response_format") or tool_choice == "required" or isinstance(tool_choice, dict):
                features[case["id"]] = "structured_output"
    return features


def attribute_feature(test: TestResult) -> str:
    test_id = f"{test.classname}::{test.name}"
    if test.suite == "prompt_tokens":
        case_id = test.name.partition("[")[2].rstrip("]")
        if case_id in prompt_token_case_features():
            return prompt_token_case_features()[case_id]
    for feature, suite_re, name_re, msg_re in FEATURE_RULES:
        if suite_re and not re.search(suite_re, test.suite):
            continue
        if name_re and not re.search(name_re, test_id, re.IGNORECASE):
            continue
        if msg_re and not re.search(msg_re, test.message):
            continue
        return feature
    return "other"


def parse_junit(path: Path, name: str) -> SuiteResult:
    suite = SuiteResult(name=name)
    if not path.exists():
        suite.collection_error = "No JUnit report produced (pytest did not run)"
        return suite
    root = ET.parse(path).getroot()
    for ts in root.iter("testsuite"):
        suite.duration = max(suite.duration, float(ts.get("time", 0) or 0))
        for tc in ts.iter("testcase"):
            outcome, message, details = "passed", "", ""
            for tag in ("failure", "error", "skipped"):
                node = tc.find(tag)
                if node is not None:
                    outcome = {"failure": "failed"}.get(tag, tag)
                    details = node.text or ""
                    message = node.get("message") or details
                    break
            # Collection errors show up as an <error> testcase with an empty classname.
            if outcome == "error" and not tc.get("classname"):
                lines = message.strip().splitlines()
                suite.collection_error = lines[-1] if lines else "collection error"
                continue
            suite.tests.append(TestResult(name, tc.get("classname", ""), tc.get("name", "?"),
                                          outcome, message, details))
    return suite


def parse_workers(values: list[str], suites: list[str]) -> dict[str, int]:
    """Parse `--workers 16` and/or `--workers params=4 prompt_tokens=8`."""
    default = 1
    overrides: dict[str, int] = {}
    for value in values:
        if "=" in value:
            suite, _, n = value.partition("=")
            if suite not in SUITES:
                raise SystemExit(f"--workers: unknown suite {suite!r}")
            overrides[suite] = int(n)
        else:
            default = int(value)
    return {s: overrides.get(s, default) for s in suites}


# ---------------------------------------------------------------------------
# Live progress
# ---------------------------------------------------------------------------

# Seconds between [progress] lines while suites run (0 disables).
PROGRESS_INTERVAL = float(os.environ.get("VERIFIER_PROGRESS_INTERVAL", "60"))
# A pytest -q progress line: one character per finished case (R = rerun), then [ NN%].
_PYTEST_PROGRESS = re.compile(r"^([.FEsxXR]+)\s*(?:\[\s*(\d+)%\])?$")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def suite_progress(suite: str, log_path: Path) -> str:
    """One-line progress for a running suite, read from its log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "starting"
    if suite == "verify":
        runs = re.findall(r"=== verify\.py run (\d+)/(\d+)", text)
        bars = re.findall(r"Processing:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)", text)
        if not bars:
            return "starting"
        done, total = int(bars[-1][0]), int(bars[-1][1])
        run, runs_total = (int(runs[-1][0]), int(runs[-1][1])) if runs else (1, 1)
        pct = 100 * ((run - 1) * total + done) // (runs_total * total) if total else 0
        where = f"pass {run}/{runs_total}, " if runs_total > 1 else ""
        return f"{pct}% done ({where}{done}/{total} cases)"
    done = failed = 0
    pct = None
    for line in text.splitlines():
        if line.startswith("=="):
            break  # FAILURES / short summary: progress lines all come before
        m = _PYTEST_PROGRESS.match(line.strip())
        if not m:
            continue
        done += sum(m.group(1).count(c) for c in ".FEsxX")
        failed += m.group(1).count("F") + m.group(1).count("E")
        if m.group(2):
            pct = int(m.group(2))
    if not done:
        return "0% done (collecting / starting workers)"
    return (f"{pct}% done" if pct is not None else "running") + f" ({done} cases, {failed} failed so far)"


def wait_with_progress(procs: dict, out_dir: Path, on_done) -> None:
    """Wait for {suite: (popen, ...)} to finish, printing [progress] lines meanwhile.

    on_done(suite, exit_code, elapsed_seconds) is called as each one finishes.
    """
    import time

    start = time.monotonic()
    next_report = start + PROGRESS_INTERVAL
    pending = dict(procs)
    while pending:
        for suite, entry in list(pending.items()):
            code = entry[0].poll()
            if code is not None:
                del pending[suite]
                on_done(suite, code, time.monotonic() - start)
        if pending and PROGRESS_INTERVAL > 0 and time.monotonic() >= next_report:
            elapsed = format_duration(time.monotonic() - start)
            for suite in pending:
                print(f"[progress] {suite}: {suite_progress(suite, out_dir / f'{suite}.log')} · {elapsed} elapsed",
                      flush=True)
            next_report += PROGRESS_INTERVAL
        if pending:
            time.sleep(1)


def run_suites(args: argparse.Namespace, meta: dict, out_dir: Path, workers: dict[str, int]) -> dict[str, dict]:
    common = [
        "--base-url", args.base_url,
        "--smoke-model", args.model,
        "--think-mode", args.think_mode,
        "-p", "no:cacheprovider",
        "-q",
    ]
    extra = {
        "params": [],
        "tool_call_json_schema": [
            "--thinking",
            "--tool-choice", args.tool_choice,
            "--reruns", str(args.reruns), "--reruns-delay", "2",
            f"--tool-json-report={out_dir / 'tool-call-schema-report.json'}",
        ],
        "k3_features": [],
        "prompt_tokens": [
            "--prompt-tokens-reference", meta["prompt_tokens_reference"],
            *(["--strip-reasoning-effort"] if meta["strip_reasoning_effort"] else []),
        ],
    }
    # The pytest options above are explicit; do not let the same env vars override them.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROMPT_TOKENS_REFERENCE", "STRIP_REASONING_EFFORT")}
    procs = {}
    for suite in args.suites:
        xdist = ["-n", str(workers[suite])] if workers[suite] > 1 else []
        cmd = [
            sys.executable, "-m", "pytest", f"tests/{suite}", *common, *xdist, *extra[suite],
            f"--junitxml={out_dir / f'{suite}.xml'}",
        ]
        log = (out_dir / f"{suite}.log").open("w")
        print(f"[run] {suite} (workers={workers[suite]}): {shlex.join(cmd[2:])}", flush=True)
        procs[suite] = (subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), cmd, log)

    runs = {}

    def on_done(suite: str, code: int, elapsed: float) -> None:
        _proc, cmd, log = procs[suite]
        log.close()
        print(f"[done] {suite}: exit code {code} after {format_duration(elapsed)}", flush=True)
        runs[suite] = {"exit_code": code, "workers": workers[suite], "wall_seconds": round(elapsed, 1),
                       "command": shlex.join(["pytest", *cmd[3:]])}

    wait_with_progress(procs, out_dir, on_done)
    return {suite: runs[suite] for suite in procs}


def overall_status(graded: list[SuiteResult]) -> str:
    if any(s.status == "FAIL" for s in graded):
        return "FAIL"
    return "PASS" if any(s.status == "PASS" for s in graded) else "NA"


def summarize(meta: dict, suites: list[SuiteResult]) -> dict:
    graded = [s for s in suites if s.graded]
    failures = [t for s in graded for t in s.failures]
    passed = sum(s.count("passed") for s in graded)
    executed = sum(s.executed for s in graded)
    waived = sum(t.waived for t in failures)
    by_feature: dict[str, list[TestResult]] = {}
    for t in failures:
        by_feature.setdefault(t.feature, []).append(t)
    return {
        "status": overall_status(graded),
        "model": meta["model"],
        "provider": meta["provider"],
        "report_id": meta["report_id"],
        "passed": passed,
        "executed": executed,
        "failed": len(failures),
        "waived": waived,
        "pass_rate": ratio(passed, executed),
        "pass_rate_excl_unsupported": ratio(passed, executed - waived),
        "unsupported_features": meta["unsupported"],
        "unsupported_share_of_failures": ratio(waived, len(failures)),
        "failures_by_feature": {
            f: {"failed": len(ts), "share": len(ts) / len(failures), "unsupported": f in meta["unsupported"]}
            for f, ts in sorted(by_feature.items(), key=lambda kv: -len(kv[1]))
        },
        "suites": {
            s.name: {
                "graded": s.graded,
                "status": s.status,
                "passed": s.count("passed"),
                "failed": s.count("failed"),
                "errors": s.count("error"),
                "skipped": s.count("skipped"),
                "waived": s.waived,
                "pass_rate": s.pass_rate,
                "pass_rate_excl_unsupported": s.pass_rate_excl_unsupported,
                "collection_error": s.collection_error or None,
            }
            for s in suites
        },
    }


def render(meta: dict, suites: list[SuiteResult], summary: dict) -> str:
    graded = [s for s in suites if s.graded]
    ungraded = [s for s in suites if not s.graded]
    unsupported = meta["unsupported"]
    overall = summary["status"]
    n_fail, n_waived = summary["failed"], summary["waived"]

    title = f"{meta['model']} on {meta['provider']}"
    report_meta = {**meta, "tests_passed": summary["passed"], "tests_executed": summary["executed"],
                   "pass_rate": pct(summary["pass_rate"])}
    lines = [
        f"## Kimi Vendor Verifier Test Report: {title}",
        "",
        f"### Metadata: {title}",
        "",
        "```json",
        json.dumps(report_meta, indent=4, ensure_ascii=False),
        "```",
        "",
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: {STATUS_ICON[overall]} `{overall}`",
        f"- Overall pass rate: `{pct(summary['pass_rate'])}` ({summary['passed']}/{summary['executed']} passed)",
    ]
    if unsupported:
        names = ", ".join(f"`{f}`" for f in unsupported)
        lines += [
            f"- Pass rate excluding unsupported features: `{pct(summary['pass_rate_excl_unsupported'])}` "
            f"({summary['passed']}/{summary['executed'] - n_waived} passed)",
            f"- Unsupported features (waived): {names} — {n_waived}/{n_fail} failures "
            f"(`{pct(summary['unsupported_share_of_failures'])}` of all failures)",
        ]
    for s in suites:
        if s.collection_error:
            detail = f"did not run: {s.collection_error}"
        else:
            detail = f"{s.count('passed')}/{s.executed} passed, {pct(s.pass_rate)}"
            if s.waived:
                detail += f", {s.waived} waived"
            if s.count("skipped"):
                detail += f", {s.count('skipped')} skipped"
        if not s.graded:
            detail += f"; not graded for {meta['model']}"
        lines.append(f"- `{s.name}`: {STATUS_ICON[s.status]} `{s.status}` ({detail})")
    if overall == "PASS":
        lines.append("- All acceptance criteria passed.")
    else:
        lines.append("- Acceptance criteria not met: every graded test must pass or be waived as unsupported.")

    lines += [
        "",
        "---",
        "",
        "### Summary",
        "",
        "| Suite | Description | Graded | Passed | Failed | Waived | Skipped | Pass Rate | Pass Rate (excl. unsupported) | Duration (s) | Status |",
        "|:------|:------------|:-------|-------:|-------:|-------:|--------:|----------:|------------------------------:|-------------:|:-------|",
    ]
    for s in suites:
        lines.append(
            f"| `{s.name}` | {SUITES.get(s.name, '')} | {'yes' if s.graded else 'no'} | {s.count('passed')} "
            f"| {s.count('failed') + s.count('error')} | {s.waived} | {s.count('skipped')} | {pct(s.pass_rate)} "
            f"| {pct(s.pass_rate_excl_unsupported)} | {s.duration:.1f} | {STATUS_ICON[s.status]} {s.status} |"
        )
    lines.append(
        f"| **Total (graded)** | | | **{summary['passed']}** | **{n_fail}** | **{n_waived}** "
        f"| **{sum(s.count('skipped') for s in graded)}** | **{pct(summary['pass_rate'])}** "
        f"| **{pct(summary['pass_rate_excl_unsupported'])}** | | {STATUS_ICON[overall]} {overall} |"
    )

    lines += ["", "---", "", "### Failures by Feature", ""]
    if not n_fail:
        lines.append("No failures in graded suites.")
    else:
        lines += [
            f"Share of the {n_fail} failures in graded suites attributed to each API feature."
            + (" Features marked unsupported are waived." if unsupported else ""),
            "",
            "| Feature | Failed Tests | Share of Failures | Suites | Status |",
            "|:--------|-------------:|------------------:|:-------|:-------|",
        ]
        failures = [t for s in graded for t in s.failures]
        for feature, info in summary["failures_by_feature"].items():
            suites_hit = sorted({t.suite for t in failures if t.feature == feature})
            status = "🟨 Unsupported (waived)" if info["unsupported"] else "❌ Failing"
            lines.append(f"| {FEATURES[feature]} | {info['failed']} | {info['share'] * 100:.1f}% "
                         f"| {', '.join(f'`{x}`' for x in suites_hit)} | {status} |")

    for s in suites:
        lines += ["", "---", "", f"### Pytest Verifier — {s.name} for {title}", ""]
        if not s.graded:
            lines += [f"⚪ Not graded: this suite is reserved for models matching "
                      f"`{MODEL_SCOPED_SUITES.get(s.name, '')}`. Results are informational only.", ""]
        if s.collection_error:
            lines += [f"❌ Suite did not run: `{md_escape(s.collection_error)}`"]
            continue
        lines.append(f"Pass rate: **{pct(s.pass_rate)}** ({s.count('passed')}/{s.executed} passed"
                     + (f", {s.waived} waived" if s.waived else "")
                     + (f", {s.count('skipped')} skipped" if s.count("skipped") else "") + ")")
        failing = s.failures
        if not failing:
            lines += ["", "All executed tests passed."]
        else:
            groups: dict[tuple[str, str], list[TestResult]] = {}
            for t in failing:
                groups.setdefault((classify(t.message), t.feature), []).append(t)
            lines += [
                "",
                "#### Failure causes",
                "",
                "| Cause | Feature | Failed Tests | Share of Failures | Example Test |",
                "|:------|:--------|-------------:|------------------:|:-------------|",
            ]
            for (cause, feature), tests in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                tag = " (unsupported)" if feature in unsupported else ""
                lines.append(f"| {md_escape(cause)} | {FEATURES[feature]}{tag} | {len(tests)} "
                             f"| {len(tests) / len(failing) * 100:.0f}% | `{md_escape(tests[0].name)}` |")
            lines += [
                "",
                f"<details><summary>Failing tests ({len(failing)})</summary>",
                "",
                "| Test | Feature | Cause | Message |",
                "|:-----|:--------|:------|:--------|",
            ]
            for t in failing:
                msg = re.sub(r"\s+", " ", t.message.strip())
                msg = msg[:240] + ("…" if len(msg) > 240 else "")
                tag = " (waived)" if t.waived else ""
                lines.append(f"| `{md_escape(t.name)}` | {FEATURES[t.feature]}{tag} "
                             f"| {md_escape(classify(t.message))} | {md_escape(msg)} |")
            lines += ["", "</details>"]
        skipped = [t for t in s.tests if t.outcome == "skipped"]
        if skipped:
            lines += ["", f"<details><summary>Skipped tests ({len(skipped)})</summary>", "",
                      "| Test | Reason |", "|:-----|:-------|"]
            for t in skipped:
                reason = t.message.strip().split(": ", 1)[-1] if t.message else ""
                lines.append(f"| `{md_escape(t.name)}` | {md_escape(reason)} |")
            lines += ["", "</details>"]

    lines += [
        "",
        "---",
        "",
        f"Full failure messages and tracebacks for every test: `verifier_results_{meta['report_id']}.json`.",
        "",
        "Note: pass rates exclude skipped tests, and totals exclude suites that are not graded for this "
        "model. A test counts as failed only after all configured reruns failed. Failures attributed to a "
        "feature declared unsupported are waived: they are still listed, but do not fail acceptance. "
        "`tool_call_json_schema` run with `tool_choice=auto` is not directly comparable to the default "
        "`tool_choice=required`, because the model may answer in plain text instead of calling the tool. "
        "`prompt_tokens` compares against the case constants (collected on K3) when "
        "`prompt_tokens_reference` is `cases`; otherwise the expected count is computed from that "
        "tokenizer's chat template plus the image token count of its vision processor.",
        "",
    ]
    return "\n".join(lines)


def build_json_report(meta: dict, suites: list[SuiteResult], summary: dict, markdown: str) -> dict:
    """Machine-readable report in the shape of tt-inference-server's report JSON
    (metadata / sections / acceptance_*), with every test's full failure message.

    Deliberately NOT named report_*.json: the exabox merger ingests those and
    this verifier is report-only.
    """
    sections = []
    categories = []
    blockers: dict[str, str] = {}
    for s in suites:
        results = []
        suite_blockers: dict[str, str] = {}
        suite_waived: dict[str, str] = {}
        for t in s.tests:
            entry = {"test": t.name, "classname": t.classname, "outcome": t.outcome}
            if t.outcome != "passed":
                entry["message"] = t.message
                if t.details and t.details != t.message:
                    entry["details"] = t.details
            if t.is_failure:
                entry.update(feature=t.feature, cause=classify(t.message), waived=t.waived)
                key = f"{s.name}:{t.name}"
                if t.waived or not s.graded:
                    suite_waived[key] = f"{classify(t.message)} ({'unsupported: ' + t.feature if t.waived else 'not graded'})"
                else:
                    suite_blockers[key] = classify(t.message)
            results.append(entry)
        if s.collection_error:
            suite_blockers[f"{s.name}:collection"] = s.collection_error
        if s.graded:
            blockers.update(suite_blockers)
        sections.append({
            "kind": "vendor_verifier",
            "title": f"Pytest Verifier — {s.name}",
            "task_type": "api_contract",
            "id": s.name,
            "data": {
                "suite": s.name,
                "description": SUITES.get(s.name, ""),
                "graded": s.graded,
                "status": s.status,
                "passed": s.count("passed"),
                "failed": s.count("failed"),
                "errors": s.count("error"),
                "skipped": s.count("skipped"),
                "waived": s.waived,
                "pass_rate": s.pass_rate,
                "pass_rate_excl_unsupported": s.pass_rate_excl_unsupported,
                "duration_s": round(s.duration, 1),
                "collection_error": s.collection_error or None,
            },
            "results": results,
        })
        categories.append({
            "name": s.name,
            "status": s.status,
            "total": s.executed,
            "passed": s.count("passed"),
            "failed": s.count("failed") + s.count("error"),
            "na": 0,
            "skipped": s.count("skipped"),
            "blockers": suite_blockers if s.graded else {},
            "waived": {**suite_waived, **({} if s.graded else suite_blockers)},
        })

    acceptance_md = markdown.split("### Acceptance Criteria", 1)[-1].split("\n---\n", 1)[0]
    return {
        "metadata": {
            "model_name": meta["model"].split("/")[-1],
            "model_repo": meta["model"],
            "provider": meta["provider"],
            "generated_at": meta["generated_at"],
            "report_id": meta["report_id"],
            "workflow": "vendor_verifier",
            "verifier": "kimi-vendor-verifier",
            "report_partial": any(s.collection_error for s in suites),
            "report_blocks": len(sections),
            "server_mode": "API",
            **{k: v for k, v in meta.items() if k not in ("model", "provider", "generated_at", "report_id")},
        },
        "sections": sections,
        "acceptance_criteria": summary["status"] == "PASS",
        "acceptance_blockers": blockers,
        "acceptance_criteria_metadata": {
            "enforcement_result": summary["status"],
            "pass_rate": summary["pass_rate"],
            "pass_rate_excl_unsupported": summary["pass_rate_excl_unsupported"],
            "unsupported_features": meta["unsupported"],
            "failures_by_feature": summary["failures_by_feature"],
            "categories": categories,
        },
        "acceptance_summary_markdown": "### Acceptance Criteria" + acceptance_md.rstrip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--provider", default="vendor", help="Provider name shown in the report title")
    parser.add_argument("--think-mode", choices=["kimi", "opensource", "none"],
                        default=os.environ.get("THINK_MODE", "kimi"))
    parser.add_argument("--tool-choice", choices=["required", "auto"],
                        default=os.environ.get("TOOL_CHOICE", "required"))
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--workers", nargs="+", default=["1"], metavar="N|SUITE=N",
                        help="pytest-xdist workers: one number for every suite, and/or SUITE=N overrides")
    parser.add_argument("--reruns", type=int, default=3, help="Reruns for tool_call_json_schema")
    parser.add_argument("--unsupported", nargs="*", choices=list(FEATURES), default=None,
                        help="Features the deployment does not support; their failures are waived")
    parser.add_argument("--grade-model-scoped", choices=["auto", "always", "never"], default="auto",
                        help="Grade model-scoped suites (k3_features) only for matching models (auto), "
                             "always, or never")
    parser.add_argument("--prompt-tokens-reference", default=os.environ.get("PROMPT_TOKENS_REFERENCE", "auto"),
                        help="Expected prompt_tokens source: 'cases' (K3 constants), a Hugging Face repo id or "
                             "local tokenizer dir, or 'auto' (cases for K3 models, otherwise the model id)")
    parser.add_argument("--strip-reasoning-effort", choices=["auto", "yes", "no"],
                        default=os.environ.get("STRIP_REASONING_EFFORT", "auto"),
                        help="Drop reasoning_effort from prompt_tokens requests (auto: for non-K3 models)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--render-only", metavar="RUN_DIR",
                        help="Rebuild the report from RUN_DIR without running the tests")
    args = parser.parse_args()

    if args.render_only:
        out_dir = Path(args.render_only)
        meta = json.loads((out_dir / "run_meta.json").read_text())
        if args.unsupported is not None:
            meta["unsupported"] = args.unsupported
    else:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model are required (or set KIMI_BASE_URL / MODEL_NAME)")
        if not os.environ.get("KIMI_API_KEY"):
            parser.error("KIMI_API_KEY must be set in the environment")
        workers = parse_workers(args.workers, args.suites)
        is_k3 = bool(re.search(K3_PATTERN, args.model, re.IGNORECASE))
        reference = args.prompt_tokens_reference
        if reference == "auto":
            reference = "cases" if is_k3 or "/" not in args.model else args.model
        strip_effort = args.strip_reasoning_effort == "yes" or (
            args.strip_reasoning_effort == "auto" and not is_k3)
        now = datetime.now()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.provider}_{args.model}")
        out_dir = Path(args.output_dir) / f"{slug}_{now:%Y-%m-%d_%H-%M-%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": args.model,
            "provider": args.provider,
            "base_url": args.base_url,
            "generated_at": f"{now:%Y-%m-%d %H:%M:%S}",
            "report_id": out_dir.name,
            "think_mode": args.think_mode,
            "tool_choice": args.tool_choice,
            "suites": args.suites,
            "workers": workers,
            "unsupported": args.unsupported or [],
            "prompt_tokens_reference": reference,
            "strip_reasoning_effort": strip_effort,
            "run_command": shlex.join(["python", "test_report.py", *sys.argv[1:]]),
            "verifier_commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                capture_output=True, text=True).stdout.strip() or None,
        }
        meta["runs"] = run_suites(args, meta, out_dir, workers)
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    meta.setdefault("unsupported", [])
    meta["ungraded_suites"] = [
        s for s, pattern in MODEL_SCOPED_SUITES.items()
        if s in meta["suites"] and (
            args.grade_model_scoped == "never"
            or (args.grade_model_scoped == "auto" and not re.search(pattern, meta["model"], re.IGNORECASE))
        )
    ]

    suites = [parse_junit(out_dir / f"{name}.xml", name) for name in meta["suites"]]
    for s in suites:
        s.graded = s.name not in meta["ungraded_suites"]
        for t in s.failures:
            t.feature = attribute_feature(t)
            t.waived = t.feature in meta["unsupported"]

    report_meta = {k: v for k, v in meta.items() if k != "runs"}
    summary = summarize(report_meta, suites)
    report_path = out_dir / f"report_{meta['report_id']}.md"
    markdown = render(report_meta, suites, summary)
    report_path.write_text(markdown)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    json_path = out_dir / f"verifier_results_{meta['report_id']}.json"
    json_path.write_text(json.dumps(build_json_report(report_meta, suites, summary, markdown),
                                    indent=4, ensure_ascii=False) + "\n")
    print(f"[report] {report_path}")
    print(f"[json] {json_path}")
    print(f"[summary] {out_dir / 'summary.json'}")
    print(f"[status] {summary['status']} pass_rate={pct(summary['pass_rate'])} "
          f"excl_unsupported={pct(summary['pass_rate_excl_unsupported'])}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
