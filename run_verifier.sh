#!/usr/bin/env bash
# Run all pytest verifier suites in parallel against a vendor endpoint and
# write a Markdown report (plus summary.json) via test_report.py.
#
# Safe to call from any directory (e.g. from another repo's CI):
#
#   KIMI_API_KEY=sk-... \
#   KIMI_BASE_URL=https://console.tenstorrent.com/v1 \
#   MODEL_NAME=moonshotai/Kimi-K2.7-Code \
#   PROVIDER=Tenstorrent TOOL_CHOICE=auto UNSUPPORTED="vision structured_output" \
#   /path/to/Kimi-Vendor-Verifier/run_verifier.sh
#
# Settings (environment variables; unset ones fall back to <repo>/.env, then defaults):
#   KIMI_API_KEY            API key (required, never written to the report)
#   KIMI_BASE_URL           OpenAI-compatible base URL, including /v1 (required)
#   MODEL_NAME              Model id as served by the vendor (required)
#   PROVIDER                Provider name for the report title      (default: vendor)
#   THINK_MODE              kimi | opensource | none                (default: kimi)
#   TOOL_CHOICE             required | auto                         (default: required)
#   UNSUPPORTED             Space-separated features to waive, e.g. "vision structured_output"
#   SUITES                  Space-separated suites to run           (default: all four)
#   PARAMS_WORKERS          pytest-xdist workers per suite          (default: 16 each)
#   TOOL_SCHEMA_WORKERS
#   K3_WORKERS
#   PROMPT_TOKENS_WORKERS
#   TOOL_SCHEMA_RERUNS      Reruns for tool_call_json_schema        (default: 3)
#   PROMPT_TOKENS_REFERENCE Expected prompt_tokens source: cases | <HF repo or dir> | auto
#                           (default: auto = cases for K3 models, otherwise MODEL_NAME)
#   STRIP_REASONING_EFFORT  auto | yes | no: drop reasoning_effort from prompt_tokens requests
#                           (default: auto = for non-K3 models)
#   DEFAULT_SAMPLING        yes | no: send temperature 1.0, top_p 0.95, top_k 50, n 1 and
#                           presence/frequency_penalty 0 on requests that do not set them (default: no)
#   SAMPLING_PARAMS         JSON object overriding those values, e.g. '{"top_k": 0, "seed": null}'
#   CACHE_BYPASS            yes | no: unique id at the start of every prompt so no request can hit
#                           the server's prefix cache; changes prompt token counts (default: no)
#   REPORT_DIR              Where run directories are created       (default: <repo>/reports)
#
# Extra arguments are passed through to test_report.py.
# Exit code: 0 if acceptance passed, 1 if it failed, 2 on setup errors.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "run_verifier: $*" >&2; exit 2; }

# Load <repo>/.env without overriding variables that are already set.
if [[ -f "$REPO_DIR/.env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key="${BASH_REMATCH[2]}"
        value="${BASH_REMATCH[3]}"
        value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
        if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
    done < "$REPO_DIR/.env"
fi

: "${KIMI_API_KEY:?KIMI_API_KEY is required}"
: "${KIMI_BASE_URL:?KIMI_BASE_URL is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
PROVIDER="${PROVIDER:-vendor}"
THINK_MODE="${THINK_MODE:-kimi}"
TOOL_CHOICE="${TOOL_CHOICE:-required}"
UNSUPPORTED="${UNSUPPORTED:-}"
SUITES="${SUITES:-params tool_call_json_schema k3_features prompt_tokens}"
PARAMS_WORKERS="${PARAMS_WORKERS:-16}"
TOOL_SCHEMA_WORKERS="${TOOL_SCHEMA_WORKERS:-16}"
K3_WORKERS="${K3_WORKERS:-16}"
PROMPT_TOKENS_WORKERS="${PROMPT_TOKENS_WORKERS:-16}"
TOOL_SCHEMA_RERUNS="${TOOL_SCHEMA_RERUNS:-3}"
PROMPT_TOKENS_REFERENCE="${PROMPT_TOKENS_REFERENCE:-auto}"
STRIP_REASONING_EFFORT="${STRIP_REASONING_EFFORT:-auto}"
DEFAULT_SAMPLING="${DEFAULT_SAMPLING:-no}"
CACHE_BYPASS="${CACHE_BYPASS:-no}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/reports}"

[[ "$KIMI_BASE_URL" == */v1 || "$KIMI_BASE_URL" == */v1/ ]] || \
    echo "run_verifier: warning: KIMI_BASE_URL ($KIMI_BASE_URL) does not end in /v1" >&2

command -v uv >/dev/null 2>&1 || die "uv is not installed (https://docs.astral.sh/uv/)"

# prompt_tokens fixtures live in Git LFS; fetch them if only pointers are present.
if [[ " $SUITES " == *" prompt_tokens "* ]] && \
   head -c 40 "$REPO_DIR/testdata/prompt_token_cases/cases.jsonl" | grep -q "git-lfs"; then
    if git -C "$REPO_DIR" lfs version >/dev/null 2>&1; then
        echo "run_verifier: fetching prompt_tokens LFS fixtures" >&2
        git -C "$REPO_DIR" lfs pull --include="testdata/prompt_token_cases/**"
    else
        # No git-lfs (e.g. a bare CI runner): fetch the objects over HTTPS from
        # GitHub's LFS media endpoint for this checkout's origin and commit.
        remote="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
        [[ "$remote" =~ github\.com[:/]([^/]+/[^/]+)$ ]] || \
            die "prompt_tokens needs Git LFS data; install git-lfs or drop prompt_tokens from SUITES"
        slug="${BASH_REMATCH[1]%.git}"
        rev="$(git -C "$REPO_DIR" rev-parse HEAD)"
        echo "run_verifier: git-lfs not found; downloading prompt_tokens fixtures from $slug@$rev" >&2
        while IFS= read -r -d '' f; do
            head -c 40 "$f" | grep -q "git-lfs" || continue
            rel="${f#"$REPO_DIR"/}"
            size="$(sed -n 's/^size //p' "$f")"
            curl -fsSL --retry 3 -o "$f.download" "https://media.githubusercontent.com/media/$slug/$rev/$rel" || \
                die "failed to download LFS object $rel"
            [[ "$(wc -c < "$f.download" | tr -d ' ')" == "$size" ]] || die "size mismatch for LFS object $rel"
            mv "$f.download" "$f"
        done < <(find "$REPO_DIR/testdata/prompt_token_cases" -type f -print0)
    fi
fi

# Use the verifier's own environment, not whatever venv the caller has active.
unset VIRTUAL_ENV

args=(
    --base-url "$KIMI_BASE_URL"
    --model "$MODEL_NAME"
    --provider "$PROVIDER"
    --think-mode "$THINK_MODE"
    --tool-choice "$TOOL_CHOICE"
    --reruns "$TOOL_SCHEMA_RERUNS"
    --prompt-tokens-reference "$PROMPT_TOKENS_REFERENCE"
    --strip-reasoning-effort "$STRIP_REASONING_EFFORT"
    --default-sampling "$DEFAULT_SAMPLING"
    --cache-bypass "$CACHE_BYPASS"
    --output-dir "$REPORT_DIR"
    --workers
        "params=$PARAMS_WORKERS"
        "tool_call_json_schema=$TOOL_SCHEMA_WORKERS"
        "k3_features=$K3_WORKERS"
        "prompt_tokens=$PROMPT_TOKENS_WORKERS"
)
# shellcheck disable=SC2206  # word splitting of the space-separated lists is intended
args+=(--suites $SUITES)
if [[ -n "$UNSUPPORTED" ]]; then
    # shellcheck disable=SC2206
    args+=(--unsupported $UNSUPPORTED)
fi

cd "$REPO_DIR"
exec uv run --project "$REPO_DIR" python test_report.py "${args[@]}" "$@"
