# [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)

English | [中文](README_zh.md)

## K3 Evaluation Results

Listed in the order of submission time.

model: kimi-k3

thinking effort: max

| Provider | OCRBench | MMMU Pro Vision | BEAM (1M) | DeepSWE |
|----------|----------|-----------------|-----------|---------|
| Moonshot | 0.89 | 0.82 | 0.31 | 0.675 |
| Fireworks | 0.89 | 0.82 | 0.3037 | 0.664 |
| Baseten | 0.889 | 0.804 | 0.3219 | 0.693 |
| Together | 0.897 | 0.820 | 0.3160 | 0.678 |
| DigitalOcean | 0.89 | 0.816 | TBD | TBD |
| Inferact (vLLM ref.) | 0.891 | 0.818 | 0.3188 | 0.695 |
| Nebius | 0.878 | 0.814 | 0.2913 | 0.673 |
| Modal | 0.887 | 0.817 | 0.322 | 0.658 |
| RadixArk (SGlang ref.) | 0.895 | 0.820 | 0.3107 | 0.667 |

## Overview

| Type | Task | Description |
|------|------|-------------|
| Inspect-ai benchmark | OCRBench | OCR text recognition benchmark |
| Inspect-ai benchmark | MMMU Pro Vision | Multimodal understanding benchmark (visual QA) |
| Inspect-ai benchmark | AIME 2025 | Mathematical reasoning benchmark; no longer used for K3 |
| Standalone script | BEAM (1M) | Long-term memory benchmark over 1M-token conversations, see [beam/](beam/README.md) |
| Pytest verifier | `tests/params/` | API parameter constraint pre-flight validation |
| Pytest verifier | `tests/tool_call_json_schema/` | Tool-call argument validation for walle-valid MFJS schemas |
| Pytest verifier | `tests/k3_features/` | K3 feature contract validation, including dynamic tools, response_format, tool_choice, and thinking effort |
| Pytest verifier | `tests/prompt_tokens/` | Verifies vendor-reported `usage.prompt_tokens` against expected constants |
| Agent benchmark | [DeepSWE](https://github.com/datacurve-ai/deep-swe) | Multi-step tool-use and coding-agent evaluation, run on the Pier platform |

## Environment Setup

### Install Dependencies

```bash
uv sync && uv pip install -e .
```

### Configure Environment Variables

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
```

Or copy `.env.example` to `.env` and fill in the configuration.

## Pre-flight Validation

Before running benchmarks, complete the API parameter, tool-call schema, K3 feature, and prompt-token pre-flight checks.

### Parameter Constraint Validation: `tests/params/`

Validate that the API correctly constrains immutable parameters such as `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, and `n`.

```bash
# Kimi official API
uv run pytest tests/params --smoke-model kimi/your-model-id --think-mode kimi -v

# Open-source deployment (vLLM/SGLang/KTransformers)
uv run pytest tests/params --smoke-model your-model-id --think-mode opensource -v
```

Run formal benchmarks only after all tests pass.

### Tool Call JSON Schema Validation: `tests/tool_call_json_schema/`

Validate that the vendor can use walle-valid MFJS schemas as tool-call `parameters` and return `tool_calls[].function.arguments` that conform to the schema.

```bash
uv run pytest -n 4 tests/tool_call_json_schema \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    --think-mode "$THINK_MODE" \
    --thinking \
    --reruns 3 \
    --reruns-delay 2 \
    --tool-json-report=tool-call-schema-report.json \
    -ra -v
```

This check runs walle `valid.jsonl` cases, sends each schema as `tools[].function.parameters`, forces a tool call, and uses `jsonschema` to validate the returned `function.arguments` locally. Each selected case is run once with `stream=false` and once with `stream=true`; streaming chunks are reassembled before validation. In GitLab CI, this check runs as the `verify_tool_call_json_schema` job; the same retry options are applied. In addition to the JUnit XML report, a JSON report (`tool-call-schema-report.json`) is produced containing the selected cases, per-case outcomes, and the same `summary` block (total / by_status / by_selection_reason / by_mode) that the original standalone script printed at the end of its log.

#### Arguments

| Argument | Meaning | Default |
|----------|---------|---------|
| `--smoke-model` | Vendor model name | `MODEL_NAME` env var |
| `--base-url` | API base URL | `KIMI_BASE_URL` env var, otherwise Moonshot API |
| `--api-key` | API key | `KIMI_API_KEY` env var |
| `--case-dir` | walle `validator_cases` directory | `testdata/walle_validator_cases/validator_cases` |
| `--selection` | Case selection mode: `all` / `explicit` / `object` | `all` |
| `--max-cases` | Maximum number of selected cases to run | No limit |
| `--thinking` | Enable thinking mode for the tool-call request | Off |
| `--think-mode` | Thinking parameter format: `kimi`, `opensource`, or `none` | `THINK_MODE` env var, otherwise `kimi` |
| `--max-tokens` | Maximum output tokens for each tool-call response | `2048` |
| `--tool-choice` | `tool_choice` for each request: `required` forces the call; `auto` is for vendors without guided decoding and fails cases where the model answers in plain text | `TOOL_CHOICE` env var, otherwise `required` |
| `--tool-json-report` | Path to the additional JSON report artifact | `tool-call-schema-report.json` |

### K3 Feature Validation: `tests/k3_features/`

Validate K3 API features such as dynamic tools, `response_format`, `tool_choice`, and thinking effort.

```bash
uv run pytest tests/k3_features \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `--smoke-model` | Vendor model name | `MODEL_NAME` env var |
| `--base-url` | API base URL | `KIMI_BASE_URL` env var, otherwise Moonshot API |
| `--api-key` | API key | `KIMI_API_KEY` env var |

The CI job `verify_k3_features` runs the pytest suite and reports results from the test log.

### Prompt Token Validation: `tests/prompt_tokens/`

Validate that vendor-reported `usage.prompt_tokens` is accurate. This check
sends the text cases in `testdata/prompt_token_cases/cases.jsonl` and vision
cases in `testdata/prompt_token_cases/vision_cases.jsonl` with `stream=true`,
then compares the stream-reported `usage.prompt_tokens` against the expected
constant stored in each case. See
[testdata/prompt_token_cases/README.md](testdata/prompt_token_cases/README.md)
for the case format.

```bash
uv run pytest tests/prompt_tokens \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

The case constants were collected on K3, whose chat template adds content other Kimi models do not (e.g. a thinking-effort block). For other models, pass `--prompt-tokens-reference <hf-repo-or-dir>` to compute the expected count from that model's published tokenizer and chat template instead: the rendered prompt length plus, for each image, the token count from the model's vision processor config (minus the one `<|media_pad|>` placeholder the template emits). Add `--strip-reasoning-effort` for models without thinking effort (e.g. Kimi-K2.7), so the K3-only `reasoning_effort` field is not sent.

```bash
uv run pytest tests/prompt_tokens -n 16 \
    --base-url "${KIMI_BASE_URL}" \
    --smoke-model moonshotai/Kimi-K2.7-Code \
    --prompt-tokens-reference moonshotai/Kimi-K2.7-Code \
    --strip-reasoning-effort
```

### Run All Suites with a Report: `run_verifier.sh`

`run_verifier.sh` runs every pytest suite in parallel (16 pytest-xdist workers per suite by default) and writes a Markdown report plus a machine-readable `summary.json` to `reports/<provider>_<model>_<timestamp>/`. The report lists per-suite pass rates, groups failing tests by cause, and attributes every failure to an API feature. Features declared in `UNSUPPORTED` (e.g. `vision`, `structured_output`) are waived: they are still listed but do not fail acceptance, and the report gives their share of all failures. `k3_features` is graded only when the model name contains `K3`; for other models it runs for information only, and `prompt_tokens` automatically uses the model's own Hugging Face tokenizer as the reference and drops `reasoning_effort` (override with `PROMPT_TOKENS_REFERENCE` / `STRIP_REASONING_EFFORT`). The script can be called from any directory and exits `0` on acceptance, `1` otherwise.

```bash
KIMI_API_KEY=... \
KIMI_BASE_URL=https://api.example.com/v1 \
MODEL_NAME=moonshotai/Kimi-K2.7-Code \
PROVIDER=Example TOOL_CHOICE=auto UNSUPPORTED="vision structured_output" \
./run_verifier.sh
```

See the header of `run_verifier.sh` for all settings (worker counts per suite, suites, reruns, report directory). `uv run python test_report.py --render-only reports/<run-dir>` rebuilds a report from an existing run without calling the API.

### Why These Tests Are Not in Inspect-ai

`tests/k3_features/`, `tests/prompt_tokens/`, and `tests/tool_call_json_schema/` validate raw chat-completions API behavior, including non-standard `messages[].tools` dynamic-tool declarations, malformed requests that should return HTTP 400, the location of usage fields in streaming chunks, and raw `tools[].function.parameters` JSON schema handling. inspect-ai is better suited for scored benchmark evaluations and may normalize or reject these raw payloads before they reach the vendor endpoint, so these tests remain standalone pytest suites.

## Recommended Benchmark Parameters

### K2.6 and Earlier

| Benchmark | Mode | Temperature | TopP | Max Tokens | Epochs |
|-----------|------|-------------|------|------------|--------|
| OCRBench | Non-Thinking | 0.6 | 0.95 | 16384 | 1 |
| OCRBench | Thinking | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Non-Thinking | 0.6 | 0.95 | 65536 | 1 |
| MMMU | Thinking | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Non-Thinking | 0.6 | 0.95 | 98304 | 32 |
| AIME 2025 | Thinking | 1.0 | 0.95 | 98304 | 32 |

### K2.7

| Benchmark | Mode (thinking, preserve thinking) | Temperature | TopP | Max Tokens | Epochs |
|-----------|------------------------------------|-------------|------|------------|--------|
| OCRBench | Thinking, Keep all | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Thinking, Keep all | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Thinking, Keep all | 1.0 | 0.95 | 98304 | 32 |

### K3

K3 no longer uses AIME 2025 and adds the `reasoning_effort` parameter.

| Benchmark | Mode (thinking, preserve thinking, effort) | Temperature | TopP | Max Tokens | Epochs |
|-----------|--------------------------------------------|-------------|------|------------|--------|
| OCRBench | Thinking, Keep all, low | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | Thinking, Keep all, high | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | Thinking, Keep all, max | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Thinking, Keep all, low | 1.0 | 0.95 | 98304 | 1 |
| MMMU | Thinking, Keep all, high | 1.0 | 0.95 | 98304 | 1 |
| MMMU | Thinking, Keep all, max | 1.0 | 0.95 | 98304 | 1 |

## Running Inspect-ai Benchmarks

### OCRBench

#### Non-Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking + reasoning effort (K3)

```bash
uv run python eval.py ocrbench \
    --model "${THINK_MODE}/${MODEL_NAME}" \
    --max-tokens 16384 \
    --thinking \
    --think-mode "$THINK_MODE" \
    --stream \
    --max-connections 50 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking-effort high
```

### MMMU Pro Vision

#### Non-Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --think-mode kimi --max-tokens 65536 --stream
```

#### Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 65536 --stream
```

#### Thinking + reasoning effort (K3)

```bash
uv run python eval.py mmmu \
    --model "$MODEL" \
    --max-tokens 98304 \
    --thinking \
    --think-mode "$THINK_MODE" \
    --stream \
    --max-connections 50 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking-effort high
```

### AIME 2025

#### Non-Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --think-mode kimi --max-tokens 98304 --stream
```

#### Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

Run OCRBench first as a quick deployment sanity check, then run the full MMMU and other benchmark evaluations.

## Agent Benchmark (DeepSWE)

To cover that capability, we also add an agent benchmark based on the open-source [DeepSWE](https://github.com/datacurve-ai/deep-swe) benchmark. DeepSWE contains 113 coding-agent tasks and is used here through the open-source Pier platform. The agent runtime uses Kimi-code `0.23.6` or later.

Before running the benchmark on Pier, register `kimi-code` on the Pier agent first. See the Kimi-code `0.23.6` setup reference: https://github.com/MoonshotAI/kimi-code/tree/%40moonshot-ai/kimi-code%400.23.6

## Inspect-ai Parameters

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `benchmark` | Evaluation task: `ocrbench`, `mmmu`, `aime2025` | `ocrbench` |
| `--model` | Model identifier, for example `kimi/your-model-id` | **Required** |
| `--max-tokens` | Maximum output tokens, see recommended benchmark parameters | **Required** |
| `--thinking` | Enable thinking mode | Off |
| `--think-mode` | Thinking parameter format: `none`, `kimi`, or `opensource` | `none` |
| `--temperature` | Sampling temperature | Optional; determined by server or model defaults if omitted |
| `--top-p` | Top-p sampling | Optional; determined by server or model defaults if omitted |
| `--stream` | Enable streaming, recommended for long reasoning requests | Off |
| `--max-connections` | Maximum concurrent connections | Per benchmark |
| `--epochs` | Number of sampling epochs | Per benchmark |
| `--client-timeout` | HTTP timeout in seconds | `86400` |
| `--thinking-effort` | K3 reasoning effort parameter: `low`, `high`, `max` | None |

### Thinking Mode Parameters

| Model Type | Parameter Combination | `extra_body` Sent |
|------------|-----------------------|-------------------|
| Kimi official + thinking off | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
| Kimi official + thinking on | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
| Kimi official + reasoning effort low | `--thinking --think-mode kimi --thinking-effort low` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "low"}}` |
| Kimi official + reasoning effort high | `--thinking --think-mode kimi --thinking-effort high` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "high"}}` |
| Kimi official + reasoning effort max | `--thinking --think-mode kimi --thinking-effort max` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "max"}}` |
| Open-source framework + thinking off | `--think-mode opensource` | `{"chat_template_kwargs": {"thinking": false}}` |
| Open-source framework + thinking on | `--thinking --think-mode opensource` | `{"chat_template_kwargs": {"thinking": true}}` |
| Open-source framework + reasoning effort low | `--thinking --think-mode opensource --thinking-effort low` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "low"}}` |
| Open-source framework + reasoning effort high | `--thinking --think-mode opensource --thinking-effort high` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "high"}}` |
| Open-source framework + reasoning effort max | `--thinking --think-mode opensource --thinking-effort max` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "max"}}` |

### Viewing Results

```bash
# View logs with inspect view
uv run inspect view

# Logs are saved in the logs/ directory
```

### Resuming an Interrupted Benchmark

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## Notes

### AIME 2025

AIME evaluation produces many output tokens; please note the following:

1. Timeout settings: the client default is `--client-timeout 86400` (24 hours), and the server, gateway, or proxy timeout should also be long enough.
2. Streaming: strongly recommended with `--stream`; non-streaming requests are more prone to timeout in thinking mode.
3. Concurrency control: if you see many 429s or `RemoteProtocolError`s, lower `--max-connections`.
4. Quick validation: first run all samples with `--epochs 1`, then run the full evaluation after configuration is confirmed.

```bash
# Step 1: Quick validation (30 samples x 1 epoch)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream --epochs 1

# Step 2: Full evaluation (30 samples x 32 epochs)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

### Automatic Retry Mechanism

The following network-related errors are automatically retried with exponential backoff (1-60 seconds), no manual configuration needed:

| Error Type | Description |
|------------|-------------|
| `RateLimitError` / `429` | Server-side rate limiting |
| `APIConnectionError` | Connection failure |
| `ReadError` / `RemoteProtocolError` | Network read error |

Non-network errors, such as model output format issues, are not retried and are logged directly for later analysis.

## Project Structure

```text
├── eval.py              # Main evaluation CLI entrypoint
├── tests/params/        # Pre-flight parameter validation
├── tests/tool_call_json_schema/ # walle tool-call schema validation
├── tests/prompt_tokens/ # usage.prompt_tokens accuracy validation
├── tests/k3_features/   # K3 feature contract validation
├── kimi_model.py        # Kimi Model API implementation
├── aime2025.py          # AIME 2025 evaluation task
├── mmmu_pro_vision.py   # MMMU Pro Vision evaluation task
├── ocr_bench.py         # OCRBench evaluation task
├── testdata/            # JSON schema and prompt-token test data
├── beam/                # BEAM 1M context data and scripts, see the directory README
├── logs/                # Evaluation logs
└── pyproject.toml       # Project configuration
```

## Contact

If you have any questions or suggestions, please contact contact-kvv@kimi.com.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
