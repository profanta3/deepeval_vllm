# deepeval-vllm

Compatibility wrapper for DeepEval's `LocalModel` that adds native structured output support for [vLLM](https://github.com/vllm-project/vllm) servers.

## Why

DeepEval's built-in `LocalModel` talks to any OpenAI-compatible endpoint, but when a metric needs a structured result it just asks the model for JSON in the prompt and then tries to salvage a JSON object out of whatever came back (`trim_and_load_json`). With smaller local models that fails often enough to poison an eval run -- malformed JSON, missing fields, prose wrapped around the payload.

vLLM can enforce the schema server-side with constrained decoding. `VLLMLocalModel` passes the Pydantic schema through as vLLM's `structured_outputs` request field, so the response is guaranteed to match the schema instead of merely being asked to.

Everything else -- client construction, retries, multimodal input, config resolution from env vars -- is inherited from `LocalModel` unchanged.

## Installation

```bash
uv add deepeval-vllm
```

Requires Python 3.12+.

## Usage

Point it at your vLLM server and hand it to any DeepEval metric:

```python
from deepeval_vllm import VLLMLocalModel

model = VLLMLocalModel(
    model="Qwen/Qwen3-8B",
    base_url="http://localhost:8000/v1",
    # ...
)
```

The constructor signature is `LocalModel`'s, so the usual env vars work too:

```bash
export LOCAL_MODEL_NAME="Qwen/Qwen3-8B"
export LOCAL_MODEL_BASE_URL="http://localhost:8000/v1"
export LOCAL_MODEL_API_KEY="not-needed"
```

```python
model = VLLMLocalModel()
```

## Requirements

- A running vLLM server with an OpenAI-compatible API, recent enough to accept the `structured_outputs` request field (>=v0.12.0). Older builds use `guided_json` instead and will reject or ignore these requests.
- `deepeval >= 4.1.8`


## Development

```bash
uv sync
```

The package lives in `deepeval_vllm/`; `vllm_wrapper.py` overrides `generate` and `a_generate` and nothing else.
