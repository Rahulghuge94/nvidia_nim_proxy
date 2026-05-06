# NVIDIA NIM Compatibility Proxy

An installable, dependency-free Python proxy for NVIDIA NIM. It exposes
OpenAI-compatible chat completions, a basic Codex/OpenAI Responses adapter, and
a basic Anthropic/Claude Messages adapter.

## Install

From this folder:

```powershell
python -m pip install -e .
```
From git clone

After installation, run either command:

```powershell
nim-proxy --host 127.0.0.1 --port 8000
```

```powershell
python -m nvidia_nim_proxy --host 127.0.0.1 --port 8000
```

The old script entry point still works too:

```powershell
python nim_proxy.py --host 127.0.0.1 --port 8000
```

## Configuration

PowerShell:

```powershell
$env:NVIDIA_API_KEY="your_nvidia_api_key"
$env:NVIDIA_NIM_MODEL="deepseek-ai/deepseek-v4-pro"
```

`NVDIA_API_KEY` is also accepted because the original example used that spelling,
but `NVIDIA_API_KEY` is preferred.

Optional settings:

- `NVIDIA_NIM_UPSTREAM` - defaults to `https://integrate.api.nvidia.com/v1`
- `NVIDIA_NIM_MODEL` - default model used when clients omit `model`
- `NVIDIA_NIM_TIMEOUT` - upstream timeout in seconds

## Supported Routes

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`

Any other `/v1/...` route is passed through to NVIDIA's upstream API.

## OpenAI Compatible Clients

Use this base URL:

```text
http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-proxy-key",
)

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-pro",
    messages=[{"role": "user", "content": "Write a Python HTTP request."}],
)

print(completion.choices[0].message.content)
```

## Codex / Responses API Clients

Use this base URL:

```text
http://127.0.0.1:8000/v1
```

Example:

```powershell
curl http://127.0.0.1:8000/v1/responses `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer local-proxy-key" `
  -d '{"model":"deepseek-ai/deepseek-v4-pro","input":"Say hello from NIM."}'
```

The proxy maps basic `POST /v1/responses` requests into NVIDIA chat completions
and returns a Responses-shaped JSON object with `output_text`.

## Claude / Anthropic Messages Clients

Use this base URL:

```text
http://127.0.0.1:8000
```

Example:

```powershell
curl http://127.0.0.1:8000/v1/messages `
  -H "Content-Type: application/json" `
  -H "x-api-key: local-proxy-key" `
  -H "anthropic-version: 2023-06-01" `
  -d '{"model":"deepseek-ai/deepseek-v4-pro","max_tokens":512,"messages":[{"role":"user","content":"Say hello from NIM."}]}'
```

The proxy maps basic `POST /v1/messages` requests into NVIDIA chat completions
and returns an Anthropic Messages-shaped JSON object.

## Test

```powershell
python -m unittest discover -s tests
```

```powershell
python -m py_compile nim_proxy.py src/nvidia_nim_proxy/*.py
```

## Notes

- The local client API key can be any value; the proxy always sends your
  `NVIDIA_API_KEY` to NVIDIA upstream.
- Non-streaming requests are translated into the target response format.
- Streaming requests are forwarded as NVIDIA/OpenAI-style server-sent events.
- The package has no runtime dependencies outside the Python standard library.
