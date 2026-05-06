"""HTTP server and protocol adapters for the NVIDIA NIM proxy."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro"


@dataclass(frozen=True)
class ProxyConfig:
    """Runtime configuration for the local proxy."""

    host: str = "127.0.0.1"
    port: int = 8000
    upstream_base: str = NIM_DEFAULT_BASE
    api_key: Optional[str] = None
    default_model: str = DEFAULT_MODEL
    timeout_seconds: int = 300


class ProxyError(Exception):
    """Error that should be returned to the HTTP client as JSON."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class NvidiaNIMProxyHandler(BaseHTTPRequestHandler):
    upstream_base: str = NIM_DEFAULT_BASE
    api_key: Optional[str] = None
    default_model: str = DEFAULT_MODEL
    timeout_seconds: int = 300

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, Accept, X-Requested-With, Anthropic-Version, X-API-Key",
        )

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_PUT(self) -> None:
        self._proxy_passthrough()

    def do_PATCH(self) -> None:
        self._proxy_passthrough()

    def do_DELETE(self) -> None:
        self._proxy_passthrough()

    def do_HEAD(self) -> None:
        self._proxy_passthrough()

    def _handle_request(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.command == "GET" and path in {"/v1/models", "/models"}:
                self._handle_models()
            elif self.command == "POST" and path in {"/v1/responses", "/responses"}:
                self._handle_responses()
            elif self.command == "POST" and path in {"/v1/messages", "/messages"}:
                self._handle_anthropic_messages()
            else:
                self._proxy_passthrough()
        except ProxyError as error:
            self._send_json(error.status, {"error": {"message": error.message}})
        except urllib.error.HTTPError as error:
            self._relay_http_error(error)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Request body must be valid JSON"}})
        except Exception as error:  # pragma: no cover
            self._send_json(502, {"error": {"message": f"Upstream proxy error: {error}"}})

    def _handle_models(self) -> None:
        fallback = {
            "object": "list",
            "data": [
                {
                    "id": self.default_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nvidia",
                }
            ],
        }

        if not self.api_key:
            self._send_json(200, fallback)
            return

        try:
            status, headers, body = self._send_upstream("GET", "/models", None, {})
            self._relay_raw(status, headers, body)
        except Exception:
            self._send_json(200, fallback)

    def _handle_responses(self) -> None:
        payload = self._read_json_body()
        stream = bool(payload.get("stream"))
        chat_payload = self._responses_to_chat_payload(payload)

        status, headers, body = self._send_upstream(
            "POST",
            "/chat/completions",
            json.dumps(chat_payload).encode("utf-8"),
            {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        )

        if stream:
            self._relay_raw(status, headers, body)
            return

        upstream = self._decode_json(body)
        self._send_json(status, self._chat_to_response(upstream, payload))

    def _handle_anthropic_messages(self) -> None:
        payload = self._read_json_body()
        stream = bool(payload.get("stream"))
        chat_payload = self._anthropic_to_chat_payload(payload)

        status, headers, body = self._send_upstream(
            "POST",
            "/chat/completions",
            json.dumps(chat_payload).encode("utf-8"),
            {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        )

        if stream:
            self._relay_raw(status, headers, body)
            return

        upstream = self._decode_json(body)
        self._send_json(status, self._chat_to_anthropic_message(upstream, payload))

    def _proxy_passthrough(self) -> None:
        if not self.api_key:
            self._send_json(500, {"error": {"message": "NVIDIA_API_KEY is not defined"}})
            return

        target_path = self._normalized_upstream_path()
        body = self._read_request_body()
        headers = self._build_passthrough_headers()

        try:
            status, response_headers, response_body = self._send_upstream(
                self.command,
                target_path,
                body,
                headers,
                include_query=True,
            )
            self._relay_raw(status, response_headers, response_body)
        except urllib.error.HTTPError as error:
            self._relay_http_error(error)
        except Exception as error:  # pragma: no cover
            self._send_json(502, {"error": {"message": f"Upstream proxy error: {error}"}})

    def _responses_to_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._openai_input_to_messages(payload.get("input"))
        if not messages:
            raise ProxyError(400, "Responses requests must include input")

        chat_payload = self._common_chat_payload(payload)
        chat_payload["messages"] = messages
        return chat_payload

    def _anthropic_to_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = []
        system = payload.get("system")
        if system:
            messages.append({"role": "system", "content": self._content_to_text(system)})

        for message in payload.get("messages", []):
            role = message.get("role", "user")
            messages.append({"role": role, "content": self._content_to_text(message.get("content", ""))})

        if not messages:
            raise ProxyError(400, "Anthropic messages requests must include messages")

        return {
            "model": payload.get("model") or self.default_model,
            "messages": messages,
            "temperature": payload.get("temperature", 1),
            "top_p": payload.get("top_p", 1),
            "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens", 4096)),
            "stream": bool(payload.get("stream")),
        }

    def _common_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        chat_payload: dict[str, Any] = {
            "model": payload.get("model") or self.default_model,
            "temperature": payload.get("temperature", 1),
            "top_p": payload.get("top_p", 1),
            "stream": bool(payload.get("stream")),
        }

        max_tokens = payload.get("max_output_tokens", payload.get("max_tokens"))
        if max_tokens is not None:
            chat_payload["max_tokens"] = max_tokens

        for name in ("stop", "presence_penalty", "frequency_penalty", "seed"):
            if name in payload:
                chat_payload[name] = payload[name]

        return chat_payload

    def _openai_input_to_messages(self, value: Any) -> list[dict[str, str]]:
        if isinstance(value, str):
            return [{"role": "user", "content": value}]

        if not isinstance(value, list):
            return []

        messages = []
        for item in value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue

            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content", item.get("text", ""))
            messages.append({"role": role, "content": self._content_to_text(content)})

        return messages

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content)

    def _chat_to_response(self, upstream: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
        choice = self._first_choice(upstream)
        text = self._choice_text(choice)
        created = upstream.get("created") or int(time.time())
        response_id = upstream.get("id") or f"resp_{uuid.uuid4().hex}"

        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": upstream.get("model") or original.get("model") or self.default_model,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "output_text": text,
            "usage": upstream.get("usage"),
        }

    def _chat_to_anthropic_message(self, upstream: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
        choice = self._first_choice(upstream)
        text = self._choice_text(choice)
        stop_reason = "end_turn"
        if choice.get("finish_reason") == "length":
            stop_reason = "max_tokens"

        return {
            "id": upstream.get("id") or f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": upstream.get("model") or original.get("model") or self.default_model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": self._anthropic_usage(upstream.get("usage") or {}),
        }

    def _first_choice(self, upstream: dict[str, Any]) -> dict[str, Any]:
        choices = upstream.get("choices") or []
        return choices[0] if choices else {}

    def _choice_text(self, choice: dict[str, Any]) -> str:
        message = choice.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return ""

    def _anthropic_usage(self, usage: dict[str, Any]) -> dict[str, int]:
        return {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        }

    def _send_upstream(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: dict[str, str],
        include_query: bool = False,
    ) -> tuple[int, Any, bytes]:
        if not self.api_key:
            raise ProxyError(500, "NVIDIA_API_KEY is not defined")

        if include_query:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.query:
                path = f"{path}?{parsed.query}"

        url = self.upstream_base + path
        request_headers = {"Authorization": f"Bearer {self.api_key}", **headers}
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)

        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.status, response.headers, response.read()

    def _normalized_upstream_path(self) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path.startswith("/v1/"):
            path = path[3:]
        return path or "/"

    def _read_json_body(self) -> dict[str, Any]:
        body = self._read_request_body()
        if not body:
            return {}
        decoded = json.loads(body.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ProxyError(400, "Request JSON body must be an object")
        return decoded

    def _read_request_body(self) -> Optional[bytes]:
        content_length = self.headers.get("Content-Length")
        if not content_length:
            return None

        try:
            length = int(content_length)
        except ValueError:
            return None

        return self.rfile.read(length) if length > 0 else None

    def _build_passthrough_headers(self) -> dict[str, str]:
        headers = {"Accept": self.headers.get("Accept", "application/json")}

        for header_name, header_value in self.headers.items():
            header_name_lower = header_name.lower()
            if header_name_lower in {
                "host",
                "authorization",
                "content-length",
                "accept-encoding",
                "connection",
            }:
                continue

            headers[header_name] = header_value

        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]

        return headers

    def _decode_json(self, body: bytes) -> dict[str, Any]:
        if not body:
            return {}
        decoded = json.loads(body.decode("utf-8"))
        if isinstance(decoded, dict):
            return decoded
        return {}

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _relay_raw(self, status: int, upstream_headers: Any, body: bytes) -> None:
        self.send_response(status)
        self._send_cors()

        for header_name, header_value in upstream_headers.items():
            name_lower = header_name.lower()
            if name_lower in {
                "transfer-encoding",
                "connection",
                "content-length",
                "date",
                "server",
                "content-encoding",
            }:
                continue
            self.send_header(header_name, header_value)

        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _relay_http_error(self, error: urllib.error.HTTPError) -> None:
        body = error.read() or b""
        self.send_response(error.code)
        self._send_cors()
        self.send_header("Content-Type", error.headers.get_content_type() or "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[nim-proxy] {self.address_string()} - {format % args}\n")


def make_handler(config: ProxyConfig) -> type[NvidiaNIMProxyHandler]:
    """Build a request handler class bound to a specific runtime config."""

    class ConfiguredNvidiaNIMProxyHandler(NvidiaNIMProxyHandler):
        upstream_base = config.upstream_base.rstrip("/")
        api_key = config.api_key
        default_model = config.default_model
        timeout_seconds = config.timeout_seconds

    return ConfiguredNvidiaNIMProxyHandler


def create_server(config: ProxyConfig) -> ThreadingHTTPServer:
    """Create a configured server without starting it."""

    return ThreadingHTTPServer((config.host, config.port), make_handler(config))


def run_server(config: ProxyConfig) -> int:
    """Run the proxy server until interrupted."""

    with create_server(config) as httpd:
        print(
            f"NVIDIA NIM compatibility proxy listening on http://{config.host}:{config.port} "
            f"-> {config.upstream_base.rstrip('/')}"
        )
        print(f"Default model: {config.default_model}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down proxy")
            return 0

    return 0
