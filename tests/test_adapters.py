import unittest

from nvidia_nim_proxy.server import NvidiaNIMProxyHandler


class AdapterTests(unittest.TestCase):
    def test_responses_input_string_maps_to_chat_messages(self) -> None:
        handler = object.__new__(NvidiaNIMProxyHandler)
        handler.default_model = "test/model"

        payload = handler._responses_to_chat_payload({"input": "hello"})

        self.assertEqual(payload["model"], "test/model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

    def test_anthropic_messages_map_to_chat_messages(self) -> None:
        handler = object.__new__(NvidiaNIMProxyHandler)
        handler.default_model = "test/model"

        payload = handler._anthropic_to_chat_payload(
            {
                "system": "be brief",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                "max_tokens": 10,
            }
        )

        self.assertEqual(payload["model"], "test/model")
        self.assertEqual(payload["max_tokens"], 10)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
