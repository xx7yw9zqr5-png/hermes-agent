"""A proxy's upstream quota failure must move to the next configured fallback."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_503_openrouter_quota_reaches_nvidia_without_network():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:20128/v1",
                        provider="custom", model="OC-Provider", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.save_trajectories = False
    agent._fallback_chain = [
        {"provider": "9router", "model": "OPENROUTER-Provider"},
        {"provider": "9router", "model": "NVIDIA-Provider"},
    ]
    agent._fallback_index = 0
    calls = []

    def completion(**kwargs):
        calls.append(agent.model)
        if agent.model == "OC-Provider":
            error = Exception("Too many requests")
            error.status_code = 429
            error.body = {}
            raise error
        if agent.model == "OPENROUTER-Provider":
            error = Exception("Service Unavailable")
            error.status_code = 503
            error.body = {"error": {"previous_errors": [{"status": 402}],
                                    "message": "requires more credits, or fewer max_tokens"}}
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="NVIDIA response", tool_calls=None,
                                                              reasoning_content=None), finish_reason="stop")],
            model=agent.model, usage=None,
        )

    agent.client.chat.completions.create.side_effect = completion

    def activate(*, reason=None, reset_at=None):
        assert agent._fallback_index < len(agent._fallback_chain)
        agent.model = agent._fallback_chain[agent._fallback_index]["model"]
        agent._fallback_index += 1
        agent._fallback_activated = True
        return True

    with (
        patch.object(agent, "_try_activate_fallback", side_effect=activate),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("answer me")

    assert calls == ["OC-Provider", "OPENROUTER-Provider", "NVIDIA-Provider"]
    assert result["completed"] is True
    assert result["final_response"] == "NVIDIA response"
