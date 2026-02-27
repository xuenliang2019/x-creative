"""Tests for Talker constraint prompt injection."""

import pytest

from x_creative.core.types import ConstraintSpec, ProblemFrame
from x_creative.saga.belief import BeliefState
from x_creative.saga.talker import Talker


class _DummyCompletion:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_reason = "stop"
        self.model = "dummy/model"


class _CaptureRouter:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_messages = []

    async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001, ANN002, ARG002
        self.last_messages = messages
        return _DummyCompletion(self._content)


@pytest.mark.asyncio
async def test_talker_injects_user_constraints_into_prompt() -> None:
    router = _CaptureRouter("ok")
    talker = Talker(router=router)
    belief = BeliefState()
    problem = ProblemFrame(
        description="Test problem",
        target_domain="general",
        structured_constraints=[
            ConstraintSpec(text="must do A", origin="user", type="hard", priority="critical"),
            ConstraintSpec(text="must do B", origin="user", type="hard", priority="critical"),
        ],
    )

    await talker.generate(belief, problem)

    joined = "\n".join(str(message.get("content", "")) for message in router.last_messages)
    assert "C1" in joined
    assert "C2" in joined
    assert "must do A" in joined
    assert "must do B" in joined

