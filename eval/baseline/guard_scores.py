from __future__ import annotations

from data_gen import io as data_io
from data_gen.schema import Conversation, Turn

from eval.baseline.classifiers.base import Guard, Message
from eval.baseline.classifiers.granite_guardian import GraniteGuardian
from eval.baseline.classifiers.llama_guard import LlamaGuard
from eval.baseline.classifiers.prompt_guard import PromptGuard

GUARDS = {"llama_guard": LlamaGuard, "granite": GraniteGuardian, "prompt_guard": PromptGuard}


def load_turns(dataset: str, conv_id: str) -> list[Turn]:
    """The turn text for one conversation."""
    for rec in data_io.read_jsonl(data_io.conversation_path(dataset)):
        if rec["conv_id"] == conv_id:
            return Conversation.from_record(rec).turns
    raise KeyError(f"{conv_id} not found in {data_io.conversation_path(dataset)}")


def prompt_messages(turns: list[Turn], t: int) -> list[Message]:
    """Input-filter mode: the current user turn only, no assistant reply."""
    return [{"role": "user", "content": turns[t].user or ""}]


def current_messages(turns: list[Turn], t: int) -> list[Message]:
    """The current exchange in isolation: the user turn and its assistant reply."""
    messages: list[Message] = [{"role": "user", "content": turns[t].user or ""}]
    if turns[t].assistant:
        messages.append({"role": "assistant", "content": turns[t].assistant})
    return messages


def history_messages(turns: list[Turn], t: int) -> list[Message]:
    """The full conversation through turn t, including the current assistant reply."""
    messages: list[Message] = []

    for j in range(t + 1):
        messages.append({"role": "user", "content": turns[j].user or ""})
        if turns[j].assistant:
            messages.append({"role": "assistant", "content": turns[j].assistant})

    return messages


def user_history_messages(turns: list[Turn], t: int) -> list[Message]:
    return [{"role": "user", "content": turns[j].user or ""} for j in range(t + 1)]


GUARD_VARIANTS = {
    "prompt": prompt_messages,
    "current": current_messages,
    "history": history_messages,
    "user_history": user_history_messages,
}


def score_conversation(guard: Guard, turns: list[Turn]) -> dict[str, list[float]]:
    """{variant: [T]} per-turn P(unsafe) for one conversation, one series per guard variant."""
    return {
        kind: [guard.score_messages(build(turns, t)) for t in range(len(turns))]
        for kind, build in GUARD_VARIANTS.items()
    }