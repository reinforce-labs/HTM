from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import dspy
import numpy as np
from tqdm import tqdm

from data_gen.judge import _load_dotenv
from data_gen.prompts.prompts import fence
from data_gen.prompts.signals import LABELED_SIGNALS
from htm.adapter import Adapter
from htm.config import load_config as load_htm_config
from htm.policy import Policy
from htm.trust import TrustState
from signal_layer.config import Config, LMCfg
from signal_layer.data import Corpus

Level = Literal["0", "1", "2", "3", "4"]

LEVEL_SCALE = 4.0   # judge scale; /4 maps onto the [0, 1] signal range the controller expects

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# ---- endpoints --------------------------------------------------------------------------

def _api_key(cfg: LMCfg) -> str:
    _load_dotenv()
    key = os.getenv(cfg.api_key_env, "")
    if key:
        return key
    if "localhost" in cfg.api_base or "127.0.0.1" in cfg.api_base:
        return "local"      # vLLM / llama.cpp ignore the key
    raise ValueError(f"{cfg.api_key_env} is not set (add it to .env or the environment).")


def build_lm(cfg: LMCfg, cls: type = dspy.LM) -> dspy.BaseLM:
    if cfg.local:   # in-process transformers on this machine's GPU (classifiers pattern, no server)
        from signal_layer.student import LocalLM
        return LocalLM(cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    return cls(     # cls lets the reflection path swap in a logging subclass (pipeline._LoggingLM)
        cfg.model,
        api_base=cfg.api_base,
        api_key=_api_key(cfg),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ---- seed prompt --------------------------------------------------------------------------

def seed_instructions(summary_memory: bool = False) -> str:
    """The starting prompt GEPA evolves — one self-contained file (rubric anchors,
    memory protocol, output rules), no dependence on data_gen's teacher prompt.
    Summary mode starts from the seed that describes the rolling-summary protocol."""
    name = "student_seed_summary.txt" if summary_memory else "student_seed.txt"
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


# ---- trust-in-context ----------------------------------------------------------------------

class _TrustReadout:
    """Runs the frozen trust dynamics incrementally over the student's OWN scores, so each
    turn's context can carry the controller's current state. Same components and config as
    htm.Controller — one instance per conversation (TrustState/Policy are stateful)."""

    HISTORY_K = None    # history mode: None = the entire trust trajectory (convs are short, ~7-40
                        # turns, so this stays small); set an int k to show only the last k values

    def __init__(self, history: bool = False) -> None:
        cfg = load_htm_config()
        self._adapter = Adapter(list(LABELED_SIGNALS), cfg.adapter, cfg.caution)
        self._trust = TrustState(memory_n=cfg.trust.memory_n, u_max=cfg.trust.u_max)
        self._policy = Policy(bands=cfg.policy.bands, deescalation_margin=cfg.policy.deescalation_margin)
        self._action = "open"                   # Policy starts fully open
        self._history = history
        self._values = [self._trust.trust]      # trust after each scored turn, oldest first

    def line(self) -> str:
        # The student's whole memory of prior turns, framed as a credit score. History mode
        # adds the trajectory so the SHAPE (cliff / drift / recovery / early-flag-then-return)
        # is visible — the scalar collapses different histories onto the same number.
        if self._history:
            values = self._values if self.HISTORY_K is None else self._values[-self.HISTORY_K:]
            history = " -> ".join(f"{v:.2f}" for v in values)
            return f"TRUST STATE: trust={self._trust.trust:.2f} history={history}"
        return f"TRUST STATE: trust={self._trust.trust:.2f}"

    def update(self, scores: list[float]) -> None:
        caution = float(self._adapter.caution(np.asarray([scores], dtype=np.float64))[0])
        update = self._trust.step(caution)
        self._action = self._policy.step(update.control).value
        self._values.append(self._trust.trust)


# ---- the dspy program -------------------------------------------------------------------

class ScoreTurn(dspy.Signature):
    """Placeholder — the evolving instructions are attached via with_instructions()."""

    trust: str = dspy.InputField(desc="TRUST STATE line — the trust controller's numeric summary of prior turns — or '(none)'")
    new_turn: str = dspy.InputField(desc="the current turn: user message plus assistant reply, if any")

    harmful_intent_alignment: Level = dspy.OutputField()
    refusal_persistence: Level = dspy.OutputField()
    contextual_reframing: Level = dspy.OutputField()
    intent_concealment: Level = dspy.OutputField()
    harmful_compliance: Level = dspy.OutputField()
    task_legitimacy: Level = dspy.OutputField()


class ScoreTurnWithMemory(dspy.Signature):
    """Summary-memory variant: the student's memory of prior turns is its OWN one-line
    factual case notes, updated every turn (replaces the trust line entirely).
    updated_memory comes FIRST: a ~1.7B model emits the six enums and stops, dropping any
    trailing free-text field — notes-then-scores makes the field unskippable (and is the
    summarize-then-judge order anyway)."""

    memory: str = dspy.InputField(desc="your case notes from earlier turns — one line of facts — or '(none)' on the first turn")
    new_turn: str = dspy.InputField(desc="the current turn: user message plus assistant reply, if any")

    updated_memory: str = dspy.OutputField(desc="FIRST: the case notes carried to the next turn — facts only, under 30 words")
    harmful_intent_alignment: Level = dspy.OutputField()
    refusal_persistence: Level = dspy.OutputField()
    contextual_reframing: Level = dspy.OutputField()
    intent_concealment: Level = dspy.OutputField()
    harmful_compliance: Level = dspy.OutputField()
    task_legitimacy: Level = dspy.OutputField()


def _render_turn(t: int, turn: dict, user_only: bool = False) -> str:
    if user_only:
        return f"TURN {t}:\n" + fence(f"USER: {turn['user'] or '(none)'}")
    return f"TURN {t}:\n" + fence(f"USER: {turn['user'] or '(none)'}\nASSISTANT: {turn.get('assistant') or '(none)'}")


def _level(value: str) -> float:
    return min(max(int(value), 0), 4) / LEVEL_SCALE


class _SignalProgram(dspy.Module):
    """One evolved prompt applied per turn: conversation record -> [T, 6] signal rows.
    The student sees ONLY the current turn plus the trust state (its sole memory of
    prior turns); include_trust=False is the isolated-turn ablation. This is the object
    dspy.GEPA compiles; SignalScorer is its owner."""

    progress = False           # per-turn tqdm; leave off during GEPA (threaded rollouts interleave bars)
    checkpoint_dir: Path | None = None   # set by Optimization; captures each candidate GEPA evaluates

    MEMORY_MAX_CHARS = 300      # defensive cap: a runaway summary can't grow the prompt

    def __init__(self, include_trust: bool, trust_history: bool = False, summary_memory: bool = False,
                user_only: bool = False) -> None:
        super().__init__()
        self.include_trust = include_trust and not summary_memory   # summary REPLACES the trust line
        self.trust_history = trust_history
        self.summary_memory = summary_memory
        self.user_only = user_only          # drop the assistant reply from the rendered turn
        self._signature = ScoreTurnWithMemory if summary_memory else ScoreTurn
        self.step = dspy.Predict(self._signature.with_instructions(seed_instructions(summary_memory)))

    def set_instructions(self, text: str) -> None:
        """Swap the per-turn scoring instructions — used to load a .txt prompt variant."""
        self.step = dspy.Predict(self._signature.with_instructions(text))

    def forward(self, conv: dict) -> dspy.Prediction:
        if self.checkpoint_dir is not None:
            self._maybe_checkpoint()

        readout = _TrustReadout(history=self.trust_history) if self.include_trust else None
        memory = "(none)"                   # summary mode: the student's own rolling case notes
        memories: list[str] = []
        matrix: list[list[float]] = []
        turns: list[dict] = []
        conv_id = conv.get("conv_id", "?")
        n = len(conv["turns"])

        for t, turn in enumerate(tqdm(conv["turns"], desc="turns", unit="turn", disable=not self.progress, leave=False)):
            try:
                if self.summary_memory:
                    out = self.step(memory=memory, new_turn=_render_turn(t, turn, self.user_only))
                    updated = " ".join((out.updated_memory or "").split())          # collapse whitespace
                    memory = updated[:self.MEMORY_MAX_CHARS] or memory              # blank -> carry forward
                else:
                    trust_line = readout.line() if readout is not None else "(none)"
                    out = self.step(trust=trust_line, new_turn=_render_turn(t, turn, self.user_only))
                scores = [_level(getattr(out, name)) for name in LABELED_SIGNALS]
                turns.append({"valid": True, "error": ""})
                if self.progress:   # per-turn "ok" spam floods parallel eval; keep it opt-in (inspect only)
                    tqdm.write(f"[score] {conv_id} turn {t + 1}/{n} ok", file=sys.stdout)
            except Exception as exc:  # a failed turn scores neutral; never kill the trajectory.
                scores = [0.0] * len(LABELED_SIGNALS)   # summary mode: memory carries forward unchanged —
                turns.append({"valid": False, "error": str(exc)})   # never reset to "(none)" mid-conversation
                tqdm.write(f"[score] {conv_id} turn {t + 1}/{n} FAILED: {exc}", file=sys.stdout)

            matrix.append(scores)
            if self.summary_memory:
                memories.append(memory)     # the note the NEXT turn will see
            if readout is not None:
                readout.update(scores)      # this turn's scores shape the next turn's context

        return dspy.Prediction(matrix=matrix, turns=turns, memory=memories if self.summary_memory else None)

    def _maybe_checkpoint(self) -> None:
        """Save the current candidate iff its instructions differ from what's on disk. Dedup is
        via the on-disk .txt (robust across the program copies GEPA makes per candidate). Writes a
        numbered snapshot for history plus a stable `latest_candidate` pointer; dspy's gepa_state/
        keeps the authoritative full set with scores."""
        current = self.step.signature.instructions
        latest = self.checkpoint_dir / "latest_candidate.txt"
        if latest.exists() and latest.read_text(encoding="utf-8") == current:
            return   # same candidate still being evaluated — nothing new to save

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(self.checkpoint_dir.glob("candidate_*.txt"))) + 1
        (self.checkpoint_dir / f"candidate_{n:03d}.txt").write_text(current, encoding="utf-8")
        latest.write_text(current, encoding="utf-8")
        self.save(self.checkpoint_dir / "latest_candidate.json")
        tqdm.write(f"[checkpoint] candidate #{n} captured -> {self.checkpoint_dir}/candidate_{n:03d}.txt", file=sys.stdout)


# ---- public surface -----------------------------------------------------------------------

@dataclass(frozen=True)
class TurnScore:
    valid: bool
    error: str = ""     # set when the turn failed to parse and scored neutral


@dataclass(frozen=True)
class ScoredConversation:
    matrix: np.ndarray          # [T, 6] in [0, 1], columns ordered by LABELED_SIGNALS
    turns: list[TurnScore]
    memory: list[str] | None = None     # summary mode: the case notes after each turn (what turn t+1 saw)


class SignalScorer:
    """Owns the student LM, the per-turn program, and the prompt artifact. Constructing
    one routes dspy's global LM/adapter to the student (JSONAdapter: the server's
    structured outputs keep a ~1.5B model's enum JSON valid)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        dspy.configure(lm=build_lm(cfg.student), adapter=dspy.JSONAdapter())
        self.program = _SignalProgram(cfg.include_trust, cfg.trust_history, cfg.summary_memory, cfg.user_only)    # seed instructions

    @classmethod
    def from_artifact(cls, cfg: Config, artifact: Path | str | None = None) -> "SignalScorer":
        """Scorer with the evolved instructions from `artifact` (or cfg.artifact by default).
        An explicitly-passed path that doesn't exist falls back to the seed (this is how the
        seed/GEPA-off ablation is requested); the configured default silently missing is almost
        always a mistake, so that case raises instead of quietly scoring with the seed prompt."""
        scorer = cls(cfg)
        path = Path(artifact) if artifact else cfg.artifact
        if path.exists():
            if path.suffix == ".txt":       # a raw prompt variant (e.g. the no-trust twin), not dspy state
                scorer.program.set_instructions(path.read_text(encoding="utf-8").strip())
            else:
                scorer.program.load(path)
        elif artifact is None:
            raise FileNotFoundError(
                f"configured artifact {cfg.artifact} does not exist -- run `--stage optimize` first, "
                f"or pass --artifact seed to explicitly evaluate the unoptimized seed prompt"
            )
        return scorer

    def score(self, conv: dict) -> ScoredConversation:
        pred = self.program(conv=conv)
        return ScoredConversation(
            matrix=np.asarray(pred.matrix, dtype=np.float64),
            turns=[TurnScore(**t) for t in pred.turns],
            memory=getattr(pred, "memory", None),
        )

    def save(self, path: Path | str | None = None) -> Path:
        """Write the program state (loadable) plus a readable .txt of the evolved instructions."""
        path = Path(path) if path else self.cfg.artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        self.program.save(path)
        path.with_suffix(".txt").write_text(self.program.step.signature.instructions, encoding="utf-8")
        return path


class SignalsSourceLike(Protocol):
    """The seam eval code is written against; any signal source (e.g. LLMSignalsSource)
    satisfies it."""

    signal_names: list[str]

    def signals(self, dataset: str, conv_id: str, user_only: bool = False) -> np.ndarray: ...


class LLMSignalsSource:
    """The signal source backed by the evolved student.
    Contract: signals() -> [T, S] in [0, 1], columns ordered by signal_names. Reads
    conversations directly (no embedding cache). Whether the student sees the assistant's
    reply at all is fixed at construction via cfg.user_only (see _SignalProgram); the
    per-call `user_only` argument below is accepted for interface parity and ignored."""

    def __init__(self, cfg: Config, artifact: Path | str | None = None) -> None:
        self.cfg = cfg
        self.signal_names = list(LABELED_SIGNALS)
        self._scorer = SignalScorer.from_artifact(cfg, artifact)
        self._corpus = Corpus(cfg.data)
        self.last_memory: list[str] | None = None   # summary mode: case notes of the last scored conv
                                                    # (run_eval persists them into the run JSON)

    def signals(self, dataset: str, conv_id: str, user_only: bool = False) -> np.ndarray:
        scored = self._scorer.score(self._corpus.conversation(dataset, conv_id))
        self.last_memory = scored.memory
        return scored.matrix

    def signals_with_memory(self, dataset: str, conv_id: str) -> tuple[np.ndarray, list[str] | None]:
        """Per-conversation scoring for the parallel eval path: returns (matrix, memory) without
        mutating shared state (unlike signals(), which stashes last_memory on self and would race
        across threads). Safe to call concurrently once warm() has pre-loaded the corpus cache."""
        scored = self._scorer.score(self._corpus.conversation(dataset, conv_id))
        return scored.matrix, scored.memory

    def warm(self, datasets) -> None:
        """Pre-load each dataset's conversation records into the corpus cache single-threaded, so
        concurrent scoring threads don't race on (or redundantly repeat) the lazy first load."""
        for ds in datasets:
            self._corpus.conversation(ds)

    def release(self) -> None:
        self._corpus.clear()
