`data_gen` generates the processed datasets and labels used throughout the project. Each stage can be run independently and resumes from previously completed work, making the pipeline incremental and fault tolerant.

## Pipeline

The data generation pipeline consists of the following stages:

1. **`acquire.py`** – Downloads each dataset, normalizes conversations into a common format, filters and deduplicates examples, assigns stable content-hash IDs, and writes `raw.jsonl`.
2. **`add_user_turns.py`** – Replays user-only conversations (currently MHJ) through a victim LLM to synthesize assistant responses, writing them to `assistant.jsonl` without modifying the original conversations.
3. **`teacher.py`** – Runs the trajectory-level judge, which processes the full conversation with a rolling summary and produces HTM signal annotations (e.g., intent, cooperation, compliance drift) together with a one-line goal and conversation summary.
4. **`step.py`** – Runs the per-turn judge, scoring each user and assistant message independently without conversational context.
5. **`payload.py`** – Identifies the payload turn for each adversarial conversation, enabling lead-time evaluation.

## Labeling

Two separate judges are used because they require different context.

- The **teacher judge** observes the entire trajectory and produces cumulative signal annotations used by HTM.
- The **step judge** evaluates each message independently to provide isolated harm labels without leaking trajectory information.

Separating these tasks allows each judge to use the most appropriate model and prompt while keeping the resulting labels conceptually distinct.

## Output format

Every generated record stores the raw model output, parsing status, model identifier, and prompt version. This preserves provenance, prevents failed parses from being silently converted into labels, and allows outputs to be revalidated or reparsed without rerunning model inference.