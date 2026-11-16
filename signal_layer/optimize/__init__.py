from pathlib import Path

_PKG = Path(__file__).resolve().parent
DATA_DIR = _PKG / "data"
MATRICES_PATH = DATA_DIR / "train_matrices.json"

_PROMPTS = _PKG.parent / "prompts"
DYNAMICS_SEED = _PROMPTS / "dynamics_seed.txt"
DYNAMICS_OPTIMIZED = _PROMPTS / "dynamics_optimized.txt"
DYNAMICS_STATE_DIR = _PROMPTS / "best_prompt" / "gepa_dynamics_state"