import torch

from pathlib import Path
import os


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(os.getcwd())

OUTPUT_ROOT = PROJECT_ROOT / "results"
CHECKPOINTS = OUTPUT_ROOT / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data"


ITER_METHODS = ["simulated_annealing", "particle_swarm"]
