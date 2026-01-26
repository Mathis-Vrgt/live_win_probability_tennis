import yaml
from pathlib import Path

# Dossier src/config/
BASE_DIR = Path(__file__).resolve().parent

def load_yaml(filename):
    path = BASE_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)

TOURNAMENT_INFO = load_yaml("tournaments.yaml")
POINTS_BY_ROUND = load_yaml("points_tournaments.yaml")
