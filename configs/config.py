import os, yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def load_yaml_config(config_path:Path = BASE_DIR / "configs" / "config.yaml"):
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


# Global Configs
CONFIG = load_yaml_config()

# Environment Variables
DEVICE = os.getenv("DEVICE",CONFIG.get("app",{}).get("device","cpu"))
LOG_LEVEL = os.getenv("LOG_LEVEL","INFO")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR",BASE_DIR / "artifacts" / "models"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)