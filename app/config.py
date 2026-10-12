import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

def load_yaml_config(config_path: Path = BASE_DIR / "configs" / "config.yaml") -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# Global configs
CONFIG = load_yaml_config()

# Exported settings
DEVICE = os.getenv("DEVICE", CONFIG.get("app", {}).get("device", "cpu"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", BASE_DIR / "artifacts" / "models"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)