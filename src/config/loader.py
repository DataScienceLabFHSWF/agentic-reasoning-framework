from pathlib import Path
import yaml


class Config:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.root = self.config_path.parent.parent

        with open(self.config_path, "r") as f:
            self.data = yaml.safe_load(f)

    def path(self, key: str) -> Path:
        """Return absolute path for a configured path."""
        rel = self.data["paths"][key]
        return (self.root / rel).resolve()

    def get(self, section: str, key: str):
        return self.data[section][key]