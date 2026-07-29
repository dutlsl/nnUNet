import yaml
from typing import Any, Dict


class ConfigDict(dict):
    """Dictionary subclass enabling dot-notation access to attributes."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return ConfigDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_config(config_path: str) -> ConfigDict:
    """Load YAML config file into dot-accessible ConfigDict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return ConfigDict(data)


def save_config(cfg: ConfigDict, save_path: str):
    """Save ConfigDict to YAML file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
