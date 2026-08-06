import os
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


def resolve_project_root(fallback_file: str = None) -> str:
    """
    Resolve the nnUNet project root directory via priority chain:
    1. Environment variable NNUNET_PROJECT_ROOT
    2. Relative to the given fallback_file (5 levels up from trainer)
    3. Current working directory
    """
    env_root = os.environ.get('NNUNET_PROJECT_ROOT')
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    if fallback_file:
        candidate = os.path.abspath(os.path.join(os.path.dirname(fallback_file), '..', '..', '..', '..', '..'))
        if os.path.isdir(os.path.join(candidate, 'configs')):
            return candidate

    return os.getcwd()


def _resolve_placeholders(obj: Any, project_root: str) -> Any:
    """Recursively replace ${PROJECT_ROOT} placeholders in config values."""
    if isinstance(obj, str):
        return obj.replace('${PROJECT_ROOT}', project_root)
    elif isinstance(obj, dict):
        return {k: _resolve_placeholders(v, project_root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_placeholders(item, project_root) for item in obj]
    return obj


def load_config(config_path: str, resolve_paths: bool = True) -> ConfigDict:
    """
    Load YAML config file into dot-accessible ConfigDict.
    If resolve_paths=True, replaces ${PROJECT_ROOT} placeholders with the actual project root.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if resolve_paths:
        project_root = resolve_project_root(fallback_file=config_path)
        data = _resolve_placeholders(data, project_root)

    return ConfigDict(data)


def save_config(cfg: ConfigDict, save_path: str):
    """Save ConfigDict to YAML file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
