import os

_config = None


def _load_config():
    global _config
    if _config is not None:
        return _config
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config


def setup_hf_token():
    """从 config.yaml 读取 HuggingFace token 并写入环境变量 HF_TOKEN。"""
    cfg = _load_config()
    token = cfg.get("huggingface", {}).get("token", "")
    if token:
        os.environ["HF_TOKEN"] = token


def get(key, default=None):
    """按点分隔的 key 读取配置值，如 'huggingface.token'。"""
    cfg = _load_config()
    keys = key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val
