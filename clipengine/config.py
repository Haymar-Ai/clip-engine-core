"""Engine configuration: the defaults in config.example.json, then an optional user file, then
environment variables.

  load()        the merged config (dict), read once per process and cached
  init(path)    pick the user file for this process and the processes it starts (the render
                runs in a child process and reads the same file through CLIPENGINE_CONFIG)

The user file is JSON with the same shape as config.example.json. It is merged over the defaults
key by key: a nested object merges into the default object, any other value replaces it, so a
user file only names what it changes.

Environment:
  CLIPENGINE_CONFIG       path of the user file
  CLIPENGINE_SCRATCH_DIR  replaces scratch_dir (transcript, second-ASR and topic-map caches,
                          working renders)
  CLIPENGINE_FONT_DIR     replaces render.font_dir (where the caption font is looked up first)

scratch_dir may be relative; it resolves against the working directory of the process that
first loads the config, and the resolved path is what children inherit.
"""
import copy
import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DEFAULTS_PATH = ROOT / "config.example.json"
ENV_FILE = "CLIPENGINE_CONFIG"
ENV_SCRATCH = "CLIPENGINE_SCRATCH_DIR"
ENV_FONT_DIR = "CLIPENGINE_FONT_DIR"
DEFAULT_SCRATCH = ".cache"

_CACHE = {}


class ConfigError(ValueError):
    """A user config file that cannot be read or is not a JSON object."""


def merge(base, over):
    """`over` merged into a copy of `base`: objects merge key by key, other values replace."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path, required):
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text())
    except OSError as e:
        if not required:
            return {}
        raise ConfigError(f"config file {p} unreadable ({type(e).__name__})") from None
    except ValueError as e:
        raise ConfigError(f"config file {p} is not valid JSON ({e})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must hold a JSON object")
    return data


def _key():
    return tuple(os.environ.get(k) or "" for k in (ENV_FILE, ENV_SCRATCH, ENV_FONT_DIR))


def load():
    """The merged config. Cached per (user file, scratch dir, font dir) environment."""
    key = _key()
    if key in _CACHE:
        return _CACHE[key]
    cfg = _read(DEFAULTS_PATH, required=False)
    user = os.environ.get(ENV_FILE)
    if user:
        cfg = merge(cfg, _read(user, required=True))
    if os.environ.get(ENV_SCRATCH):
        cfg["scratch_dir"] = os.environ[ENV_SCRATCH]
    if os.environ.get(ENV_FONT_DIR):
        cfg.setdefault("render", {})["font_dir"] = os.environ[ENV_FONT_DIR]
    scratch = Path(str(cfg.get("scratch_dir") or DEFAULT_SCRATCH)).expanduser()
    cfg["scratch_dir"] = str(scratch if scratch.is_absolute() else (Path.cwd() / scratch))
    _CACHE[key] = cfg
    return cfg


def init(path=None, scratch_dir=None):
    """Use `path` as the user file (and `scratch_dir`, when given) for this process and its
    children; returns the merged config. Raises ConfigError for an unreadable file."""
    if path:
        p = Path(path).expanduser().resolve()
        _read(p, required=True)
        os.environ[ENV_FILE] = str(p)
    if scratch_dir:
        os.environ[ENV_SCRATCH] = str(Path(scratch_dir).expanduser().resolve())
    cfg = load()
    # children (the render process) resolve the same scratch dir whatever their cwd
    os.environ[ENV_SCRATCH] = cfg["scratch_dir"]
    _CACHE.clear()
    return load()


def scratch_dir():
    return Path(load()["scratch_dir"])
