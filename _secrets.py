# -*- coding: utf-8 -*-
"""producer/_secrets.py — 统一密钥读取入口。

优先级：环境变量 > producer/secrets.json > 代码内默认值。

secrets.json 被 .gitignore 排除，不会进入版本库；
其模板见 secrets.example.json。
"""
import json
import os

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    _CACHE = data
    return _CACHE


def get(key, default=""):
    """按 key 取值：环境变量（大写）优先，其次 secrets.json，最后默认值。"""
    env_key = key.upper()
    val = os.environ.get(env_key)
    if val:
        return val
    data = _load()
    if key in data and data[key] not in (None, ""):
        return data[key]
    return default


def get_int(key, default=0):
    try:
        return int(get(key, default))
    except (TypeError, ValueError):
        return default
