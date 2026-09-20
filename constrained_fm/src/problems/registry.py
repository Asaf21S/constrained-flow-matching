# -*- coding: utf-8 -*-
"""Name -> Problem factory lookup, so a config selects a dataset by string."""

from __future__ import annotations

from typing import Any, Callable

from constrained_fm.src.problems.base import Problem

_PROBLEMS: dict[str, Callable[..., Problem]] = {}


def register_problem(name: str, factory: Callable[..., Problem]) -> None:
    if name in _PROBLEMS:
        raise ValueError(f"problem '{name}' is already registered")
    _PROBLEMS[name] = factory


def available_problems() -> tuple[str, ...]:
    return tuple(sorted(_PROBLEMS))


def get_problem(name: str, params: dict[str, Any] | None = None) -> Problem:
    if name not in _PROBLEMS:
        raise KeyError(f"unknown problem '{name}'; registered: {available_problems()}")
    return _PROBLEMS[name](**(params or {}))


__all__ = ["register_problem", "available_problems", "get_problem"]
