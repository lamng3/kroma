"""Compatibility wrapper around the inference strategies."""

from kroma.agents.inference import expand_term, infer

inference = infer

__all__ = ["expand_term", "inference"]
