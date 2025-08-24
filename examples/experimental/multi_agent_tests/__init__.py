"""
Multi-agent test scenarios for Sotopia.

This package contains test cases and examples for running negotiations
and interactions with 3+ agents using the MultiAgentSotopiaEnv.
"""

from .constants import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_URL,
    STANDARD_MODELS,
    FAST_MODELS,
    AUCTION_SCENARIO,
    GROUP_DECISION_SCENARIO,
    RESOURCE_SHARING_SCENARIO,
)

__all__ = [
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_URL",
    "STANDARD_MODELS",
    "FAST_MODELS",
    "AUCTION_SCENARIO",
    "GROUP_DECISION_SCENARIO",
    "RESOURCE_SHARING_SCENARIO",
]
