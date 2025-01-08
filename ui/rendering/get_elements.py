import requests
import streamlit as st
from typing import Any
from .render_utils import get_full_name
from sotopia.database import BaseCustomEvaluationDimension


def get_models() -> dict[str, dict[Any, Any]]:
    # use synchronous code to get the agents
    with requests.get(f"{st.session_state.API_BASE}/models") as resp:
        models = resp.json()
    return {model: model for model in models}


def get_scenarios() -> dict[str, dict[Any, Any]]:
    # use synchronous code to get the scenarios
    with requests.get(f"{st.session_state.API_BASE}/scenarios") as resp:
        scenarios = resp.json()
    return {scenario["codename"]: scenario for scenario in scenarios}


def get_agents(id: str = "") -> dict[str, dict[Any, Any]]:
    # use synchronous code to get the agents
    if id:
        with requests.get(f"{st.session_state.API_BASE}/agents/id/{id}") as resp:
            agents = resp.json()
    else:
        with requests.get(f"{st.session_state.API_BASE}/agents") as resp:
            agents = resp.json()
    return {get_full_name(agent): agent for agent in agents}


def get_evaluation_dimensions() -> dict[str, list[BaseCustomEvaluationDimension]]:
    with requests.get(f"{st.session_state.API_BASE}/evaluation_dimensions") as resp:
        evaluation_dimensions = resp.json()
    return {
        dimension_list_name: [
            BaseCustomEvaluationDimension(**dimension) for dimension in dimension_list
        ]
        for dimension_list_name, dimension_list in evaluation_dimensions.items()
    }


def get_distinct_evaluation_dimensions() -> list[BaseCustomEvaluationDimension]:
    all_dimension_lists: dict[str, list[BaseCustomEvaluationDimension]] = (
        get_evaluation_dimensions()
    )
    distinct_dimensions: list[BaseCustomEvaluationDimension] = []
    distinct_dimension_names: set[str] = set()

    for dimension_list_name, dimensions in all_dimension_lists.items():
        for dimension in dimensions:
            if dimension.name not in distinct_dimension_names:
                distinct_dimensions.append(dimension)
                distinct_dimension_names.add(dimension.name)

    return distinct_dimensions
