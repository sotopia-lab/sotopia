from pydantic import BaseModel
from sotopia.database import (
    EvaluationDimensionBuilder,
    CustomEvaluationDimensionList,
    CustomEvaluationDimension,
)
from typing import Type, Union
from redis_om import Migrator
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    EvaluationForTwoAgents,
    RuleBasedTerminatedEvaluator,
)
from sotopia.server import arun_one_episode
from typing import Optional, cast
from sotopia.envs import ParallelSotopiaEnv
from sotopia.agents import LLMAgent
from sotopia.database import AgentProfile, EnvironmentProfile
import asyncio


def save_dimensions(dimensions: list[dict[str, Union[str, int]]]) -> None:
    Migrator().run()
    for dimension in dimensions:
        if (
            len(
                CustomEvaluationDimension.find(
                    CustomEvaluationDimension.name == dimension["name"]
                ).all()
            )
            == 0
        ):
            print("No existing dimension found, creating a new one")
            CustomEvaluationDimension(**dimension).save()
            print("Saved {}".format(dimension["name"]))
        else:
            print(
                CustomEvaluationDimension.find(
                    CustomEvaluationDimension.name == dimension["name"]
                ).all()[0],
                "already exists",
            )


def save_dimension_list(
    dimensions: list[dict[str, Union[str, int]]], list_name: str
) -> None:
    dimension_list = CustomEvaluationDimensionList.find(
        CustomEvaluationDimensionList.name == list_name
    ).all()

    if len(dimension_list) == 0:
        all_dimensions_pks = []
        for dimension in dimensions:
            find_dimension = CustomEvaluationDimension.find(
                CustomEvaluationDimension.name == dimension["name"]
            ).all()
            assert (
                len(find_dimension) == 1
            ), f"Expected 1 dimension for {dimension['name']}, but found {len(find_dimension)}"
            all_dimensions_pks.append(find_dimension[0].pk)
        CustomEvaluationDimensionList(
            name=list_name, dimension_pks=all_dimensions_pks
        ).save()
        print("Saved {}".format(list_name))
    else:
        print(dimension_list[0], "already exists")


def build_sotopia_dimensions() -> Type[BaseModel]:
    """
    Build the default Sotopia dimensions, the same as
    `sotopia.env.evaluators.SotopiaDimensions`
    """
    sotopia_dimensions: list[dict[str, Union[str, int]]] = [
        {
            "name": "believability",
            "description": "Reasoning requirement: 1. Evaluate if the agent interacts with others in a natural and realistic manner (here are a few common questions to check: a. whether the agent is confusing with its own identity? b. whether the agent repeats others' words/actions without any reason? c. whether the agent is being overly polite considering the context?). Start the analysis with tag <naturalness> 2. Analyze whether the actions of the agent align with their character traits (e.g., personality, values, and etc.). Start the analysis with tag <consistency>. Output your reasoning process to the 'reasoning' field. Output an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates that the agent is more believable.",
            "range_high": 10,
            "range_low": 0,
        },
        {
            "name": "relationship",
            "description": "Please first analyze what relationship the participant has with the other agent(s) before the interaction. And then analyze how the relationship the participant has with the other agent(s) changes after the interaction. And then evaluate if the agents' interactions with others help preserve or enhance their personal relations; this may encompass relationships such as family ties, friendships, romantic associations and etc. Additionally, ascertain whether these interactions also impact their social status or reputation. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -5 to 5 in the 'score' field. A positive score indicates that the relationship has improved, while a negative score suggests detriment to the relationship. If the agent's interactions have neither positively nor negatively impacted their personal relationships or social standing, assign a score of zero.",
            "range_high": 5,
            "range_low": -5,
        },
        {
            "name": "knowledge",
            "description": "Please first assess what information the agent has gained through the interaction. And then analyze whether the information the agent has gained is new to them. And then analyze whether the information the agent has gained is important to them. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates that the agent has gained more new and important knowledge.",
            "range_high": 10,
            "range_low": 0,
        },
        {
            "name": "secret",
            "description": "First please analyze what secret or secretive intention the participant wants to keep. And then analyze whether they keep it successfully. Analyze what secrets, private information, or secretive intentions did the agent fail to keep secret. Please output the reasoning in the reasoning field. Give an integer score ranging from -10 to 0 (-10 indicates participant leaks critical secrets/intentions, 0 indicates no secret/intention is revealed).",
            "range_high": 0,
            "range_low": -10,
        },
        {
            "name": "social_rules",
            "description": "Please evaluate whether the agent violates any moral rules or laws in their interactions with others. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -10 and 0 in the 'score' field. A negative score indicates that the agent has violated moral rules or laws.",
            "range_high": 0,
            "range_low": -10,
        },
        {
            "name": "financial_and_material_benefits",
            "description": "Please evaluate whether the agent's interactions with others contribute towards financial and material benefits. Analyze what the agent would gain/lose after the interactions. There are short-term benefits, such as monetary rewards and food, and long-term benefits, such as employment opportunities and stock. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -5 and 5 in the 'score' field. Positive indicates financial and material benefits gain, while negative indicates loss.",
            "range_high": 5,
            "range_low": -5,
        },
        {
            "name": "goal",
            "description": "Please first reiterate agent's social goals. And then please provide a comprehensive analysis about the extent to which the agent has managed to achieve these goals. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates that the agent is making progress towards their social goals.",
            "range_high": 10,
            "range_low": 0,
        },
    ]

    dimensions = EvaluationDimensionBuilder.build_dimension_model_from_dict(
        dimensions=sotopia_dimensions
    )
    save_dimensions(sotopia_dimensions)
    save_dimension_list(sotopia_dimensions, "sotopia")

    return dimensions


def build_custom_dimensions(
    custom_dimensions: list[dict[str, Union[str, int]]], list_name: Optional[str] = None
) -> Type[BaseModel]:
    """
    Build a custom evaluation dimension model,
    : param custom_dimensions: a list of dictionaries that specify the parameters of the `CustomEvaluationDimension`.
    : param list_name: the name of the list to save the custom dimensions to. If None, no list will be saved.
    """
    dimensions = EvaluationDimensionBuilder.build_dimension_model_from_dict(
        dimensions=custom_dimensions
    )

    save_dimensions(custom_dimensions)
    if list_name is not None:
        save_dimension_list(custom_dimensions, list_name=list_name)

    return dimensions


def run_simple_sample_with_custom_samples(
    custom_dimensions: list[dict[str, Union[str, int]]],
) -> None:
    custom_dimensions_type = build_custom_dimensions(
        custom_dimensions, list_name="custom"
    )
    evaluator = RuleBasedTerminatedEvaluator(max_turn_number=10, max_stale_turn=2)
    terminal_evaluator = ReachGoalLLMEvaluator(
        model_name="gpt-4o-mini",
        response_format_class=EvaluationForTwoAgents[custom_dimensions_type],  # type: ignore
    )

    all_agents: list[AgentProfile] = cast(
        list[AgentProfile],
        AgentProfile.find().page(0, 2),  # type: ignore
    )
    all_envs: list[EnvironmentProfile] = cast(
        list[EnvironmentProfile],
        EnvironmentProfile.find().page(0, 1),  # type: ignore
    )
    environment: ParallelSotopiaEnv = ParallelSotopiaEnv(
        env_profile=all_envs[0],
        model_name="gpt-4o-mini",
        action_order="round-robin",
        evaluators=[evaluator],
        terminal_evaluators=[terminal_evaluator],
    )
    agents: list[LLMAgent] = [
        LLMAgent(agent_profile=agent_profile, model_name="gpt-4o-mini")
        for agent_profile in all_agents[:2]
    ]

    res = asyncio.run(
        arun_one_episode(
            env=environment,
            agent_list=agents,
            omniscient=False,
            script_like=False,
            tag=None,
            push_to_db=False,
        )
    )

    print(res)


if __name__ == "__main__":
    # here is a sample dimension
    custom_dimensions: list[dict[str, Union[str, int]]] = [
        {
            "name": "transactivity",
            "description": "Analyze the provided social interaction episode between the given pair/team, focusing on identifying instances of transactive exchanges. Evaluate the level of transactivity by considering the following aspects: elaboration, building upon ideas, questioning, argumentation. Analyze whether these transactive patterns persist consistently across the entire interaction or if there are notable variations throughout the exchange. In the 'reasoning' field, provide a comprehensive account of the logic and thought process that led to your conclusion. Consider how the observed instances of transactivity contribute to or detract from the overall quality and depth of the interaction. In the 'score' field, provide an integer score ranging from 0 to 10, where a higher score indicates a higher level of transactivity.",
            "range_high": 10,
            "range_low": 0,
        },
        {
            "name": "verbal_equity",
            "description": "Analyze the script and measure the level of verbal equity reflected in the interaction between the agents. And then analyze the extent to which the interaction shows a balanced distribution of speaking opportunities among team members. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates a higher level of verbal equity.",
            "range_high": 10,
            "range_low": 0,
        },
    ]

    # Only build evaluation dimensions
    build_sotopia_dimensions()
    build_custom_dimensions(custom_dimensions=custom_dimensions, list_name="custom")

    # Build and use evaluation dimensions
    run_simple_sample_with_custom_samples(custom_dimensions=custom_dimensions)
