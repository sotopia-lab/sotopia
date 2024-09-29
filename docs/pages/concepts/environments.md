## Overview

Environment is a concept in Sotopia to orchestrating the interactions between agents. Environments provide a social context for the agents, passing messages between agents and evaluating the quality of social interactions. Environments can be customized to test different scenarios, different turn-ordering, and different ways to evaluate the social interactions.


### ParallelSotopiaEnv
The [`ParallelSotopiaEnv`](/python_API/envs/parallel) is a general environment that runs multiple agents in parallel.
The "parallel" in the name refers to the fact that the agents could run in parallel.
However, there are three kinds of ordering supported by the `ParallelSotopiaEnv`:
1. Random ordering: in each turn, a randomly chosen agent can act.
2. Round-robin ordering: in turn `i`, agent `i % n` acts, where `n` is the number of agents.
3. Simultaneous ordering: All agents take actions at the same time.

Except for 3, in each turn one of agent's `aact` method is called, while other agents' `aact` methods are not called.

#### Evaluators

There are two kinds of evaluators in the `ParallelSotopiaEnv`:

1. Response evaluators: These evaluators are called after each agent's action.
2. Terminal evaluators: These evaluators are called after the environment reaches a terminal state.

The first kind of evaluator determines whether the interaction should terminate, and the second kind of evaluator determines the final reward for each agent.

One example of the first kind of evaluator is [`RuleBasedTerminatedEvaluator`](/python_API/envs/evaluators#rulebasedterminatedevaluator), which test if the interaction has staled for too long or if a certain number of turns has passed.
And one example of the second kind of evaluator is [`ReachGoalLLMEvaluator[T_eval_dim]`](/python_API/envs/evaluators#reachgoalllmevaluator), which evaluates the agents based on a specification of dimensions passed as the type variable `T_eval_dim`.

`T_eval_dim` is a type variable that specifies the dimensions of the evaluation, which should subclass the `pydantic.BaseModel` class. And example of it is [`SotopiaDimensions`](/python_API/envs/evaluators#sotopiadimensions), which is a class that contains the dimensions of the evaluation used in the original Sotopia paper.

#### Usage of `ParallelSotopiaEnv`

To use the `ParallelSotopiaEnv`, you need to create a list of agents and a list of evaluators, and pass them to the constructor of the `ParallelSotopiaEnv`. Then you can call the `astep` method to pass the actions of the agents to the environment.
As of now, the simplest way to use the `ParallelSotopiaEnv` is to use the `run_async_server` function in the `sotopia.server` module. Check out the `minimalist_demo` example:

```python
asyncio.run(
    run_async_server(
        model_dict={
            "env": "gpt-4",
            "agent1": "gpt-3.5-turbo",
            "agent2": "gpt-3.5-turbo",
        },
        sampler=UniformSampler(),
    )
)
```
