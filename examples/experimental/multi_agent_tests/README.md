# Multi-Agent Tests

This directory contains test scenarios for Sotopia's multi-agent support (3+ agents). The multi-agent system extends the original 2-agent framework to support complex negotiations and interactions between multiple participants.

## Features

- **3+ Agent Support**: Run scenarios with seller + multiple buyers, group discussions, etc.
- **Dynamic Agent Management**: Automatically handles N agents without hardcoded limits
- **Backward Compatibility**: Still supports 2-agent scenarios seamlessly
- **Type Safety**: Full mypy compliance with proper inheritance patterns

## Prerequisites

- Redis installed and running
- OpenAI API key (if not using local models)
- Python 3.11+ with sotopia installed

## Setup

1. **Copy Redis data from negotiation_arena** (if needed):
    ```bash
    # From the sotopia root directory
    cp -r examples/experimental/negotiation_arena/redis-data examples/experimental/multi_agent_tests/
    ```

2. **Start Redis with the data**:
    ```bash
    # From this directory (examples/experimental/multi_agent_tests/)
    redis-stack-server --dir ./redis-data
    ```

3. **Set up your API key** (optional, for OpenAI models):
    ```bash
    # Option 1: Environment variable
    export OPENAI_API_KEY="your-api-key-here"

    # Option 2: Create openai_api.txt in your home directory
    echo "your-api-key-here" > ~/openai_api.txt
    export OPENAI_API_KEY=$(cat ~/openai_api.txt)
    ```

## Running Multi-Agent Tests

### 3-Agent Auction Negotiation
```bash
python multi_agent_negotiation_test.py
```

This scenario features:
- **1 Seller**: Antique dealer wanting to maximize profit
- **2 Buyers**: Collector vs. Reseller with different strategies and budgets
- **Competitive Dynamics**: Buyers can compete or collaborate
- **Real-time Negotiation**: Multi-turn conversation with strategic decisions

### Expected Behavior

The multi-agent system will:
1. **Initialize** all 3 agents with their profiles and goals
2. **Create observations** for each agent with their perspective
3. **Manage turns** allowing each agent to participate in the conversation
4. **Handle evaluation** with multi-agent scoring and analysis
5. **Generate terminal evaluation** considering all participants' performance

## Technical Details

### Architecture
- **MultiAgentSotopiaEnv**: Extends ParallelSotopiaEnv for 3+ agents
- **MultiAgentBackground**: Inherits from ScriptBackground with N-agent support
- **UniformSampler**: Enhanced to work with multi-agent environments
- **Proper Typing**: No type ignores, uses proper inheritance patterns

### Key Differences from 2-Agent System
- Supports variable number of agents (not limited to agent_1, agent_2)
- Dynamic goal hiding (agents see their own goals, others marked as "Unknown")
- Multi-agent evaluation pipeline
- Scalable conversation management

## Troubleshooting

**Redis Connection Issues**:
```bash
# Make sure Redis is running on the correct port
redis-cli ping  # Should return "PONG"
```

**Import Errors**:
```bash
# Make sure you're in the sotopia root directory
cd /path/to/sotopia
python -m examples.experimental.multi_agent_tests.multi_agent_negotiation_test
```

**Agent Profile Issues**:
The test will automatically create agent profiles if they don't exist. If you see "Agent not found, creating new agent profile", this is normal.

## Adding New Multi-Agent Tests

To create a new multi-agent test:

1. **Define your agents** with unique names and profiles
2. **Create the scenario** with clear multi-agent dynamics
3. **Set agent goals** that create interesting interactions
4. **Use the sampler** with 3+ agents in the agent_candidates list
5. **Configure models** for each agent in the model_dict

Example structure:
```python
# Your agents
agents = [agent1, agent2, agent3, ...]  # 3 or more

# Sampler setup
sampler = UniformSampler[Observation, AgentAction](
    env_candidates=[your_env],
    agent_candidates=agents
)

# Model configuration
model_dict = {
    "env": "gpt-4o",
    "agent1": "gpt-4o",
    "agent2": "gpt-4o",
    "agent3": "gpt-4o",
    # Add more agents as needed...
}
```

## Contributing

When adding new multi-agent tests:
- Follow the existing naming convention
- Include proper documentation
- Test with different numbers of agents (3, 4, 5+)
- Ensure type safety (no type ignores)
- Add relevant comments explaining the multi-agent dynamics

Happy multi-agent testing! 🤖🤖🤖
