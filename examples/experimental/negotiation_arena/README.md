# Negotiation Arena

In this example, we set up a negotiation arena where agents can negotiate in multiple scenarios, such as buy/sell transactions.
We adopt the scenarios from https://github.com/vinid/NegotiationArena.

## Prerequisites

- Redis installed

## Setup

1. At the root of this folder, create the data directory for Redis:
    ```bash
    mkdir -p sotopia-data/redis-data
    ```
2. Copy your `dump.rdb` file into that folder:
    ```bash
    cp /path/to/your/dump.rdb sotopia-data/redis-data/
    ```
3. Start Redis, pointing to the snapshot:
    ```bash
    redis-stack-server --dir ~/Social_RL/sotopia-data/redis-data
    ```

## Running the Scripts

Once Redis is running and the snapshot is loaded, you can run the negotiation scenarios:

```bash
# 1. Basic buy/sell negotiation between two agents
python3 NegotiationArena_1_Buy_Sell.py

# 2. Each agent has access to a set of resources and a goal, they exchange resources to reach a deal
python3 NegotiationArena_2_Trading.py

# 3. Ultimatum game where one agent proposes a split and the other accepts or rejects
python3 NegotiationArena_3_Ultimatum.py
```

Happy negotiating!