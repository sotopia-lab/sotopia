# Simple example of custom agents
In this example, we consider two super simple agents:

- Simple Tick Agent: send out a text message at each second.
- Simple Echo Agent: reply the text message with "Hello, (received message)"

## Implementation of the agents
You can find the implmentation of the above agents in `tick_and_echo_agents.py`.
The documentation for creating these agents is [here](https://docs.sotopia.world/experimental/agents). This will not be focus of this example.

## Assemble the agents in the dataflow file
Dataflow file is used in `aact` to configurate dataflow with writing Python code.
You can think of the implementation of each node as the LEGO pieces, and the dataflow file as the building instruction manual.

To assemble the two agents above, you can add the following lines to the dataflow file:

```toml
[[nodes]]
node_name = "tick_agent" # a unique and arbitrary name in the dataflow
node_class = "simple_tick_agent" # the class alias

[nodes.node_args] # arguments for the agent's __init__
input_channel = "tick/secs/1"
output_channel = "tick"
```

Similar to the above, you can also create the `echo_agent`.

## Auxiliary nodes

Apart from the agents, there are two auxiliary nodes that you can use immediately (without writing Python code for them).

```toml
# tick node provides a stable clock for the tick agent
[[nodes]]
node_name = "tick"
node_class = "tick"

# print node lets you view the realtime messages in your channel
[[nodes]]
node_name = "print"
node_class = "print"

[nodes.node_args.print_channel_types]
# These are all the channels in this example, you can also just print out some of them
"tick/secs/1" = "tick"
"tick" = "text"
"echo_tick" = "text"
```

## Run this example
### Prerequisite
Redis hosted at `localhost:6379`. You would need to update the url in the tick_and_echo_agents.toml if you want to use a different redis database.

### Command
To run this example, please use aact to launch.

```bash
uv run aact run-dataflow examples/experimental/tick_and_echo_agents/tick_and_echo_agents.toml
```

## What you can expect to see
You will see lines of json strings printed out, each of which is a message between the nodes. The following six lines
```json
{"timestamp":"2024-10-14T15:23:53.876886","channel":"tick/secs/1","data":{"data_type":"tick","tick":0}}
{"timestamp":"2024-10-14T15:23:53.878931","channel":"tick","data":{"data_type":"text","text":"Tick 0"}}
{"timestamp":"2024-10-14T15:23:53.882522","channel":"echo_tick","data":{"data_type":"text","text":"Hello, Tick 0!"}}
{"timestamp":"2024-10-14T15:23:54.878441","channel":"tick/secs/1","data":{"data_type":"tick","tick":1}}
{"timestamp":"2024-10-14T15:23:54.880346","channel":"tick","data":{"data_type":"text","text":"Tick 1"}}
{"timestamp":"2024-10-14T15:23:54.881737","channel":"echo_tick","data":{"data_type":"text","text":"Hello, Tick 1!"}}
```
shows the following events:

1. tick node ticks
2. tick agent received the tick and output a message "Tick 0"
3. echo agent received the message and echo "Hello, Tick 0!"
4. 1-3 are repeated.

Each of the message has its timestamp for easy debugging and data recording. You can see that on a normal laptop the latency between nodes reacting to the message is around 1-2 ms, which is often fast enough for real time applications.
