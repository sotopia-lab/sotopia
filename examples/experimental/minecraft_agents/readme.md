# Sotopia-Minecraft User Guide

```
cd examples/experimental/group_discussion_agents
uvicorn group_discussion_agents:app --reload --port 8080
```

Enter `Minecraft Java Edition`, select `Singleplayer`, `1.20.1 version`, and `Survival Mode`, then click `Open to LAN 55916`.

```
// Open a new terminal
cd examples/experimental/group_discussion_agents
export OPENAI_API_KEY=sk-  // Enter your OpenAI API key here
uv run aact run-dataflow group_discussion_agents.toml
```

Download [XianzheFan/mindcraft-sotopia-multiagent](https://github.com/XianzheFan/mindcraft-sotopia-multiagent), then go to the main folder.

```
node src/agent/index.js
```
