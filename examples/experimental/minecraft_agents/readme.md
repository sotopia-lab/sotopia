# User Guide

```
cd examples/experimental/minecraft_agents
uvicorn group_discussion_agents:app --reload --port 8080
```

Enter `Minecraft Java Edition`, select `Singleplayer`, `1.20.1 version`, and `Survival Mode`, then click `Open to LAN 55916`.

```
// Open a new terminal
cd examples/experimental/minecraft_agents
export OPENAI_API_KEY=sk-  // Enter your OpenAI API key here
uv run aact run-dataflow group_discussion_agents.toml
```

Download https://anonymous.4open.science/r/SoMi-ToM-1-580, then go to the main folder.

```
node src/agent/index.js
```
