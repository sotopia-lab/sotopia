## Demo Realtime API

You would need `portaudio` to run this demo:

```bash
# On Mac
brew install portaudio

# On Linux
apt-get install portaudio19-dev
```

Execute this command in the repo folder to run the example:

```python
uv run --extra realtime aact run-dataflow examples/experimental/realtime/realtime_chat.toml
```
