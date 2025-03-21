To run this example, please use aact to launch.

```bash
python examples/experimental/sotopia_original_replica/simulate.py
```

this example can be also run in a web interface by running:

```bash
fastapi run sotopia/api/fastapi_server.py --port 8080
```
Then in another terminal, run:
```bash
python examples/experimental/sotopia_original_replica/websocket_simulation_client.py
```
You would see the msgs coming from the websocket server.

![Alt text](./origin.svg)
