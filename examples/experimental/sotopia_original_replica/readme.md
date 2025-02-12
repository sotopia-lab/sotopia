To run this example, please use aact to launch.

```bash
aact run-dataflow examples/experimental/sotopia_original_replica/origin.toml
```

To view the flow of the information, please run:

```bash
aact draw-dataflow examples/experimental/sotopia_original_replica/origin.toml --svg-path examples/experimental/sotopia_original_replica/origin.svg
```

To quickly generate your own simluation config, format your input like in the `raw_config.toml` file
to generate an executable file, run:
```bash
cd examples/experimental/sotopia_original_replica
python generate_executable.py --input=raw_config.toml  # output will be stored in output.toml
aact run-dataflow output.toml  # calling aact to run the simulation
```

![Alt text](./origin.svg)
