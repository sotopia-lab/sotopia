from typing import Annotated
import typer

app = typer.Typer()


@app.command()
def cli(
    model: Annotated[
        str, typer.Option(help="The language model you want to benchmark.")
    ],
    partner_model: Annotated[
        str, typer.Option(help="The partner model you want to use.")
    ],
    task: Annotated[str, typer.Option(help="The task id you want to benchmark.")],
) -> None:
    """A simple command-line interface example."""
    typer.echo(f"Running benchmark for {model} and {partner_model} on task {task}.")
