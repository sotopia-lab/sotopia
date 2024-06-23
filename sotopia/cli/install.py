import subprocess
from typing import Literal, Optional
from pydantic import BaseModel
import rich
import rich.errors

from rich.prompt import Prompt

from pathlib import Path

import typer
from .menu import Menu
import tempfile

from .app import app


def _get_system() -> Literal["Linux", "Darwin", "Windows"]:
    import platform

    system = platform.system()
    if system == "Linux":
        return "Linux"
    elif system == "Darwin":
        return "Darwin"
    elif system == "Windows":
        return "Windows"
    else:
        raise ValueError(
            f"You are using {system} which is not supported. Please use Linux, MacOS or Windows."
        )


class Dataset(BaseModel):
    id: str
    display_name: str
    url: str
    venue: str
    license: str
    citation: str


class Datasets(BaseModel):
    datasets: list[Dataset]


def _get_dataset_to_be_loaded(
    published_datasets: Datasets, console: rich.console.Console
) -> str:
    menu = Menu(
        *(
            f"{dataset.display_name} ({dataset.venue}, {dataset.license})"
            for dataset in published_datasets.datasets
        ),
        "None of the above, I want only an empty database.",
        "No, I have a custom URL.",
        start_index=0,
        align="left",
        rule_title="Select the dataset to be loaded",
    )

    dataset = menu.ask(return_index=True)
    assert isinstance(dataset, int)

    if dataset < len(published_datasets.datasets):
        console.log(
            f"""Loading the database with data from {published_datasets.datasets[dataset].url}.
This data is from the {published_datasets.datasets[dataset].display_name}.
Licensed under {published_datasets.datasets[dataset].license}.
Please cite the following paper(s) if you use this data:
{published_datasets.datasets[dataset].citation}"""
        )
        return published_datasets.datasets[dataset].url
    elif dataset == len(published_datasets.datasets):
        console.log("Starting redis with an empty database.")
        return ""
    else:
        custom_load_database_url = Prompt.ask(
            "Enter the URL to load the database with initial data from.",
        )
        if custom_load_database_url == "":
            console.log("Starting redis with an empty database.")
            return ""
        else:
            console.log(
                f"Loading the database with initial data from {custom_load_database_url}."
            )
            return custom_load_database_url


@app.command()
def install(
    use_docker: Optional[bool] = typer.Option(None, help="Install redis using docker."),
    load_database: Optional[bool] = typer.Option(
        None, help="Load the database with initial sotopia(-pi) data."
    ),
    custom_load_database_url: Optional[str] = typer.Option(
        None, help="Load the database with initial data from a custom URL."
    ),
) -> None:
    console = rich.console.Console()
    system = _get_system()

    if use_docker is None:
        use_docker = (
            Prompt.ask(
                "Do you want to use docker? (Recommended) You will need to have docker installed.",
                choices=["Yes", "No"],
                default="Yes",
                console=console,
            )
            == "Yes"
        )

    if use_docker:
        try:
            subprocess.check_output("command -v docker", shell=True)
        except subprocess.CalledProcessError:
            if system == "Darwin":
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/desktop/install/mac-install/,
                    or install it using homebrew with `brew install --cask docker`.
                    And then run this command again.
                    """
                )
            elif system == "Linux":
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/engine/install/ubuntu/.
                    And then run this command again.
                    """
                )
            else:
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/desktop/install/windows-install/.
                    And then run this command again.
                    """
                )
        else:
            if system == "Windows":
                console.log("""For Windows, unfortunately only docker is supported.
                    Check the official documentation:
                    https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/windows/.
                """)
                exit(1)
            elif system == "Darwin":
                # check if homebrew is installed
                try:
                    subprocess.check_output("command -v brew", shell=True)
                except subprocess.CalledProcessError:
                    console.log(
                        """Homebrew is required for install redis without docker on MacOS.
                        Please check https://brew.sh/.
                        And then run this command again.
                        """
                    )
                    exit(1)

    if load_database is None:
        load_database = (
            Prompt.ask(
                "Do you want to load the database with published data?",
                choices=["Yes", "No"],
                default="Yes",
                console=console,
            )
            == "Yes"
        )

    if load_database and custom_load_database_url is None:
        load_from_custom_url = (
            Prompt.ask(
                "Do you want to use a custom URL to load your data?",
                choices=["Yes", "No"],
                default="No",
                console=console,
            )
            == "Yes"
        )
        if load_from_custom_url:
            custom_load_database_url = Prompt.ask(
                "Enter the URL to load the database with initial data from.",
            )

    url = ""
    if load_database:
        fn = Path(__file__).parent / "published_datasets.json"
        published_datasets = Datasets.parse_file(fn)
        url = _get_dataset_to_be_loaded(published_datasets, console)

    tmpdir_context = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_context.__enter__()

    if url:
        try:
            subprocess.run(f"curl -L {url} -o {tmpdir}/dump.rdb", shell=True)
            console.log("Database downloaded successfully.")
        except subprocess.CalledProcessError:
            console.log("Database download failed. Please check the URL and try again.")
    else:
        console.log("Starting redis with an empty database.")

    if use_docker:
        current_directory = Path(__file__).parent
        directory = Prompt.ask(
            "Enter the directory where you want to store the data. Press enter to use the current directory.",
            default=current_directory,
        )
        try:
            subprocess.run(f"mkdir -p {directory}/redis-data", shell=True)
        except subprocess.CalledProcessError:
            console.log(
                "Failed to create directory. Please check the path and try again."
            )
            exit(1)
        if Path.exists(Path(directory) / "redis-data/dump.rdb"):
            cover_existing = Prompt.ask(
                "The directory already contains a dump.rdb file. Do you want to overwrite it?",
                choices=["Yes", "No"],
                default="No",
                console=console,
            )
            if cover_existing == "No":
                console.log(
                    "Exiting the installation. Please provide a different directory."
                )
                exit(0)
        else:
            subprocess.run(
                f"mv {tmpdir}/dump.rdb {directory}/redis-data/dump.rdb", shell=True
            )
        try:
            subprocess.run(
                f"docker run --name redis-stack -p 6379:6379 -p 8001:8001 -v {directory}/redis-data:/data/ redis/redis-stack:latest",
                shell=True,
            )
            console.log("Redis started successfully.")
        except subprocess.CalledProcessError:
            console.log("Redis start failed. Please check the logs and try again.")
    else:
        if system == "Darwin":
            try:
                subprocess.run("brew tap redis-stack/redis-stack", shell=True)
                subprocess.run("brew install redis-stack", shell=True)
                if load_database:
                    if Path("/opt/homebrew/var/db/redis-stack/dump.rdb").exists():
                        cover_existing = Prompt.ask(
                            "The directory already contains a dump.rdb file. Do you want to overwrite it?",
                            choices=["Yes", "No"],
                            default="No",
                            console=console,
                        )
                        if cover_existing == "No":
                            console.log(
                                "Exiting the installation. Please provide a different directory."
                            )
                            exit(0)
                    subprocess.run(
                        f"mv {tmpdir}/dump.rdb /opt/homebrew/var/db/redis-stack/dump.rdb",
                        shell=True,
                    )
                subprocess.run("redis-stack-server --daemonize yes", shell=True)
                console.log("Redis started successfully.")
            except subprocess.CalledProcessError:
                console.log("Redis start failed. Please check the logs and try again.")
        elif system == "Linux":
            try:
                subprocess.run(
                    "curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.focal.x86_64.tar.gz -o redis-stack-server.tar.gz",
                    shell=True,
                )
                subprocess.run("tar -xvzf redis-stack-server.tar.gz", shell=True)
                if load_database:
                    subprocess.run(
                        f"mv {tmpdir}/dump.rdb ./redis-stack-server-7.2.0-v10/var/db/redis-stack/dump.rdb",
                        shell=True,
                    )
                subprocess.run(
                    "./redis-stack-server-7.2.0-v10/bin/redis-stack-server --daemonize yes",
                    shell=True,
                )
            except subprocess.CalledProcessError:
                console.log("Redis start failed. Please check the logs and try again.")

    tmpdir_context.__exit__(None, None, None)


if __name__ == "__main__":
    app()
