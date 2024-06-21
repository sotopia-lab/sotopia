import subprocess
from typing import Literal, Optional
import rich
import rich.errors
import typer

from rich.prompt import Prompt

from pathlib import Path

app = typer.Typer()


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


@app.command()
def cli(
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
                "Do you want to load the database with initial sotopia(-pi) data?",
                choices=["Yes", "No"],
                default="Yes",
                console=console,
            )
            == "Yes"
        )

    if load_database and custom_load_database_url is None:
        load_from_custom_url = (
            Prompt.ask(
                "Do you want to use a custom URL?",
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

    if load_database:
        url = (
            custom_load_database_url
            or "https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true"
        )
        console.log(f"""Loading the database with initial data from {url}.
            The data is from the sotopia(-pi) dataset (CC-BY-SA 4.0). Please cite the dataset if you use it.
            @inproceedings{{wang2024sotopiapi,
                title={{SOTOPIA-$\\pi$: Interactive Learning of Socially Intelligent Language Agents}},
                author={{Ruiyi Wang and Haofei Yu and Wenxin Zhang and Zhengyang Qi and Maarten Sap and Graham Neubig and Yonatan Bisk and Hao Zhu}},"
                booktitle={{Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL) 2024}},
                year={{2024}},
            }}
            @inproceedings{{
                zhou2024sotopia,
                title={{{{SOTOPIA}}: Interactive Evaluation for Social Intelligence in Language Agents}},
                author={{Xuhui Zhou and Hao Zhu and Leena Mathur and Ruohong Zhang and Haofei Yu and Zhengyang Qi and Louis-Philippe Morency and Yonatan Bisk and Daniel Fried and Graham Neubig and Maarten Sap}},
                booktitle={{The Twelfth International Conference on Learning Representations}},
                year={{2024}},
                url={{https://openreview.net/forum?id=mM7VurbA4r}}
            }}
            """)

        Path("redis-data").mkdir(exist_ok=True)
        try:
            subprocess.run(f"curl -L {url} -o redis-data/dump.rdb", shell=True)
            console.log("Database downloaded successfully.")
        except subprocess.CalledProcessError:
            console.log("Database download failed. Please check the URL and try again.")
    else:
        Path("redis-data").mkdir(exist_ok=True)
        console.log("Starting redis with an empty database.")

    if use_docker:
        try:
            subprocess.run(
                "docker run --name redis-stack -p 6379:6379 -p 8001:8001 -v ($pwd)/redis-data/:/data/ redis/redis-stack:latest",
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
                    subprocess.run(
                        "mv redis-data/dump.rdb /opt/homebrew/var/db/redis-stack/dump.rdb",
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
                        "mv redis-data/dump.rdb ./redis-stack-server-7.2.0-v10/var/db/redis-stack/dump.rdb",
                        shell=True,
                    )
                subprocess.run(
                    "./redis-stack-server-7.2.0-v10/bin/redis-stack-server --daemonize yes",
                    shell=True,
                )
            except subprocess.CalledProcessError:
                console.log("Redis start failed. Please check the logs and try again.")


if __name__ == "__main__":
    app()
