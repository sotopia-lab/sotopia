import platform
from typer.testing import CliRunner

from sotopia.cli import app
import subprocess

runner = CliRunner()


def test_install() -> None:
    result = runner.invoke(
        app,
        ["install", "--no-use-docker", "--load-database", "--overwrite-existing-data"],
    )
    assert result.exit_code == 0

    if platform.system() == "Darwin":
        subprocess.call("redis-cli shutdown")
    elif platform.system() == "Linux":
        subprocess.call("./redis-stack-server-7.2.0-v10/bin/redis-cli shutdown")

    result = runner.invoke(
        app,
        ["install", "--use-docker", "--load-database", "--overwrite-existing-data"],
    )
    assert result.exit_code == 0
