from typer.testing import CliRunner

from sotopia.cli import app

runner = CliRunner()


def test_install() -> None:
    result = runner.invoke(
        app,
        ["install", "--no-use-docker", "--load-database", "--overwrite-existing-data"],
    )
    assert result.exit_code == 0
