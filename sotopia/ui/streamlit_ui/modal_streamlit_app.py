import shlex
import subprocess
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "gpg",
        "lsb-release",
        "wget",
        "procps",  # for ps command
        "redis-tools",  # for redis-cli
    )
    .pip_install("sotopia", "streamlit~=1.40.2")
    .pip_install(
        "pydantic>=2.5.0,<3.0.0",
        "aiohttp>=3.9.3,<4.0.0",
        "rich>=13.8.1,<14.0.0",
        "typer>=0.12.5",
        "aiostream>=0.5.2",
        "fastapi[all]",
        "uvicorn",
        "redis>=5.0.0",
        "rq",
        "lxml>=4.9.3,<6.0.0",
        "openai>=1.11.0,<2.0.0",
        "langchain>=0.2.5,<0.4.0",
        "PettingZoo==1.24.3",
        "redis-om>=0.3.0,<0.4.0",
        "gin-config>=0.5.0,<0.6.0",
        "absl-py>=2.0.0,<3.0.0",
        "together>=0.2.4,<1.4.0",
        "beartype>=0.14.0,<0.20.0",
        "langchain-openai>=0.1.8,<0.2",
        "hiredis>=3.0.0",
        "aact",
        "gin",
    )  # TODO similarly we need to solve this
    .run_commands(
        "rm -rf sotopia &&git clone https://github.com/sotopia-lab/sotopia.git && cd sotopia && git checkout feature/sotopia-demo-ui && pip install -e . && cd sotopia/ui/streamlit_ui"
    )
)


streamlit_script_local_path = Path(__file__).parent
print("streamlit_script_local_path************************")
print(streamlit_script_local_path)
streamlit_script_remote_path = Path("/root")


if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_project_mount = modal.Mount.from_local_dir(
    local_path=f"{str(streamlit_script_local_path)}",
    remote_path=f"{str(streamlit_script_remote_path)}",
)

# streamlit_script_mount = modal.Mount.from_local_file(
#     local_path=f"{str(streamlit_script_local_path)}/app.py",
#     remote_path=f"{str(streamlit_script_remote_path)}/app.py",
# )

app = modal.App(name="example-modal-streamlit-dev", image=image)


@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_project_mount],
)
@modal.web_server(8000)
def run() -> None:
    target = shlex.quote(f"{str(streamlit_script_remote_path)}/app.py")
    print("target************************")
    print(target)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=true --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
