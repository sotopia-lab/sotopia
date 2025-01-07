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
    .pip_install("streamlit~=1.40.2", "uv")
    .run_commands(
        "rm -rf sotopia && git clone https://github.com/sotopia-lab/sotopia.git && cd sotopia && git checkout demo && uv pip install pyproject.toml --system && pip install -e . && cd ui/streamlit_ui",
        force_build=True,
    )
    # .pip_install("pydantic==2.8.2")
    .run_commands("pip list")
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
