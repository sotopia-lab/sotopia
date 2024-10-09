FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

RUN curl -fsSL https://ollama.com/install.sh | sh\
   && pip install uv\
   && uv venv /workspaces/.venv\
   && export UV_PROJECT_ENVIRONMENT=/workspaces/.venv\
   && echo export UV_PROJECT_ENVIRONMENT=/workspaces/.venv >> /root/.bashrc
