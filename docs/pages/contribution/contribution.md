# Contributing

Thanks for your interest in contributing to Sotopia! We welcome and appreciate contributions.

## How Can I Contribute?

There are many ways that you can contribute:

1. **Download and use** Sotopia, and send [issues](https://github.com/sotopia-lab/sotopia/issues) when you encounter something that isn't working or a feature that you'd like to see.
2. **Send feedback** after each session by using our feedback mechanisms (if implemented), so we can see where things are working and failing, and also build an open dataset for training social agents.
3. **Improve the Codebase** by sending PRs (see details below). In particular, we have some [good first issue](https://github.com/sotopia-lab/sotopia/labels/good%20first%20issue) issues that may be ones to start on.

## For Sotopia Developers
If you are a developer and want to contribute to Sotopia, we really really appreciate it. Here are some guidelines for you. The same guidelines also apply to Sotopia team members.

### 0. Before You Start

Before you start contributing to Sotopia, please make sure you have access to the following:

- Python environment supporting Python 3.10+
- Uv: You can install uv using `pip install uv;` Sotoipa uses uv for managing the project.
- Docker: Sotopia uses Docker for testing and deployment. You will need Docker to run the tests.
- Redis: Sotopia uses Redis for caching and storing data. You will need to have a Redis server running locally or remotely.
- LLM server or API key: Most of Sotopia applications require access to a language model. You can use OpenAI, Anthropic, or other language models. If you don't have access to a language model, you can use a local model like Ollama, Llama.cpp, or vLLM.

Don't want to set up all these environments manually? Follow the steps in [here](#set-up-the-development-environment) to set up a Dev Container.

### 1. Code Quality

Sotopia code is typed, linted, and (mostly) tested. When you send a pull request, please make sure to run the following commands to check your code:

#### Mypy
Sotopia uses [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking. You can run mypy by:
```shell
uv run mypy --strict .
```

For most IDEs, you can also install the mypy extension to check the type while you are coding.

#### Linting
Sotopia uses several pre-commit hooks to ensure code quality. You can install pre-commit by:

```shell
uv run pre-commit install
```

Then, every time you commit, pre-commit will run the hooks to check your code. Sotopia has a github action
which autofixes some of the issues, but it is better to fix them before committing.

#### Unit and Integration Tests
Sotopia provides a docker environment for testing. You can run the following command to test your code:

```shell
# Starting from your local sotopia repository
# Requires Docker
# If you are using devcontainer, you can run pytest --ignore tests/cli
bash tests/tests.sh
```

The above command will run all tests in the `tests` folder except for `tests/cli`. If you are not changing the installation CLI, you don't need to worry about it.

If you have implemented a new feature, please also write tests for it. Sotopia github action will
check the code coverage after your PR passed the tests. The PR will not be merged if the code coverage is lower than the threshold.
Feel free to ask on Discord if you find it hard to write tests for your features.

#### Documentation
Lastly, please also update the documentation if you have changed the API or added new features. You can find the documentation in the `docs` folder.

To build the documentation, you can run:

```shell
# You will need to install [bun](https://bun.sh/) first
cd docs; bun run dev
```

### How to Open a Pull Request

* On GitHub, go to the page of your forked repository, and create a Pull Request:
   - Click on `Branches`
   - Click on the `...` beside your branch and click on `New pull request`
   - Set `base repository` to `sotopia-lab/sotopia`
   - Set `base` to `main`
   - Click `Create pull request`

The PR should appear in [Sotopia PRs](https://github.com/sotopia-lab/sotopia/pulls).

Then the Sotopia team will review your code.

#### Some PR Rules

As described [here](https://github.com/commitizen/conventional-commit-types/blob/master/index.json), a valid PR title should begin with one of the following prefixes:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white space, formatting, missing semicolons, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

For example, a PR title could be:
- `refactor: modify package path`
- `feat(frontend): xxxx`, where `(frontend)` means that this PR mainly focuses on the frontend component.

You may also check out previous PRs in the [PR list](https://github.com/sotopia-lab/sotopia/pulls).

For the description of the PR:
- If your PR is small (such as a typo fix), you can go brief.
- If it contains a lot of changes, it's better to write more details.
- If your PR is related to an issue, please mention the issue number in the description.

## Contribution Tips
The following are some tips for contributing to Sotopia, you are not required to follow them, but they may help you to contribute more effectively.

### Set up the Development Environment

We recommend using Dev Containers to set up your development environment, but note that Docker is required for install Dev Containers.

#### Using VSCode

If you use VSCode, you can install the Dev Containers extension, and then in [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette), run `Dev Containers: Open Folder in Container`.
After the container is built (the first time it may take 10+ minutes to build), you have a Redis server and local Llama server running.

#### Other IDEs or Editors

Please refer to [Dev Containers](https://containers.dev/supporting#editors) to see how to set up DevContainers in other editors.

#### Without Dev Containers

You can also set up the development environment without Dev Containers. There are three things you will need to set up manually:

- Python and uv: Please start from an environment supporting Python 3.10+ and install uv using `pip install uv; uv sync --all-extra`.
- Redis: Please refer to introduction page for the set up of Redis.
- Local LLM (optional): If you don't have access to model endpoints (e.g. OpenAI, Anthropic or others), you can use a local model. You can use Ollama, Llama.cpp,  vLLM or many others which support OpenAI compatible endpoints.


### Modern Editors
You might love vim/emacs/sublime/Notebook++. But modern editors like VSCode, PyCharm offers more features that can help you
write better code faster. Here are some extensions that you might want to install:

- Mypy Type Checker: For static type checking. Make sure to choose the right python interpreter (`/workspaces/.venv/bin/python` in the devcontainer or `.venv` locally) in the settings, and enable `Mypy: Run Using Active Interpreter`.
- Dev Containers: If you are using Dev Containers, you can install the Dev Containers as mentioned above.
- Ruff: For formatting your code.
- MDX: For the documentation website under `docs` folder.
- Even Better TOML: For editing aact workflow files.
