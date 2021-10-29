# Purdue CS 573 Fall 2021 Project

Data Mining

## Getting Started

### Environment

We use Poetry to manage dependencies and Python environment. Please make sure
you have [Poetry installed](https://python-poetry.org/docs/#installation).

### Install Dependencies

After cloning the project, enter the project directory. Use the following
command to install dependencies. The dependencies should be installed within the
project `/.venv`.

```sh
poetry install --no-root
```

Note that we set the Python version to 3.8.5. Poetry will check the Python
version before any installation. Use [pyenv](https://github.com/pyenv/pyenv) or
[asdf](https://github.com/asdf-vm/asdf) to manage your Python version.

### Activate Python Virtual Environment

After installation, activate the local Python virtual environment.

```sh
source ./.venv/bin/activate
```

Remember to activate the virtual environment every time you start a new terminal
before running the scripts or notebooks in this project.

## Development

### Add Dependencies

To add dependencies for the scripts,

```sh
poetry add <dependency>
```

To add development dependencies (e.g. jupyter and mypy),

```sh
poetry add -D <dev-dependency>
```