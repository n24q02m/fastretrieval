# Contributing to Qwen3 Embed

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- [Python](https://www.python.org/) 3.11+ (per `pyproject.toml`; tested through 3.14)
- [uv](https://docs.astral.sh/uv/)
- [mise](https://mise.jdx.dev/) (recommended)

## Setup

```bash
git clone https://github.com/n24q02m/qwen3-embed.git
cd qwen3-embed
mise run setup    # or: uv sync --group dev
```

## Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes and test:
   ```bash
   uv run pytest -m "not integration" --tb=short    # Unit tests
   uv run ruff check .                               # Lint
   uv run ruff format --check .                      # Format check
   ```

3. Commit using [Conventional Commits](https://www.conventionalcommits.org/). Only `feat:` and `fix:` prefixes are accepted (enforced by the commit-msg hook):
   ```
   feat: add new embedding model support
   fix: correct tokenizer padding behavior
   ```

4. Push and open a Pull Request against `main`

## Project Structure

```
qwen3_embed/
  __init__.py               # Public API: TextEmbedding, TextCrossEncoder
  py.typed                  # PEP 561 marker
  common/                   # Shared utilities (types, model management, ONNX)
  text/                     # Text embedding models
  rerank/cross_encoder/     # Reranking models
tests/
  test_utils.py             # Unit tests
  test_integration.py       # Integration tests (requires model download)
```

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) (4-space indent, double quotes, 99 line width)
- **Linting**: Ruff rules (E, F, I, UP, B, SIM)
- **Type checker**: [ty](https://docs.astral.sh/ty/)
- **Target**: Python 3.11+ (matches `requires-python` in `pyproject.toml`)

## Testing

- Write tests for all new functionality
- Place tests in `tests/` directory
- Mark integration tests with `@pytest.mark.integration`
- Integration tests require ~1.2 GB model download

```bash
uv run pytest                                    # All tests
uv run pytest -m "not integration" --tb=short    # Unit tests only
```

## Pull Request Guidelines

- Fill out the PR template completely
- Ensure all CI checks pass
- Keep PRs focused on a single concern
- Update documentation if behavior changes
- Add tests for new functionality

## Release Process

Releases are automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/)
and triggered through the CD workflow. Version bumps are determined by commit messages.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
