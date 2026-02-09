# Contributing to Pix2Pix

We appreciate your interest in contributing! Please follow these guidelines:

## Development Setup

```bash
# Clone and setup
git clone https://github.com/VishnuNambiar0602/Image_to_image_translation.git
cd Image_to_image_translation
pip install -e ".[dev]"
pre-commit install
```

## Code Standards

- **Type Hints**: All functions must have complete type hints
- **Docstrings**: Use Google-style docstrings for all public functions
- **Testing**: Add tests for new features
- **Formatting**: Run `black` and `isort`
- **Linting**: Pass `flake8` and `mypy`

## Pre-commit Hooks

```bash
pre-commit run --all-files
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src/pix2pix
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following code standards
4. Add tests for new functionality
5. Run all checks: `pre-commit run --all-files && pytest tests/`
6. Commit with clear messages
7. Push and create Pull Request

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include reproducible examples
- Specify Python version, PyTorch version, and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
