# Contributing to Intraday EMA-RSI Backtester

Thank you for your interest in contributing to this project! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/backtester.git
   cd backtester
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where appropriate

## Running Tests

```bash
pytest tests/ -v
```

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request

## Commit Message Format

Use the following prefixes:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for changes to existing functionality
- `Docs:` for documentation changes
- `Refactor:` for code refactoring
- `Test:` for adding or updating tests

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Sample data (if applicable)

## Feature Requests

Feature requests are welcome! Please open an issue with:
- Clear description of the feature
- Use case / why it would be helpful
- Any relevant examples

## Questions?

Feel free to open an issue for any questions about contributing.
