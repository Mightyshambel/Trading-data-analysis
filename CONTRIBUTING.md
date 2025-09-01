# Contributing to Trading Data Analysis System

Thank you for your interest in contributing to the Trading Data Analysis System! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the guidelines below
4. **Test your changes**: `python -m pytest tests/`
5. **Submit a pull request**

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and open to feedback
- **Be professional** in all interactions
- **Focus on what is best for the community**

## 💡 How Can I Contribute?

### Reporting Bugs

- **Use the GitHub issue tracker**
- **Check existing issues** before creating new ones
- **Provide detailed information**:
  - Clear and descriptive title
  - Steps to reproduce the bug
  - Expected vs actual behavior
  - Environment details (OS, Python version, etc.)
  - Screenshots if applicable

### Suggesting Enhancements

- **Use the "Feature Request" issue template**
- **Describe the enhancement clearly**
- **Explain why this enhancement would be useful**
- **Provide examples of how it would work**

### Code Contributions

We welcome contributions in these areas:

- **New technical indicators**
- **Trading strategies**
- **Data visualization improvements**
- **Performance optimizations**
- **Documentation improvements**
- **Test coverage**
- **Bug fixes**

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mightyshambel/Trading-data-analysis.git
   cd Trading-data-analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Dependencies

Create a `requirements-dev.txt` file with:

```
pytest>=7.0.0
pytest-cov>=4.0.0
flake8>=6.0.0
black>=23.0.0
isort>=5.12.0
bandit>=1.7.0
safety>=2.3.0
pydocstyle>=6.3.0
pre-commit>=3.0.0
```

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `isort`
- **Code formatting**: Use `Black`

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **bandit**: Security scanning
- **pydocstyle**: Documentation style

### Running Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Security scan
bandit -r src/

# Documentation style
pydocstyle src/
```

### Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.0
    hooks:
      - id: bandit
        args: [-r, src/]
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Test Naming Convention

```python
def test_function_name_scenario_expected_result():
    """Test description."""
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_yfinance_client.py

# Run specific test
pytest tests/test_yfinance_client.py::TestYFinanceClient::test_client_initialization
```

### Test Coverage Requirements

- **Minimum coverage**: 80%
- **Critical modules**: 90%+
- **New features**: 100% coverage required

### Writing Tests

```python
import unittest
import pandas as pd
from src.yfinance_client import YFinanceClient

class TestYFinanceClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.client = YFinanceClient()
    
    def test_function_name(self):
        """Test description."""
        # Arrange
        expected = "expected_value"
        
        # Act
        result = self.client.some_function()
        
        # Assert
        self.assertEqual(result, expected)
```

## 📚 Documentation

### Code Documentation

- **All public functions** must have docstrings
- **Use Google style** docstrings
- **Include type hints** for all parameters and return values

```python
def fetch_market_data(
    client: YFinanceClient, 
    symbol: str, 
    period: str = '1y'
) -> pd.DataFrame:
    """
    Fetch market data for a given symbol.
    
    Args:
        client: Initialized YFinanceClient instance
        symbol: Financial instrument symbol
        period: Data period (e.g., '1y', '6mo')
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If symbol is invalid
        ConnectionError: If API connection fails
    """
    pass
```

### README Updates

- **Update README.md** for new features
- **Add code examples** for new functionality
- **Update installation instructions** if needed
- **Add screenshots** for UI changes

### API Documentation

- **Document all public APIs**
- **Provide usage examples**
- **Include parameter descriptions**
- **Document return values**

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `pytest`
2. **Check code quality**: `flake8 src/ tests/`
3. **Format code**: `black src/ tests/`
4. **Update documentation** if needed
5. **Add tests** for new functionality

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Screenshots (if applicable)
Add screenshots for UI changes

## Additional Notes
Any additional information
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Address feedback** and make changes
4. **Maintainer approval** required
5. **Merge to main branch**

## 🚀 Release Process

### Versioning

We use **Semantic Versioning** (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] **Update version** in `src/__init__.py`
- [ ] **Update CHANGELOG.md**
- [ ] **Run full test suite**
- [ ] **Update documentation**
- [ ] **Create release tag**
- [ ] **Deploy to PyPI** (if applicable)

### Creating a Release

```bash
# Update version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create GitHub release
# Go to GitHub > Releases > Create new release
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12.0]
- Python: [e.g., 3.9.7]
- Package Version: [e.g., 1.0.0]

## Additional Information
Screenshots, logs, etc.
```

## 💬 Communication

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

### Community Guidelines

- **Be respectful** and constructive
- **Help others** when possible
- **Share knowledge** and experiences
- **Follow the project's code of conduct**

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for contributing to the Trading Data Analysis System! Your contributions help make this project better for everyone.

---

**Happy Coding! 🚀**
