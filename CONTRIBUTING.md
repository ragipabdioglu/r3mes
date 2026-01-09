# Contributing to R3MES

Thank you for your interest in contributing to R3MES! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/r3mes-network/r3mes/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, versions, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. Check existing [Issues](https://github.com/r3mes-network/r3mes/issues) and [Discussions](https://github.com/r3mes-network/r3mes/discussions)
2. Create a new discussion in the "Ideas" category
3. Describe the feature and its use case
4. Be open to feedback and iteration

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following our coding standards
4. Write or update tests as needed
5. Ensure all tests pass: `make test`
6. Commit with clear messages: `git commit -m "feat: add new feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Prerequisites

- Go 1.24+
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA 12.1+

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/r3mes-network/r3mes.git
cd r3mes

# Install dependencies
make setup

# Run tests
make test

# Start development environment
make dev
```

### Component-Specific Setup

#### Blockchain (Go)
```bash
cd remes
make install
make test
```

#### Backend (Python)
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest
```

#### Frontend (Next.js)
```bash
cd web-dashboard
npm install
npm run lint
npm test
```

#### Miner Engine (Python)
```bash
cd miner-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest
```

## Coding Standards

### Go (Blockchain)
- Follow [Effective Go](https://golang.org/doc/effective_go)
- Use `gofmt` and `golangci-lint`
- Write tests for all new functionality
- Document exported functions and types

### Python (Backend, Miner)
- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Use `black` for formatting, `ruff` for linting
- Write docstrings for public functions
- Maintain >80% test coverage

### TypeScript (Frontend)
- Follow [TypeScript Best Practices](https://www.typescriptlang.org/docs/handbook/declaration-files/do-s-and-don-ts.html)
- Use ESLint and Prettier
- Write component tests with Jest/React Testing Library
- Use functional components with hooks

### Rust (Desktop Launcher)
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` and `clippy`
- Document public APIs
- Write unit tests

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(miner): add multi-GPU support
fix(backend): resolve memory leak in inference
docs(readme): update installation instructions
```

## Testing

### Running Tests

```bash
# All tests
make test

# Specific components
make test-blockchain
make test-backend
make test-frontend
make test-miner

# With coverage
make test-coverage
```

### Writing Tests

- Write unit tests for all new functions
- Write integration tests for API endpoints
- Write E2E tests for critical user flows
- Aim for >80% code coverage

## Documentation

- Update relevant documentation with your changes
- Add JSDoc/docstrings for new functions
- Update API documentation if endpoints change
- Add examples for new features

## Review Process

1. All PRs require at least one approval
2. CI must pass (tests, linting, build)
3. Documentation must be updated
4. Breaking changes require discussion

## Security

- Never commit secrets or credentials
- Report security vulnerabilities privately to security@r3mes.network
- Follow secure coding practices
- Use parameterized queries for database operations

## Questions?

- Join our [Discord](https://discord.gg/r3mes)
- Start a [Discussion](https://github.com/r3mes-network/r3mes/discussions)
- Email: dev@r3mes.network

Thank you for contributing to R3MES! 🚀
