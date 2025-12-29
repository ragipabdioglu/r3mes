# Code Quality Guidelines

This document outlines code quality standards and practices for R3MES.

## Overview

Code quality is maintained through:
- **Dead code removal**: Unused imports, functions, classes
- **Duplication reduction**: DRY (Don't Repeat Yourself) principle
- **Type safety**: Type hints for better IDE support and error detection
- **Code documentation**: Docstrings and comments

## Dead Code Detection

### Unused Imports

Remove unused imports to reduce clutter:

```python
# Bad
from typing import List, Dict, Optional, Tuple  # Tuple is unused

# Good
from typing import List, Dict, Optional
```

### Unused Functions/Classes

Functions and classes that are no longer used should be removed or marked as deprecated:

```python
# Deprecated code should be marked
class OldRouter:
    """
    ⚠️ DEPRECATED: Use NewRouter instead.
    This class will be removed in v2.0.0
    """
    pass
```

## Code Duplication

### Common Patterns

Extract common logic into utility functions:

```python
# Bad - Duplicated validation
def validate_wallet_address(address: str) -> bool:
    if not address:
        return False
    if not address.startswith("remes"):
        return False
    if len(address) < 20 or len(address) > 60:
        return False
    return True

def check_wallet(wallet: str) -> bool:
    if not wallet:
        return False
    if not wallet.startswith("remes"):
        return False
    if len(wallet) < 20 or len(wallet) > 60:
        return False
    return True

# Good - Single validation function
def validate_wallet_address(address: str) -> bool:
    """Validate Cosmos wallet address format."""
    if not address:
        return False
    if not address.startswith("remes"):
        return False
    if len(address) < 20 or len(address) > 60:
        return False
    return True

def check_wallet(wallet: str) -> bool:
    """Check wallet address validity."""
    return validate_wallet_address(wallet)
```

### Shared Utilities

Create shared utility modules for common operations:

```python
# backend/app/utils/validation.py
def validate_cosmos_address(address: str, prefix: str = "remes") -> bool:
    """Validate Cosmos SDK address format."""
    # Shared validation logic
    pass
```

## Type Safety

### Function Type Hints

All functions should have type hints:

```python
# Bad
def process_request(request):
    return result

# Good
def process_request(request: Request) -> Response:
    """Process HTTP request."""
    return result
```

### Return Type Hints

Always specify return types:

```python
# Bad
def get_user(user_id):
    return user_data

# Good
def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    return user_data
```

### Generic Types

Use generic types for collections:

```python
# Bad
def get_users() -> list:
    return users

# Good
from typing import List, Dict
def get_users() -> List[Dict[str, Any]]:
    """Get all users."""
    return users
```

### Optional Types

Use `Optional` for nullable values:

```python
# Bad
def find_user(name: str) -> User:
    # May return None
    return user or None

# Good
from typing import Optional
def find_user(name: str) -> Optional[User]:
    """Find user by name."""
    return user
```

## Code Analysis Tools

### Static Analysis

Use tools for automated code quality checks:

1. **mypy**: Type checking
   ```bash
   pip install mypy
   mypy backend/app
   ```

2. **pylint**: Code quality
   ```bash
   pip install pylint
   pylint backend/app
   ```

3. **flake8**: Style checking
   ```bash
   pip install flake8
   flake8 backend/app
   ```

4. **vulture**: Dead code detection
   ```bash
   pip install vulture
   vulture backend/app
   ```

### Running Analysis

```bash
# Run all checks
make lint

# Run specific check
make type-check  # mypy
make style-check  # flake8
make dead-code-check  # vulture
```

## Best Practices

### 1. Import Organization

Organize imports in this order:
1. Standard library
2. Third-party packages
3. Local application imports

```python
# Standard library
import os
import logging
from typing import Optional

# Third-party
from fastapi import FastAPI
from pydantic import BaseModel

# Local
from .database import Database
from .cache import get_cache_manager
```

### 2. Function Length

Keep functions short and focused:
- **Target**: < 50 lines
- **Maximum**: < 100 lines
- **If longer**: Split into smaller functions

### 3. Class Responsibilities

Follow Single Responsibility Principle:
- Each class should have one reason to change
- Extract complex logic into separate classes

### 4. Error Handling

Use specific exceptions:

```python
# Bad
try:
    result = operation()
except Exception as e:
    logger.error(e)

# Good
try:
    result = operation()
except SpecificError as e:
    logger.error(f"Specific error: {e}")
    raise
except AnotherError as e:
    logger.error(f"Another error: {e}")
    raise
```

### 5. Documentation

Document all public APIs:

```python
def process_payment(
    user_id: int,
    amount: float,
    currency: str = "USD"
) -> PaymentResult:
    """
    Process a payment for a user.
    
    Args:
        user_id: User ID
        amount: Payment amount
        currency: Currency code (default: USD)
    
    Returns:
        PaymentResult with transaction details
    
    Raises:
        InsufficientFundsError: If user has insufficient funds
        PaymentProcessingError: If payment processing fails
    """
    pass
```

## Code Review Checklist

Before submitting code, check:

- [ ] No unused imports
- [ ] No dead code (unused functions/classes)
- [ ] No code duplication
- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] Error handling is specific
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] No TODO/FIXME comments (or they're documented)

## Continuous Improvement

### Regular Audits

Perform regular code quality audits:

1. **Monthly**: Run static analysis tools
2. **Quarterly**: Review and refactor duplicated code
3. **Annually**: Major cleanup and modernization

### Metrics

Track code quality metrics:

- **Type coverage**: % of functions with type hints
- **Documentation coverage**: % of public APIs documented
- **Duplication rate**: % of duplicated code
- **Dead code rate**: % of unused code

---

**Son Güncelleme**: 2025-12-24

