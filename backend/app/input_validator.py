"""
Input Validation System - Comprehensive input validation and sanitization

Provides secure input validation for all API endpoints with:
- Wallet address validation (remes1[a-z0-9]{38} pattern)
- XSS and injection attack prevention
- Numeric range validation
- String length enforcement
- SQL injection prevention
"""

import re
import html
import logging
from typing import Union, Optional, Any, Dict, List
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation error."""
    pass


class InputValidator:
    """
    Comprehensive input validation system.
    
    Validates and sanitizes all external inputs to prevent:
    - XSS attacks
    - SQL injection
    - Command injection
    - Path traversal
    - Buffer overflow attacks
    """
    
    # Wallet address pattern for R3MES (remes1 + 38 characters)
    WALLET_ADDRESS_PATTERN = re.compile(r'^remes1[a-z0-9]{38}$')
    
    # Maximum string length to prevent DoS attacks
    MAX_STRING_LENGTH = 10000
    
    # Dangerous characters that should be escaped or rejected
    DANGEROUS_CHARS = ['<', '>', '"', "'", '&', '\x00', '\r', '\n']
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
        re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)", re.IGNORECASE),
        re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
        re.compile(r"(\bUNION\s+SELECT\b)", re.IGNORECASE),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r"[;&|`$(){}[\]\\]"),
        re.compile(r"\b(rm|del|format|shutdown|reboot|kill|ps|ls|cat|grep|find|wget|curl)\b", re.IGNORECASE),
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\.[\\/]"),
        re.compile(r"[\\/]\.\."),
        re.compile(r"%2e%2e[\\/]", re.IGNORECASE),
        re.compile(r"[\\/]%2e%2e", re.IGNORECASE),
    ]
    
    @classmethod
    def validate_wallet_address(cls, address: str) -> str:
        """
        Validate R3MES wallet address format.
        
        Args:
            address: Wallet address to validate
            
        Returns:
            Validated wallet address
            
        Raises:
            ValidationError: If address format is invalid
        """
        if not isinstance(address, str):
            raise ValidationError("Wallet address must be a string")
        
        address = address.strip()
        
        if not address:
            raise ValidationError("Wallet address cannot be empty")
        
        if not cls.WALLET_ADDRESS_PATTERN.match(address):
            raise ValidationError(
                "Invalid wallet address format. Must match pattern: remes1[a-z0-9]{38}"
            )
        
        return address
    
    @classmethod
    def validate_string_input(
        cls, 
        value: str, 
        field_name: str,
        min_length: int = 1,
        max_length: Optional[int] = None,
        allow_html: bool = False,
        check_sql_injection: bool = True,
        check_command_injection: bool = True,
        check_path_traversal: bool = True
    ) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: String value to validate
            field_name: Name of the field (for error messages)
            min_length: Minimum allowed length
            max_length: Maximum allowed length (defaults to MAX_STRING_LENGTH)
            allow_html: Whether to allow HTML content
            check_sql_injection: Whether to check for SQL injection patterns
            check_command_injection: Whether to check for command injection patterns
            check_path_traversal: Whether to check for path traversal patterns
            
        Returns:
            Validated and sanitized string
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        # Remove null bytes and other control characters
        value = value.replace('\x00', '')
        value = ''.join(char for char in value if ord(char) >= 32 or char in ['\t', '\n', '\r'])
        
        # Strip whitespace
        value = value.strip()
        
        # Check length constraints
        if len(value) < min_length:
            raise ValidationError(f"{field_name} too short (minimum {min_length} characters)")
        
        max_len = max_length or cls.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValidationError(f"{field_name} too long (maximum {max_len} characters)")
        
        # Check for SQL injection patterns
        if check_sql_injection:
            for pattern in cls.SQL_INJECTION_PATTERNS:
                if pattern.search(value):
                    raise ValidationError(f"{field_name} contains potentially dangerous SQL patterns")
        
        # Check for command injection patterns
        if check_command_injection:
            for pattern in cls.COMMAND_INJECTION_PATTERNS:
                if pattern.search(value):
                    raise ValidationError(f"{field_name} contains potentially dangerous command patterns")
        
        # Check for path traversal patterns
        if check_path_traversal:
            for pattern in cls.PATH_TRAVERSAL_PATTERNS:
                if pattern.search(value):
                    raise ValidationError(f"{field_name} contains path traversal patterns")
        
        # Sanitize HTML if not allowed
        if not allow_html:
            value = html.escape(value)
        
        return value
    
    @classmethod
    def validate_numeric_input(
        cls,
        value: Union[int, float, str, Decimal],
        field_name: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_decimal: bool = True
    ) -> Union[int, float, Decimal]:
        """
        Validate numeric input with range constraints.
        
        Args:
            value: Numeric value to validate
            field_name: Name of the field (for error messages)
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_decimal: Whether to allow decimal values
            
        Returns:
            Validated numeric value
            
        Raises:
            ValidationError: If validation fails
        """
        # Convert string to number if needed
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValidationError(f"{field_name} cannot be empty")
            
            try:
                if allow_decimal:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                else:
                    value = int(value)
            except (ValueError, InvalidOperation):
                raise ValidationError(f"{field_name} must be a valid number")
        
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(f"{field_name} must be a number")
        
        # Check for NaN and infinity
        if isinstance(value, float):
            if not (value == value):  # NaN check
                raise ValidationError(f"{field_name} cannot be NaN")
            if value == float('inf') or value == float('-inf'):
                raise ValidationError(f"{field_name} cannot be infinite")
        
        # Check range constraints
        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name} below minimum value ({min_value})")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name} above maximum value ({max_value})")
        
        return value
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            Validated email address
            
        Raises:
            ValidationError: If email format is invalid
        """
        if not isinstance(email, str):
            raise ValidationError("Email must be a string")
        
        email = email.strip().lower()
        
        if not email:
            raise ValidationError("Email cannot be empty")
        
        # Basic email validation pattern
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(email):
            raise ValidationError("Invalid email format")
        
        if len(email) > 254:  # RFC 5321 limit
            raise ValidationError("Email address too long")
        
        return email
    
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """
        Validate URL format and scheme.
        
        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL format is invalid
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")
        
        url = url.strip()
        
        if not url:
            raise ValidationError("URL cannot be empty")
        
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                raise ValidationError("URL must include a scheme (http/https)")
            
            if parsed.scheme.lower() not in allowed_schemes:
                raise ValidationError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
            
            if not parsed.netloc:
                raise ValidationError("URL must include a hostname")
            
            # Check for localhost in production
            is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
            if is_production:
                hostname = parsed.hostname or ""
                if hostname.lower() in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127."):
                    raise ValidationError("Localhost URLs not allowed in production")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid URL format: {e}")
        
        return url
    
    @classmethod
    def validate_json_input(cls, json_str: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Validate and parse JSON input.
        
        Args:
            json_str: JSON string to validate
            max_size: Maximum allowed JSON size in bytes
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not isinstance(json_str, str):
            raise ValidationError("JSON input must be a string")
        
        json_str = json_str.strip()
        
        if not json_str:
            raise ValidationError("JSON input cannot be empty")
        
        if len(json_str.encode('utf-8')) > max_size:
            raise ValidationError(f"JSON input too large (maximum {max_size} bytes)")
        
        try:
            import json
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
            
        Raises:
            ValidationError: If filename is invalid
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")
        
        filename = filename.strip()
        
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        if not filename:
            raise ValidationError("Filename contains only invalid characters")
        
        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if filename.upper() in reserved_names:
            raise ValidationError(f"Filename '{filename}' is reserved")
        
        if len(filename) > 255:
            raise ValidationError("Filename too long (maximum 255 characters)")
        
        return filename
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Validated API key
            
        Raises:
            ValidationError: If API key format is invalid
        """
        if not isinstance(api_key, str):
            raise ValidationError("API key must be a string")
        
        api_key = api_key.strip()
        
        if not api_key:
            raise ValidationError("API key cannot be empty")
        
        # API key should be alphanumeric with possible hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            raise ValidationError("API key contains invalid characters")
        
        if len(api_key) < 16:
            raise ValidationError("API key too short (minimum 16 characters)")
        
        if len(api_key) > 128:
            raise ValidationError("API key too long (maximum 128 characters)")
        
        return api_key


# Convenience functions for common validations
def validate_wallet_address(address: str) -> str:
    """Convenience function to validate wallet address."""
    return InputValidator.validate_wallet_address(address)


def validate_string(value: str, field_name: str, **kwargs) -> str:
    """Convenience function to validate string input."""
    return InputValidator.validate_string_input(value, field_name, **kwargs)


def validate_number(value: Union[int, float, str], field_name: str, **kwargs) -> Union[int, float]:
    """Convenience function to validate numeric input."""
    return InputValidator.validate_numeric_input(value, field_name, **kwargs)


def validate_email(email: str) -> str:
    """Convenience function to validate email."""
    return InputValidator.validate_email(email)


def validate_url(url: str, **kwargs) -> str:
    """Convenience function to validate URL."""
    return InputValidator.validate_url(url, **kwargs)


def sanitize_filename(filename: str) -> str:
    """Convenience function to sanitize filename."""
    return InputValidator.sanitize_filename(filename)


def validate_api_key(api_key: str) -> str:
    """Convenience function to validate API key."""
    return InputValidator.validate_api_key(api_key)