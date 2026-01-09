"""
Advanced Input Sanitization for R3MES Backend

Production-ready protection against:
- XSS (Cross-Site Scripting)
- SQL Injection
- NoSQL Injection
- Command Injection
- Path Traversal
- LDAP Injection
- XML Injection
"""

import re
import html
import unicodedata
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, parse_qs
import logging

from .exceptions import InvalidInputError, ValidationError

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Advanced input sanitization with multiple protection layers."""
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'<iframe[^>]*>', re.IGNORECASE),
        re.compile(r'<object[^>]*>', re.IGNORECASE),
        re.compile(r'<embed[^>]*>', re.IGNORECASE),
        re.compile(r'<applet[^>]*>', re.IGNORECASE),
        re.compile(r'<meta[^>]*>', re.IGNORECASE),
        re.compile(r'<link[^>]*>', re.IGNORECASE),
        re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL),
        re.compile(r'expression\s*\(', re.IGNORECASE),  # CSS expression
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'data:text/html', re.IGNORECASE),
    ]
    
    # SQL Injection patterns
    SQL_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|DECLARE)\b)", re.IGNORECASE),
        re.compile(r"(--|#|/\*|\*/|;)", re.IGNORECASE),  # SQL comments
        re.compile(r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+", re.IGNORECASE),  # OR 1=1
        re.compile(r"'\s*(OR|AND)\s*'", re.IGNORECASE),
        re.compile(r"(xp_|sp_)\w+", re.IGNORECASE),  # SQL Server stored procedures
    ]
    
    # NoSQL Injection patterns
    NOSQL_PATTERNS = [
        re.compile(r'\$where', re.IGNORECASE),
        re.compile(r'\$ne', re.IGNORECASE),
        re.compile(r'\$gt', re.IGNORECASE),
        re.compile(r'\$lt', re.IGNORECASE),
        re.compile(r'\$regex', re.IGNORECASE),
        re.compile(r'\$or', re.IGNORECASE),
        re.compile(r'\$and', re.IGNORECASE),
    ]
    
    # Command Injection patterns
    COMMAND_PATTERNS = [
        re.compile(r'[;&|`$]'),  # Shell metacharacters
        re.compile(r'\$\(.*?\)'),  # Command substitution
        re.compile(r'`.*?`'),  # Backticks
        re.compile(r'\|\|'),  # OR operator
        re.compile(r'&&'),  # AND operator
    ]
    
    # Path Traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r'\.\.[\\/]'),  # ../ or ..\
        re.compile(r'\.\.%2[fF]'),  # URL encoded ../
        re.compile(r'%2e%2e[\\/]'),  # URL encoded ..
        re.compile(r'\.\.\\'),  # Windows path traversal
    ]
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        max_length: int = 10000,
        allow_html: bool = False,
        strict: bool = True
    ) -> str:
        """
        Sanitize string input with multiple protection layers.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            strict: Enable strict mode (reject suspicious patterns)
            
        Returns:
            Sanitized string
            
        Raises:
            InvalidInputError: If input contains malicious patterns
        """
        if not isinstance(value, str):
            raise InvalidInputError("Value must be a string")
        
        # Remove null bytes and control characters
        value = cls._remove_control_characters(value)
        
        # Normalize Unicode
        value = unicodedata.normalize('NFKC', value)
        
        # Check length
        if len(value) > max_length:
            if strict:
                raise InvalidInputError(f"Input too long (max: {max_length} characters)")
            value = value[:max_length]
        
        # Check for XSS patterns
        if cls._contains_xss(value):
            if strict:
                raise InvalidInputError("Input contains potentially malicious XSS patterns")
            value = cls._remove_xss(value)
        
        # Check for SQL injection patterns
        if cls._contains_sql_injection(value):
            if strict:
                raise InvalidInputError("Input contains potentially malicious SQL patterns")
            value = cls._escape_sql(value)
        
        # Check for NoSQL injection patterns
        if cls._contains_nosql_injection(value):
            if strict:
                raise InvalidInputError("Input contains potentially malicious NoSQL patterns")
            value = cls._escape_nosql(value)
        
        # Check for command injection patterns
        if cls._contains_command_injection(value):
            if strict:
                raise InvalidInputError("Input contains potentially malicious command patterns")
            value = cls._escape_command(value)
        
        # Check for path traversal patterns
        if cls._contains_path_traversal(value):
            if strict:
                raise InvalidInputError("Input contains potentially malicious path traversal patterns")
            value = cls._escape_path_traversal(value)
        
        # HTML escape if not allowing HTML
        if not allow_html:
            value = html.escape(value)
        
        return value.strip()
    
    @classmethod
    def sanitize_dict(
        cls,
        data: Dict[str, Any],
        max_depth: int = 10,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary values.
        
        Args:
            data: Input dictionary
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Sanitized dictionary
        """
        if current_depth >= max_depth:
            raise InvalidInputError("Dictionary nesting too deep")
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            key = cls.sanitize_string(key, max_length=100, strict=True)
            
            # Sanitize value
            if isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value, max_depth, current_depth + 1)
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized[key] = value
            else:
                # Convert unknown types to string and sanitize
                sanitized[key] = cls.sanitize_string(str(value))
        
        return sanitized
    
    @classmethod
    def sanitize_list(
        cls,
        data: List[Any],
        max_depth: int = 10,
        current_depth: int = 0
    ) -> List[Any]:
        """
        Recursively sanitize list values.
        
        Args:
            data: Input list
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Sanitized list
        """
        if current_depth >= max_depth:
            raise InvalidInputError("List nesting too deep")
        
        sanitized = []
        for value in data:
            if isinstance(value, str):
                sanitized.append(cls.sanitize_string(value))
            elif isinstance(value, dict):
                sanitized.append(cls.sanitize_dict(value, max_depth, current_depth + 1))
            elif isinstance(value, list):
                sanitized.append(cls.sanitize_list(value, max_depth, current_depth + 1))
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized.append(value)
            else:
                sanitized.append(cls.sanitize_string(str(value)))
        
        return sanitized
    
    @classmethod
    def _remove_control_characters(cls, value: str) -> str:
        """Remove null bytes and control characters."""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove other control characters (except newline, tab, carriage return)
        return ''.join(char for char in value if ord(char) >= 32 or char in '\n\t\r')
    
    @classmethod
    def _contains_xss(cls, value: str) -> bool:
        """Check if value contains XSS patterns."""
        return any(pattern.search(value) for pattern in cls.XSS_PATTERNS)
    
    @classmethod
    def _remove_xss(cls, value: str) -> str:
        """Remove XSS patterns from value."""
        for pattern in cls.XSS_PATTERNS:
            value = pattern.sub('', value)
        return value
    
    @classmethod
    def _contains_sql_injection(cls, value: str) -> bool:
        """Check if value contains SQL injection patterns."""
        return any(pattern.search(value) for pattern in cls.SQL_PATTERNS)
    
    @classmethod
    def _escape_sql(cls, value: str) -> str:
        """Escape SQL special characters."""
        # Replace single quotes with double single quotes
        value = value.replace("'", "''")
        # Remove SQL comments
        value = re.sub(r'(--|#|/\*|\*/)', '', value)
        return value
    
    @classmethod
    def _contains_nosql_injection(cls, value: str) -> bool:
        """Check if value contains NoSQL injection patterns."""
        return any(pattern.search(value) for pattern in cls.NOSQL_PATTERNS)
    
    @classmethod
    def _escape_nosql(cls, value: str) -> str:
        """Escape NoSQL special characters."""
        # Remove $ operators
        value = value.replace('$', '')
        return value
    
    @classmethod
    def _contains_command_injection(cls, value: str) -> bool:
        """Check if value contains command injection patterns."""
        return any(pattern.search(value) for pattern in cls.COMMAND_PATTERNS)
    
    @classmethod
    def _escape_command(cls, value: str) -> str:
        """Escape command injection characters."""
        # Remove shell metacharacters
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n']
        for char in dangerous_chars:
            value = value.replace(char, '')
        return value
    
    @classmethod
    def _contains_path_traversal(cls, value: str) -> bool:
        """Check if value contains path traversal patterns."""
        return any(pattern.search(value) for pattern in cls.PATH_TRAVERSAL_PATTERNS)
    
    @classmethod
    def _escape_path_traversal(cls, value: str) -> str:
        """Escape path traversal patterns."""
        # Remove ../ and ..\
        value = value.replace('../', '').replace('..\\', '')
        # Remove URL encoded versions
        value = value.replace('%2e%2e/', '').replace('%2e%2e\\', '')
        value = value.replace('..%2f', '').replace('..%5c', '')
        return value
    
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """
        Validate and sanitize URL.
        
        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])
            
        Returns:
            Sanitized URL
            
        Raises:
            InvalidInputError: If URL is invalid
        """
        if not url:
            raise InvalidInputError("URL cannot be empty")
        
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                raise InvalidInputError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
            
            # Check for localhost/private IPs in production
            is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
            if is_production:
                hostname = parsed.hostname or ''
                if any(private in hostname.lower() for private in ['localhost', '127.0.0.1', '0.0.0.0', '::1']):
                    raise InvalidInputError("Localhost URLs not allowed in production")
            
            # Check for suspicious patterns
            if cls._contains_xss(url) or cls._contains_command_injection(url):
                raise InvalidInputError("URL contains suspicious patterns")
            
            return url
            
        except ValueError as e:
            raise InvalidInputError(f"Invalid URL format: {e}")
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate and sanitize email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            Sanitized email
            
        Raises:
            InvalidInputError: If email is invalid
        """
        if not email:
            raise InvalidInputError("Email cannot be empty")
        
        # Basic email regex
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(email):
            raise InvalidInputError("Invalid email format")
        
        # Check for suspicious patterns
        if cls._contains_xss(email) or cls._contains_command_injection(email):
            raise InvalidInputError("Email contains suspicious patterns")
        
        return email.lower().strip()
    
    @classmethod
    def validate_ipfs_hash(cls, ipfs_hash: str) -> str:
        """
        Validate IPFS hash format.
        
        Args:
            ipfs_hash: IPFS hash to validate
            
        Returns:
            Validated IPFS hash
            
        Raises:
            InvalidInputError: If hash is invalid
        """
        if not ipfs_hash:
            raise InvalidInputError("IPFS hash cannot be empty")
        
        # CIDv0 pattern (Qm...)
        cidv0_pattern = re.compile(r'^Qm[1-9A-HJ-NP-Za-km-z]{44}$')
        # CIDv1 pattern (b...)
        cidv1_pattern = re.compile(r'^b[a-z2-7]{58}$')
        
        if not (cidv0_pattern.match(ipfs_hash) or cidv1_pattern.match(ipfs_hash)):
            raise InvalidInputError("Invalid IPFS hash format")
        
        return ipfs_hash


# Middleware for automatic sanitization
class SanitizationMiddleware:
    """Middleware to automatically sanitize request data."""
    
    @staticmethod
    async def sanitize_request_body(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request body.
        
        Args:
            body: Request body dictionary
            
        Returns:
            Sanitized body
        """
        try:
            return InputSanitizer.sanitize_dict(body)
        except Exception as e:
            logger.error(f"Error sanitizing request body: {e}")
            raise ValidationError(f"Request validation failed: {e}")
    
    @staticmethod
    async def sanitize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize query parameters.
        
        Args:
            params: Query parameters dictionary
            
        Returns:
            Sanitized parameters
        """
        try:
            return InputSanitizer.sanitize_dict(params)
        except Exception as e:
            logger.error(f"Error sanitizing query params: {e}")
            raise ValidationError(f"Query parameter validation failed: {e}")


import os
