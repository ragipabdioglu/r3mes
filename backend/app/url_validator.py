"""
URL Validator Module - SSRF Protection

Provides comprehensive URL validation to prevent Server-Side Request Forgery (SSRF) attacks.
This module validates serving node endpoints before proxying requests.

Security Features:
- Whitelist-based domain validation
- Internal/private IP blocking
- Protocol enforcement (HTTPS only in production)
- DNS rebinding protection
"""

import ipaddress
import socket
import logging
import os
from urllib.parse import urlparse
from typing import Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class URLValidationError(Exception):
    """Raised when URL validation fails."""
    pass


class ValidationResult(Enum):
    """URL validation result status."""
    VALID = "valid"
    INVALID_SCHEME = "invalid_scheme"
    INVALID_HOST = "invalid_host"
    BLOCKED_IP = "blocked_ip"
    DNS_RESOLUTION_FAILED = "dns_resolution_failed"
    NOT_IN_WHITELIST = "not_in_whitelist"
    INVALID_PORT = "invalid_port"


@dataclass
class URLValidationResponse:
    """URL validation response."""
    is_valid: bool
    result: ValidationResult
    reason: str
    resolved_ip: Optional[str] = None


class SSRFProtector:
    """
    SSRF Protection for serving node endpoints.
    
    Validates URLs before making outbound requests to prevent:
    - Access to internal services (AWS metadata, internal APIs)
    - Access to private networks (10.x.x.x, 192.168.x.x, etc.)
    - DNS rebinding attacks
    - Protocol downgrade attacks
    """
    
    # Private/internal IP ranges that should be blocked
    BLOCKED_IP_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),        # Private Class A
        ipaddress.ip_network("172.16.0.0/12"),     # Private Class B
        ipaddress.ip_network("192.168.0.0/16"),    # Private Class C
        ipaddress.ip_network("169.254.0.0/16"),    # Link-local (AWS metadata!)
        ipaddress.ip_network("127.0.0.0/8"),       # Loopback
        ipaddress.ip_network("0.0.0.0/8"),         # Current network
        ipaddress.ip_network("100.64.0.0/10"),     # Carrier-grade NAT
        ipaddress.ip_network("192.0.0.0/24"),      # IETF Protocol Assignments
        ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1
        ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2
        ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3
        ipaddress.ip_network("224.0.0.0/4"),       # Multicast
        ipaddress.ip_network("240.0.0.0/4"),       # Reserved
        ipaddress.ip_network("255.255.255.255/32"), # Broadcast
        # IPv6 blocked ranges
        ipaddress.ip_network("::1/128"),           # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),          # IPv6 unique local
        ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
        ipaddress.ip_network("ff00::/8"),          # IPv6 multicast
    ]
    
    # Allowed ports for serving nodes
    ALLOWED_PORTS = {80, 443, 8000, 8080, 8443, 3000, 5000}
    
    def __init__(
        self,
        allowed_domains: Optional[Set[str]] = None,
        require_https: bool = True,
        allow_localhost_in_dev: bool = True,
    ):
        """
        Initialize SSRF protector.
        
        Args:
            allowed_domains: Set of allowed domain names (whitelist)
            require_https: Require HTTPS protocol (recommended for production)
            allow_localhost_in_dev: Allow localhost in development mode
        """
        self.require_https = require_https
        self.allow_localhost_in_dev = allow_localhost_in_dev
        self.is_production = os.getenv("R3MES_ENV", "development").lower() == "production"
        
        # Load allowed domains from environment or use defaults
        if allowed_domains:
            self.allowed_domains = allowed_domains
        else:
            # Load from environment variable (comma-separated)
            env_domains = os.getenv("SERVING_NODE_ALLOWED_DOMAINS", "")
            if env_domains:
                self.allowed_domains = set(d.strip().lower() for d in env_domains.split(",") if d.strip())
            else:
                # Default allowed domains (should be configured in production)
                self.allowed_domains = set()
        
        # DNS cache to prevent rebinding attacks
        self._dns_cache: dict[str, str] = {}
    
    def _is_ip_blocked(self, ip_str: str) -> bool:
        """
        Check if an IP address is in a blocked range.
        
        Args:
            ip_str: IP address string
            
        Returns:
            True if IP is blocked
        """
        try:
            ip = ipaddress.ip_address(ip_str)
            for blocked_range in self.BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    return True
            return False
        except ValueError:
            # Invalid IP address format
            return True
    
    def _resolve_hostname(self, hostname: str) -> Optional[str]:
        """
        Resolve hostname to IP address with caching.
        
        Args:
            hostname: Hostname to resolve
            
        Returns:
            Resolved IP address or None if resolution fails
        """
        # Check cache first
        if hostname in self._dns_cache:
            return self._dns_cache[hostname]
        
        try:
            # Resolve hostname
            ip = socket.gethostbyname(hostname)
            # Cache the result
            self._dns_cache[hostname] = ip
            return ip
        except socket.gaierror:
            logger.warning(f"DNS resolution failed for hostname: {hostname}")
            return None
    
    def _is_valid_port(self, port: Optional[int]) -> bool:
        """
        Check if port is allowed.
        
        Args:
            port: Port number
            
        Returns:
            True if port is allowed
        """
        if port is None:
            return True  # Default ports (80/443) are allowed
        return port in self.ALLOWED_PORTS
    
    def validate_url(self, url: str) -> URLValidationResponse:
        """
        Validate a URL for SSRF protection.
        
        Args:
            url: URL to validate
            
        Returns:
            URLValidationResponse with validation result
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_HOST,
                reason=f"Failed to parse URL: {e}"
            )
        
        # 1. Validate scheme (protocol)
        if self.is_production and self.require_https:
            if parsed.scheme != "https":
                return URLValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_SCHEME,
                    reason=f"HTTPS required in production, got: {parsed.scheme}"
                )
        elif parsed.scheme not in ("http", "https"):
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_SCHEME,
                reason=f"Invalid scheme: {parsed.scheme}. Only http/https allowed."
            )
        
        # 2. Validate hostname exists
        hostname = parsed.hostname
        if not hostname:
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_HOST,
                reason="No hostname in URL"
            )
        
        hostname_lower = hostname.lower()
        
        # 3. Check localhost in production
        if self.is_production:
            if hostname_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return URLValidationResponse(
                    is_valid=False,
                    result=ValidationResult.BLOCKED_IP,
                    reason="Localhost not allowed in production"
                )
        elif not self.allow_localhost_in_dev:
            if hostname_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return URLValidationResponse(
                    is_valid=False,
                    result=ValidationResult.BLOCKED_IP,
                    reason="Localhost not allowed"
                )
        
        # 4. Validate port
        if not self._is_valid_port(parsed.port):
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_PORT,
                reason=f"Port {parsed.port} not allowed. Allowed ports: {self.ALLOWED_PORTS}"
            )
        
        # 5. Check if hostname is an IP address directly
        try:
            ip = ipaddress.ip_address(hostname)
            if self._is_ip_blocked(str(ip)):
                return URLValidationResponse(
                    is_valid=False,
                    result=ValidationResult.BLOCKED_IP,
                    reason=f"IP address {ip} is in a blocked range"
                )
            # IP address is valid and not blocked
            return URLValidationResponse(
                is_valid=True,
                result=ValidationResult.VALID,
                reason="IP address validated",
                resolved_ip=str(ip)
            )
        except ValueError:
            # Not an IP address, it's a hostname - continue validation
            pass
        
        # 6. Check domain whitelist (if configured)
        if self.allowed_domains:
            # Check exact match or subdomain match
            is_allowed = False
            for allowed_domain in self.allowed_domains:
                if hostname_lower == allowed_domain or hostname_lower.endswith(f".{allowed_domain}"):
                    is_allowed = True
                    break
            
            if not is_allowed:
                return URLValidationResponse(
                    is_valid=False,
                    result=ValidationResult.NOT_IN_WHITELIST,
                    reason=f"Domain {hostname_lower} not in allowed list: {self.allowed_domains}"
                )
        
        # 7. Resolve hostname and check resolved IP
        resolved_ip = self._resolve_hostname(hostname)
        if not resolved_ip:
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.DNS_RESOLUTION_FAILED,
                reason=f"Failed to resolve hostname: {hostname}"
            )
        
        # 8. Check if resolved IP is blocked
        if self._is_ip_blocked(resolved_ip):
            return URLValidationResponse(
                is_valid=False,
                result=ValidationResult.BLOCKED_IP,
                reason=f"Resolved IP {resolved_ip} for {hostname} is in a blocked range (potential DNS rebinding attack)"
            )
        
        # All checks passed
        return URLValidationResponse(
            is_valid=True,
            result=ValidationResult.VALID,
            reason="URL validated successfully",
            resolved_ip=resolved_ip
        )
    
    def validate_serving_endpoint(self, url: str) -> Tuple[bool, str]:
        """
        Validate a serving node endpoint URL.
        
        This is a convenience method that returns a simple tuple.
        
        Args:
            url: Serving node endpoint URL
            
        Returns:
            (is_valid, error_message) tuple
        """
        result = self.validate_url(url)
        if result.is_valid:
            return True, ""
        return False, result.reason
    
    def clear_dns_cache(self):
        """Clear the DNS cache."""
        self._dns_cache.clear()


# Global SSRF protector instance
_ssrf_protector: Optional[SSRFProtector] = None


def get_ssrf_protector() -> SSRFProtector:
    """
    Get the global SSRF protector instance.
    
    Returns:
        SSRFProtector instance
    """
    global _ssrf_protector
    if _ssrf_protector is None:
        _ssrf_protector = SSRFProtector()
    return _ssrf_protector


def validate_serving_endpoint(url: str) -> Tuple[bool, str]:
    """
    Validate a serving node endpoint URL.
    
    Convenience function that uses the global SSRF protector.
    
    Args:
        url: Serving node endpoint URL
        
    Returns:
        (is_valid, error_message) tuple
    """
    protector = get_ssrf_protector()
    return protector.validate_serving_endpoint(url)
