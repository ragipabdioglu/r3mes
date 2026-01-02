"""
Response Optimizer - API Performance Enhancement

Implements response compression, streaming, pagination, and serialization
optimizations to improve API performance and reduce bandwidth usage.
"""

import asyncio
import gzip
import json
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import time
import zlib
from io import BytesIO

from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "br"


@dataclass
class ResponseMetrics:
    """Response performance metrics."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    serialization_time: float
    total_time: float
    compression_type: CompressionType


class ResponseCompressor:
    """Handles response compression with multiple algorithms."""
    
    def __init__(self, min_size: int = 1024, compression_level: int = 6):
        """
        Initialize response compressor.
        
        Args:
            min_size: Minimum response size to compress (bytes)
            compression_level: Compression level (1-9)
        """
        self.min_size = min_size
        self.compression_level = compression_level
        self.stats = {
            'total_responses': 0,
            'compressed_responses': 0,
            'bytes_saved': 0,
            'compression_time': 0.0
        }
    
    def should_compress(self, content: bytes, content_type: str) -> bool:
        """
        Determine if content should be compressed.
        
        Args:
            content: Response content
            content_type: Content type header
            
        Returns:
            True if should compress
        """
        # Check size threshold
        if len(content) < self.min_size:
            return False
        
        # Check content type
        compressible_types = [
            'application/json',
            'text/html',
            'text/plain',
            'text/css',
            'text/javascript',
            'application/javascript',
            'application/xml',
            'text/xml'
        ]
        
        return any(ct in content_type.lower() for ct in compressible_types)
    
    def get_best_compression(self, accept_encoding: str) -> CompressionType:
        """
        Get best compression type based on Accept-Encoding header.
        
        Args:
            accept_encoding: Accept-Encoding header value
            
        Returns:
            Best supported compression type
        """
        if not accept_encoding:
            return CompressionType.NONE
        
        accept_encoding = accept_encoding.lower()
        
        # Priority order: brotli > gzip > deflate
        if 'br' in accept_encoding:
            return CompressionType.BROTLI
        elif 'gzip' in accept_encoding:
            return CompressionType.GZIP
        elif 'deflate' in accept_encoding:
            return CompressionType.DEFLATE
        
        return CompressionType.NONE
    
    def compress_content(
        self, 
        content: bytes, 
        compression_type: CompressionType
    ) -> tuple[bytes, ResponseMetrics]:
        """
        Compress content using specified algorithm.
        
        Args:
            content: Content to compress
            compression_type: Compression algorithm
            
        Returns:
            Tuple of (compressed_content, metrics)
        """
        start_time = time.time()
        original_size = len(content)
        
        try:
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(content, compresslevel=self.compression_level)
            elif compression_type == CompressionType.DEFLATE:
                compressed = zlib.compress(content, level=self.compression_level)
            elif compression_type == CompressionType.BROTLI:
                try:
                    import brotli
                    compressed = brotli.compress(content, quality=self.compression_level)
                except ImportError:
                    logger.warning("Brotli not available, falling back to gzip")
                    compressed = gzip.compress(content, compresslevel=self.compression_level)
                    compression_type = CompressionType.GZIP
            else:
                compressed = content
                compression_type = CompressionType.NONE
            
            compression_time = time.time() - start_time
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Update stats
            self.stats['total_responses'] += 1
            if compression_type != CompressionType.NONE:
                self.stats['compressed_responses'] += 1
                self.stats['bytes_saved'] += original_size - compressed_size
                self.stats['compression_time'] += compression_time
            
            metrics = ResponseMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compression_time=compression_time,
                serialization_time=0.0,
                total_time=compression_time,
                compression_type=compression_type
            )
            
            return compressed, metrics
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return content, ResponseMetrics(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time=time.time() - start_time,
                serialization_time=0.0,
                total_time=time.time() - start_time,
                compression_type=CompressionType.NONE
            )


class OptimizedJSONEncoder(json.JSONEncoder):
    """Optimized JSON encoder with custom serialization."""
    
    def default(self, obj):
        """Handle custom object serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        return super().default(obj)


class ResponseSerializer:
    """Optimized response serialization."""
    
    def __init__(self):
        """Initialize response serializer."""
        self.encoder = OptimizedJSONEncoder(separators=(',', ':'), ensure_ascii=False)
        self.stats = {
            'serializations': 0,
            'serialization_time': 0.0,
            'bytes_serialized': 0
        }
    
    def serialize_json(self, data: Any, optimize: bool = True) -> bytes:
        """
        Serialize data to optimized JSON.
        
        Args:
            data: Data to serialize
            optimize: Whether to apply optimizations
            
        Returns:
            Serialized JSON bytes
        """
        start_time = time.time()
        
        try:
            if optimize:
                # Apply optimizations
                data = self._optimize_data(data)
            
            # Serialize with optimized encoder
            json_str = self.encoder.encode(data)
            json_bytes = json_str.encode('utf-8')
            
            # Update stats
            serialization_time = time.time() - start_time
            self.stats['serializations'] += 1
            self.stats['serialization_time'] += serialization_time
            self.stats['bytes_serialized'] += len(json_bytes)
            
            return json_bytes
            
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            # Fallback to standard serialization
            return json.dumps(data, default=str).encode('utf-8')
    
    def _optimize_data(self, data: Any) -> Any:
        """
        Apply data optimizations before serialization.
        
        Args:
            data: Data to optimize
            
        Returns:
            Optimized data
        """
        if isinstance(data, dict):
            # Remove None values to reduce size
            return {k: self._optimize_data(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._optimize_data(item) for item in data]
        elif isinstance(data, datetime):
            # Use ISO format without microseconds for smaller size
            return data.replace(microsecond=0).isoformat()
        
        return data


class PaginationOptimizer:
    """Optimizes pagination for large datasets."""
    
    def __init__(self, default_page_size: int = 50, max_page_size: int = 1000):
        """
        Initialize pagination optimizer.
        
        Args:
            default_page_size: Default page size
            max_page_size: Maximum allowed page size
        """
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
    
    def validate_pagination(self, page: int, limit: int) -> tuple[int, int]:
        """
        Validate and optimize pagination parameters.
        
        Args:
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (validated_page, validated_limit)
        """
        # Validate page
        if page < 1:
            page = 1
        
        # Validate limit
        if limit < 1:
            limit = self.default_page_size
        elif limit > self.max_page_size:
            limit = self.max_page_size
        
        return page, limit
    
    def calculate_offset(self, page: int, limit: int) -> int:
        """Calculate database offset from page and limit."""
        return (page - 1) * limit
    
    def create_pagination_response(
        self,
        items: List[Any],
        page: int,
        limit: int,
        total_count: int,
        base_url: str = ""
    ) -> Dict[str, Any]:
        """
        Create paginated response with metadata.
        
        Args:
            items: Items for current page
            page: Current page number
            limit: Items per page
            total_count: Total number of items
            base_url: Base URL for pagination links
            
        Returns:
            Paginated response dictionary
        """
        total_pages = (total_count + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        response = {
            "items": items,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }
        
        # Add navigation links if base_url provided
        if base_url:
            links = {}
            if has_next:
                links["next"] = f"{base_url}?page={page + 1}&limit={limit}"
            if has_prev:
                links["prev"] = f"{base_url}?page={page - 1}&limit={limit}"
            if page != 1:
                links["first"] = f"{base_url}?page=1&limit={limit}"
            if page != total_pages:
                links["last"] = f"{base_url}?page={total_pages}&limit={limit}"
            
            response["pagination"]["links"] = links
        
        return response


class StreamingResponseGenerator:
    """Generates streaming responses for large datasets."""
    
    def __init__(self, chunk_size: int = 1024):
        """
        Initialize streaming response generator.
        
        Args:
            chunk_size: Size of each chunk in bytes
        """
        self.chunk_size = chunk_size
    
    async def stream_json_array(
        self,
        items_generator: AsyncGenerator[Any, None],
        serializer: ResponseSerializer
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream JSON array response.
        
        Args:
            items_generator: Async generator of items
            serializer: Response serializer
            
        Yields:
            JSON chunks as bytes
        """
        yield b'{"items":['
        
        first_item = True
        async for item in items_generator:
            if not first_item:
                yield b','
            else:
                first_item = False
            
            # Serialize item
            item_json = serializer.serialize_json(item)
            
            # Yield in chunks if large
            if len(item_json) > self.chunk_size:
                for i in range(0, len(item_json), self.chunk_size):
                    yield item_json[i:i + self.chunk_size]
            else:
                yield item_json
        
        yield b']}'
    
    async def stream_csv(
        self,
        items_generator: AsyncGenerator[Dict[str, Any], None],
        headers: List[str]
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream CSV response.
        
        Args:
            items_generator: Async generator of items (dicts)
            headers: CSV headers
            
        Yields:
            CSV chunks as bytes
        """
        # Yield headers
        header_line = ','.join(headers) + '\n'
        yield header_line.encode('utf-8')
        
        # Yield data rows
        async for item in items_generator:
            row_values = [str(item.get(header, '')) for header in headers]
            row_line = ','.join(row_values) + '\n'
            yield row_line.encode('utf-8')


class ResponseOptimizationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic response optimization.
    
    Applies compression, caching headers, and performance monitoring.
    """
    
    def __init__(self, app, enable_compression: bool = True, enable_caching: bool = True):
        """
        Initialize response optimization middleware.
        
        Args:
            app: FastAPI application
            enable_compression: Enable response compression
            enable_caching: Enable caching headers
        """
        super().__init__(app)
        self.enable_compression = enable_compression
        self.enable_caching = enable_caching
        self.compressor = ResponseCompressor()
        self.serializer = ResponseSerializer()
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.compression_savings = 0
    
    async def dispatch(self, request: Request, call_next):
        """Process request and optimize response."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Skip optimization for certain responses
        if self._should_skip_optimization(response):
            return response
        
        # Apply optimizations
        if self.enable_compression:
            response = await self._apply_compression(request, response)
        
        if self.enable_caching:
            response = self._apply_caching_headers(response)
        
        # Update performance metrics
        response_time = time.time() - start_time
        self.request_count += 1
        self.total_response_time += response_time
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-ID"] = getattr(request.state, 'request_id', 'unknown')
        
        return response
    
    def _should_skip_optimization(self, response: Response) -> bool:
        """Check if response should skip optimization."""
        # Skip for streaming responses
        if isinstance(response, StreamingResponse):
            return True
        
        # Skip for error responses
        if response.status_code >= 400:
            return True
        
        # Skip for small responses
        if hasattr(response, 'body') and len(response.body) < 1024:
            return True
        
        return False
    
    async def _apply_compression(self, request: Request, response: Response) -> Response:
        """Apply response compression."""
        if not hasattr(response, 'body'):
            return response
        
        content_type = response.headers.get('content-type', '')
        
        # Check if should compress
        if not self.compressor.should_compress(response.body, content_type):
            return response
        
        # Get best compression type
        accept_encoding = request.headers.get('accept-encoding', '')
        compression_type = self.compressor.get_best_compression(accept_encoding)
        
        if compression_type == CompressionType.NONE:
            return response
        
        # Compress content
        compressed_body, metrics = self.compressor.compress_content(
            response.body, compression_type
        )
        
        # Update response
        response.body = compressed_body
        response.headers['content-encoding'] = compression_type.value
        response.headers['content-length'] = str(len(compressed_body))
        response.headers['vary'] = 'Accept-Encoding'
        
        # Add compression metrics to headers (for debugging)
        response.headers['X-Compression-Ratio'] = f"{metrics.compression_ratio:.2f}"
        response.headers['X-Original-Size'] = str(metrics.original_size)
        response.headers['X-Compressed-Size'] = str(metrics.compressed_size)
        
        # Update stats
        self.compression_savings += metrics.original_size - metrics.compressed_size
        
        return response
    
    def _apply_caching_headers(self, response: Response) -> Response:
        """Apply appropriate caching headers."""
        # Default cache control
        cache_control = "public, max-age=300"  # 5 minutes
        
        # Customize based on content type
        content_type = response.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            # API responses - shorter cache
            cache_control = "public, max-age=60"  # 1 minute
        elif any(ct in content_type for ct in ['text/css', 'text/javascript', 'application/javascript']):
            # Static assets - longer cache
            cache_control = "public, max-age=86400"  # 1 day
        
        response.headers['cache-control'] = cache_control
        response.headers['etag'] = f'"{hash(response.body) % 2**32}"' if hasattr(response, 'body') else None
        
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get middleware performance statistics."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "requests_processed": self.request_count,
            "avg_response_time": avg_response_time,
            "compression_savings_bytes": self.compression_savings,
            "compressor_stats": self.compressor.stats,
            "serializer_stats": self.serializer.stats
        }


class OptimizedResponseFactory:
    """Factory for creating optimized responses."""
    
    def __init__(self):
        """Initialize response factory."""
        self.serializer = ResponseSerializer()
        self.paginator = PaginationOptimizer()
        self.streamer = StreamingResponseGenerator()
    
    def json_response(
        self,
        data: Any,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        optimize: bool = True
    ) -> JSONResponse:
        """
        Create optimized JSON response.
        
        Args:
            data: Data to serialize
            status_code: HTTP status code
            headers: Additional headers
            optimize: Whether to apply optimizations
            
        Returns:
            Optimized JSON response
        """
        # Serialize data
        json_bytes = self.serializer.serialize_json(data, optimize=optimize)
        
        # Create response
        response = Response(
            content=json_bytes,
            status_code=status_code,
            headers=headers,
            media_type="application/json"
        )
        
        return response
    
    def paginated_response(
        self,
        items: List[Any],
        page: int,
        limit: int,
        total_count: int,
        base_url: str = "",
        status_code: int = 200
    ) -> JSONResponse:
        """
        Create paginated JSON response.
        
        Args:
            items: Items for current page
            page: Page number
            limit: Items per page
            total_count: Total item count
            base_url: Base URL for pagination links
            status_code: HTTP status code
            
        Returns:
            Paginated JSON response
        """
        # Validate pagination
        page, limit = self.paginator.validate_pagination(page, limit)
        
        # Create paginated response
        response_data = self.paginator.create_pagination_response(
            items, page, limit, total_count, base_url
        )
        
        return self.json_response(response_data, status_code)
    
    def streaming_json_response(
        self,
        items_generator: AsyncGenerator[Any, None],
        media_type: str = "application/json"
    ) -> StreamingResponse:
        """
        Create streaming JSON response.
        
        Args:
            items_generator: Async generator of items
            media_type: Response media type
            
        Returns:
            Streaming JSON response
        """
        return StreamingResponse(
            self.streamer.stream_json_array(items_generator, self.serializer),
            media_type=media_type
        )
    
    def streaming_csv_response(
        self,
        items_generator: AsyncGenerator[Dict[str, Any], None],
        headers: List[str],
        filename: str = "export.csv"
    ) -> StreamingResponse:
        """
        Create streaming CSV response.
        
        Args:
            items_generator: Async generator of items
            headers: CSV headers
            filename: Download filename
            
        Returns:
            Streaming CSV response
        """
        response_headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        return StreamingResponse(
            self.streamer.stream_csv(items_generator, headers),
            media_type="text/csv",
            headers=response_headers
        )