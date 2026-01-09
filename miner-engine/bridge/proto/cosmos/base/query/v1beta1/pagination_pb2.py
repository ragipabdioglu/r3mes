"""Stub for pagination.proto - minimal implementation for Python client."""

# Minimal pagination message stubs
class PageRequest:
    def __init__(self):
        self.key = b""
        self.offset = 0
        self.limit = 0
        self.count_total = False
        self.reverse = False

class PageResponse:
    def __init__(self):
        self.next_key = b""
        self.total = 0

