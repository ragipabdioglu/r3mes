"""R3MES v1 proto stubs.

This module provides access to R3MES blockchain protocol buffer definitions.
"""

# Import generated modules with error handling
_AVAILABLE_MODULES = []

def _try_import(module_name):
    """Try to import a module, return None if it fails."""
    try:
        mod = __import__(module_name, globals(), locals(), ['*'], 1)
        return mod
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import {module_name}: {e}", ImportWarning)
        return None

# Transaction messages
tx_pb2 = _try_import('tx_pb2')
tx_pb2_grpc = _try_import('tx_pb2_grpc')
if tx_pb2: _AVAILABLE_MODULES.extend(['tx_pb2', 'tx_pb2_grpc'])

# Query messages
query_pb2 = _try_import('query_pb2')
query_pb2_grpc = _try_import('query_pb2_grpc')
if query_pb2: _AVAILABLE_MODULES.extend(['query_pb2', 'query_pb2_grpc'])

# Core types
stored_gradient_pb2 = _try_import('stored_gradient_pb2')
if stored_gradient_pb2: _AVAILABLE_MODULES.append('stored_gradient_pb2')

task_pool_pb2 = _try_import('task_pool_pb2')
if task_pool_pb2: _AVAILABLE_MODULES.append('task_pool_pb2')

node_pb2 = _try_import('node_pb2')
if node_pb2: _AVAILABLE_MODULES.append('node_pb2')

params_pb2 = _try_import('params_pb2')
if params_pb2: _AVAILABLE_MODULES.append('params_pb2')

model_pb2 = _try_import('model_pb2')
if model_pb2: _AVAILABLE_MODULES.append('model_pb2')

dataset_pb2 = _try_import('dataset_pb2')
if dataset_pb2: _AVAILABLE_MODULES.append('dataset_pb2')

serving_pb2 = _try_import('serving_pb2')
if serving_pb2: _AVAILABLE_MODULES.append('serving_pb2')

state_pb2 = _try_import('state_pb2')
if state_pb2: _AVAILABLE_MODULES.append('state_pb2')

__all__ = _AVAILABLE_MODULES

def get_available_modules():
    """Return list of successfully imported modules."""
    return _AVAILABLE_MODULES.copy()
