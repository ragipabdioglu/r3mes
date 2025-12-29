#!/usr/bin/env python3
"""
Helper script to deserialize gradient data from IPFS for Go keeper.
Supports both pickle+gzip and protobuf formats.

Usage:
    python3 deserialize_gradient.py <format> <input_file >output_file
    format: "pickle" or "protobuf"
    
Output: JSON array of floats (flattened gradient tensor)
"""
import sys
import json
import gzip
import pickle
import os
from typing import List

# Add parent directory to path to import gradient_pb2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import gradient_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


def deserialize_pickle_gradient(data: bytes) -> List[float]:
    """Deserialize pickle+gzip format gradient."""
    try:
        # Decompress gzip
        decompressed = gzip.decompress(data)
        # Deserialize pickle
        package = pickle.loads(decompressed)
        
        # Extract gradients from package
        # Format matches LoRASerializer.serialize_gradients output
        if isinstance(package, dict) and 'gradients' in package:
            gradients_dict = package['gradients']
            # Flatten all gradients into single array
            flattened = []
            for key, value in gradients_dict.items():
                if isinstance(value, dict) and 'data' in value:
                    grad_data = value['data']
                    # Handle numpy arrays
                    import numpy as np
                    if isinstance(grad_data, np.ndarray):
                        flattened.extend(grad_data.flatten().tolist())
                    elif isinstance(grad_data, (list, tuple)):
                        flattened.extend([float(x) for x in grad_data])
                    else:
                        # Try to convert to numpy
                        try:
                            np_array = np.array(grad_data)
                            flattened.extend(np_array.flatten().tolist())
                        except (ValueError, TypeError) as e:
                            # Could not convert to numpy array, skip this gradient
                            sys.stderr.write(f"Warning: Could not convert gradient data to array: {e}\n")
            return flattened
        else:
            # Try direct array (fallback)
            if isinstance(package, (list, tuple)):
                import numpy as np
                if isinstance(package, np.ndarray):
                    return package.flatten().tolist()
                return [float(x) for x in package]
            return []
    except Exception as e:
        sys.stderr.write(f"Pickle deserialization error: {e}\n")
        sys.stderr.write(f"Error type: {type(e).__name__}\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
        return []


def deserialize_protobuf_gradient(data: bytes) -> List[float]:
    """Deserialize protobuf format gradient."""
    if not PROTOBUF_AVAILABLE:
        return []
    
    try:
        package = gradient_pb2.GradientPackage()
        package.ParseFromString(data)
        
        # Flatten all gradients into single array
        flattened = []
        for grad_tensor in package.gradients:
            # grad_tensor.data is repeated float (list)
            flattened.extend([float(x) for x in grad_tensor.data])
        
        return flattened
    except Exception as e:
        sys.stderr.write(f"Protobuf deserialization error: {e}\n")
        return []


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: deserialize_gradient.py <format> < input_file\n")
        sys.stderr.write("format: 'pickle' or 'protobuf'\n")
        sys.exit(1)
    
    format_type = sys.argv[1].lower()
    
    # Read input from stdin
    input_data = sys.stdin.buffer.read()
    
    # Deserialize based on format
    if format_type == "pickle":
        result = deserialize_pickle_gradient(input_data)
    elif format_type == "protobuf":
        result = deserialize_protobuf_gradient(input_data)
    else:
        sys.stderr.write(f"Unknown format: {format_type}\n")
        sys.exit(1)
    
    # Output as JSON array
    print(json.dumps(result))


if __name__ == "__main__":
    main()

