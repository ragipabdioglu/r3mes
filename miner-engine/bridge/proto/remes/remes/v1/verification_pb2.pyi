from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RunForwardPassRequest(_message.Message):
    __slots__ = ("weights_ipfs_hash", "batch_id", "model_config_id", "dataset_ipfs_hash", "timeout_seconds")
    WEIGHTS_IPFS_HASH_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_IPFS_HASH_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    weights_ipfs_hash: str
    batch_id: str
    model_config_id: int
    dataset_ipfs_hash: str
    timeout_seconds: int
    def __init__(self, weights_ipfs_hash: _Optional[str] = ..., batch_id: _Optional[str] = ..., model_config_id: _Optional[int] = ..., dataset_ipfs_hash: _Optional[str] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class RunForwardPassResponse(_message.Message):
    __slots__ = ("loss_int", "success", "error_message", "execution_time_ms", "gpu_info")
    LOSS_INT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    GPU_INFO_FIELD_NUMBER: _ClassVar[int]
    loss_int: int
    success: bool
    error_message: str
    execution_time_ms: int
    gpu_info: str
    def __init__(self, loss_int: _Optional[int] = ..., success: bool = ..., error_message: _Optional[str] = ..., execution_time_ms: _Optional[int] = ..., gpu_info: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "version", "available_gpus", "model_loaded")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_GPUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    version: str
    available_gpus: _containers.RepeatedScalarFieldContainer[str]
    model_loaded: bool
    def __init__(self, healthy: bool = ..., version: _Optional[str] = ..., available_gpus: _Optional[_Iterable[str]] = ..., model_loaded: bool = ...) -> None: ...
