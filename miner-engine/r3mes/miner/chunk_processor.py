#!/usr/bin/env python3
"""
R3MES Chunk Processor

Production-ready chunk processor that:
1. Processes fixed-size data chunks
2. Handles various data formats (JSON, text, binary, tensors)
3. Implements batch processing for efficiency
4. Manages memory usage for large datasets
5. Provides data loading pipeline for training
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Iterator, Tuple, Union
from pathlib import Path
import torch
import numpy as np

from r3mes.utils.logger import setup_logger


class ChunkProcessor:
    """Chunk processor for handling training data chunks."""
    
    def __init__(
        self,
        batch_size: int = 4,
        max_sequence_length: int = 512,
        device: str = "auto",
        log_level: str = "INFO",
        use_json_logs: bool = False,
    ):
        """
        Initialize chunk processor.
        
        Args:
            batch_size: Batch size for processing
            max_sequence_length: Maximum sequence length for text data
            device: Device to use ("auto", "cuda", "cpu")
            log_level: Logging level
            use_json_logs: Whether to use JSON-formatted logs
        """
        # Setup logger
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }
        self.logger = setup_logger(
            "r3mes.chunk_processor",
            level=log_level_map.get(log_level.upper(), logging.INFO),
            use_json=use_json_logs,
        )
        
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Chunk processor initialized (device: {self.device}, batch_size: {batch_size})")
    
    def detect_data_format(self, data: Any) -> str:
        """
        Detect the format of chunk data.
        
        Args:
            data: Chunk data
            
        Returns:
            Data format string ("json", "text", "tensor", "numpy", "binary", "unknown")
        """
        try:
            if isinstance(data, dict):
                return "json"
            elif isinstance(data, str):
                return "text"
            elif isinstance(data, torch.Tensor):
                return "tensor"
            elif isinstance(data, np.ndarray):
                return "numpy"
            elif isinstance(data, bytes):
                return "binary"
            elif isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return "json_list"
                elif data and isinstance(data[0], str):
                    return "text_list"
                else:
                    return "list"
            else:
                return "unknown"
        except Exception as e:
            self.logger.error(f"Error detecting data format: {e}")
            return "unknown"
    
    def parse_json_data(self, data: Union[Dict, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Parse JSON data into standardized format.
        
        Args:
            data: JSON data (dict or list of dicts)
            
        Returns:
            List of parsed data items
        """
        try:
            if isinstance(data, dict):
                # Single item
                return [data]
            elif isinstance(data, list):
                # List of items
                return data
            else:
                self.logger.error(f"Invalid JSON data type: {type(data)}")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing JSON data: {e}")
            return []
    
    def parse_text_data(self, data: Union[str, List[str]]) -> List[str]:
        """
        Parse text data into list of strings.
        
        Args:
            data: Text data (string or list of strings)
            
        Returns:
            List of text strings
        """
        try:
            if isinstance(data, str):
                # Split by lines or sentences
                if '\n' in data:
                    # Split by lines
                    lines = [line.strip() for line in data.split('\n') if line.strip()]
                    return lines
                else:
                    # Single text
                    return [data]
            elif isinstance(data, list):
                # List of strings
                return [str(item) for item in data]
            else:
                self.logger.error(f"Invalid text data type: {type(data)}")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing text data: {e}")
            return []
    
    def tokenize_text(self, texts: List[str], tokenizer=None) -> torch.Tensor:
        """
        Tokenize text data.
        
        Args:
            texts: List of text strings
            tokenizer: Optional tokenizer (if None, uses simple word tokenization)
            
        Returns:
            Tokenized tensor [batch_size, sequence_length]
        """
        try:
            if tokenizer is not None:
                # Use provided tokenizer
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt"
                )
                return encoded['input_ids']
            else:
                # Simple word tokenization (fallback)
                vocab = {}
                vocab_size = 0
                
                # Build vocabulary
                for text in texts:
                    words = text.lower().split()
                    for word in words:
                        if word not in vocab:
                            vocab[word] = vocab_size
                            vocab_size += 1
                
                # Add special tokens
                vocab['<pad>'] = vocab_size
                vocab['<unk>'] = vocab_size + 1
                
                # Tokenize texts
                tokenized = []
                for text in texts:
                    words = text.lower().split()[:self.max_sequence_length]
                    tokens = [vocab.get(word, vocab['<unk>']) for word in words]
                    
                    # Pad to max length
                    while len(tokens) < self.max_sequence_length:
                        tokens.append(vocab['<pad>'])
                    
                    tokenized.append(tokens)
                
                return torch.tensor(tokenized, dtype=torch.long)
                
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            # Return dummy tensor
            return torch.zeros((len(texts), self.max_sequence_length), dtype=torch.long)
    
    def process_tensor_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process tensor data.
        
        Args:
            data: Input tensor
            
        Returns:
            Processed tensor
        """
        try:
            # Move to device
            data = data.to(self.device)
            
            # Ensure proper shape for batch processing
            if data.dim() == 1:
                # Add batch dimension
                data = data.unsqueeze(0)
            
            return data
        except Exception as e:
            self.logger.error(f"Error processing tensor data: {e}")
            return torch.empty(0)
    
    def process_numpy_data(self, data: np.ndarray) -> torch.Tensor:
        """
        Process numpy data.
        
        Args:
            data: Input numpy array
            
        Returns:
            Processed tensor
        """
        try:
            # Convert to tensor
            tensor = torch.from_numpy(data)
            
            # Process as tensor
            return self.process_tensor_data(tensor)
        except Exception as e:
            self.logger.error(f"Error processing numpy data: {e}")
            return torch.empty(0)
    
    def create_batches(self, data: List[Any], batch_size: Optional[int] = None) -> Iterator[List[Any]]:
        """
        Create batches from data list.
        
        Args:
            data: List of data items
            batch_size: Batch size (uses self.batch_size if not provided)
            
        Yields:
            Batches of data items
        """
        batch_size = batch_size or self.batch_size
        
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def process_chunk(
        self,
        chunk_data: Any,
        tokenizer=None,
        return_format: str = "tensor"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process a data chunk.
        
        Args:
            chunk_data: Raw chunk data
            tokenizer: Optional tokenizer for text data
            return_format: Return format ("tensor", "dict", "list")
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        try:
            self.logger.debug(f"Processing chunk data: {type(chunk_data)}")
            
            # Detect data format
            data_format = self.detect_data_format(chunk_data)
            self.logger.debug(f"Detected data format: {data_format}")
            
            metadata = {
                "data_format": data_format,
                "original_type": str(type(chunk_data)),
                "device": str(self.device),
            }
            
            # Process based on format
            if data_format == "json":
                # JSON data
                parsed_data = self.parse_json_data(chunk_data)
                metadata["num_items"] = len(parsed_data)
                
                if return_format == "tensor":
                    # Convert to tensor (simplified - extract text fields)
                    texts = []
                    for item in parsed_data:
                        if isinstance(item, dict):
                            # Extract text fields
                            text_fields = []
                            for key, value in item.items():
                                if isinstance(value, str):
                                    text_fields.append(value)
                            texts.append(" ".join(text_fields))
                        else:
                            texts.append(str(item))
                    
                    processed_data = self.tokenize_text(texts, tokenizer)
                else:
                    processed_data = parsed_data
                    
            elif data_format in ["text", "text_list"]:
                # Text data
                parsed_data = self.parse_text_data(chunk_data)
                metadata["num_items"] = len(parsed_data)
                
                if return_format == "tensor":
                    processed_data = self.tokenize_text(parsed_data, tokenizer)
                else:
                    processed_data = parsed_data
                    
            elif data_format == "tensor":
                # Tensor data
                processed_data = self.process_tensor_data(chunk_data)
                metadata["tensor_shape"] = list(processed_data.shape)
                
            elif data_format == "numpy":
                # Numpy data
                processed_data = self.process_numpy_data(chunk_data)
                metadata["tensor_shape"] = list(processed_data.shape)
                
            elif data_format == "binary":
                # Binary data - try to deserialize
                try:
                    # Try pickle first
                    deserialized = pickle.loads(chunk_data)
                    return self.process_chunk(deserialized, tokenizer, return_format)
                except:
                    try:
                        # Try JSON
                        json_str = chunk_data.decode('utf-8')
                        deserialized = json.loads(json_str)
                        return self.process_chunk(deserialized, tokenizer, return_format)
                    except:
                        # Treat as raw bytes
                        self.logger.warning("Could not deserialize binary data, treating as raw bytes")
                        processed_data = torch.frombuffer(chunk_data, dtype=torch.uint8)
                        metadata["binary_size"] = len(chunk_data)
                        
            else:
                # Unknown format
                self.logger.warning(f"Unknown data format: {data_format}, converting to string")
                text_data = str(chunk_data)
                processed_data = self.tokenize_text([text_data], tokenizer)
            
            # Move to device if tensor
            if isinstance(processed_data, torch.Tensor):
                processed_data = processed_data.to(self.device)
                metadata["device"] = str(self.device)
            
            self.logger.debug(f"Chunk processed: {type(processed_data)}")
            return processed_data, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}", exc_info=True)
            # Return empty tensor and error metadata
            return torch.empty(0), {"error": str(e), "data_format": "error"}
    
    def process_batch(
        self,
        batch_data: List[Any],
        tokenizer=None,
        return_format: str = "tensor"
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Process a batch of chunks.
        
        Args:
            batch_data: List of chunk data
            tokenizer: Optional tokenizer
            return_format: Return format
            
        Returns:
            Tuple of (processed_data_list, metadata_list)
        """
        try:
            processed_data_list = []
            metadata_list = []
            
            for i, chunk_data in enumerate(batch_data):
                self.logger.debug(f"Processing batch item {i+1}/{len(batch_data)}")
                
                processed_data, metadata = self.process_chunk(
                    chunk_data, tokenizer, return_format
                )
                
                processed_data_list.append(processed_data)
                metadata_list.append(metadata)
            
            self.logger.debug(f"Batch processed: {len(processed_data_list)} items")
            return processed_data_list, metadata_list
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}", exc_info=True)
            return [], []
    
    def create_data_loader(
        self,
        chunk_data_list: List[Any],
        tokenizer=None,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[List[torch.Tensor], List[Dict[str, Any]]]]:
        """
        Create a data loader for chunk processing.
        
        Args:
            chunk_data_list: List of chunk data
            tokenizer: Optional tokenizer
            shuffle: Whether to shuffle data
            batch_size: Batch size (uses self.batch_size if not provided)
            
        Yields:
            Batches of (processed_data_list, metadata_list)
        """
        try:
            data = chunk_data_list.copy()
            
            if shuffle:
                import random
                random.shuffle(data)
            
            batch_size = batch_size or self.batch_size
            
            for batch in self.create_batches(data, batch_size):
                processed_batch, metadata_batch = self.process_batch(
                    batch, tokenizer, return_format="tensor"
                )
                yield processed_batch, metadata_batch
                
        except Exception as e:
            self.logger.error(f"Error creating data loader: {e}", exc_info=True)
    
    def estimate_memory_usage(self, chunk_data: Any) -> Dict[str, float]:
        """
        Estimate memory usage for processing chunk data.
        
        Args:
            chunk_data: Chunk data to analyze
            
        Returns:
            Dictionary with memory estimates (in MB)
        """
        try:
            import sys
            
            # Get size of raw data
            raw_size_bytes = sys.getsizeof(chunk_data)
            
            # Estimate processed size based on data format
            data_format = self.detect_data_format(chunk_data)
            
            if data_format in ["text", "text_list"]:
                # Text data - estimate tokenized size
                if isinstance(chunk_data, str):
                    num_tokens = len(chunk_data.split())
                elif isinstance(chunk_data, list):
                    num_tokens = sum(len(str(item).split()) for item in chunk_data)
                else:
                    num_tokens = 100  # Default estimate
                
                # Tensor size: num_tokens * sequence_length * 4 bytes (int32)
                processed_size_bytes = min(num_tokens, self.max_sequence_length) * 4
                
            elif data_format == "tensor":
                # Tensor data
                if hasattr(chunk_data, 'numel') and hasattr(chunk_data, 'element_size'):
                    processed_size_bytes = chunk_data.numel() * chunk_data.element_size()
                else:
                    processed_size_bytes = raw_size_bytes
                    
            elif data_format == "numpy":
                # Numpy data
                if hasattr(chunk_data, 'nbytes'):
                    processed_size_bytes = chunk_data.nbytes
                else:
                    processed_size_bytes = raw_size_bytes
                    
            else:
                # Other formats - use raw size as estimate
                processed_size_bytes = raw_size_bytes
            
            return {
                "raw_size_mb": raw_size_bytes / (1024 * 1024),
                "processed_size_mb": processed_size_bytes / (1024 * 1024),
                "estimated_peak_mb": (raw_size_bytes + processed_size_bytes) / (1024 * 1024),
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating memory usage: {e}")
            return {
                "raw_size_mb": 0.0,
                "processed_size_mb": 0.0,
                "estimated_peak_mb": 0.0,
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "device": str(self.device),
            "device_available": torch.cuda.is_available() if self.device.type == "cuda" else True,
        }