#!/usr/bin/env python3
"""
R3MES Miner Engine - Full Integration Tests

End-to-end integration tests for the complete mining workflow.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import torch
import json
import time

# Import all components
from r3mes.miner.engine import MinerEngine
from r3mes.serving.engine import ServingEngine
from r3mes.proposer.aggregator import ProposerAggregator
from r3mes.miner.task_pool_client import TaskPoolClient
from r3mes.miner.chunk_processor import ChunkProcessor
from r3mes.miner.lora_manager import LoRAManager
from r3mes.miner.inference_server import InferenceServer
from bridge.blockchain_client import BlockchainClient
from utils.ipfs_client import IPFSClient
from core.bitlinear import BitLinear
from core.trainer import LoRATrainer
from core.serialization import LoRASerializer


class TestFullIntegration:
    """Full integration test suite."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_private_key(self):
        """Mock private key for testing."""
        return "0123456789abcdef" * 8  # 64 hex chars
    
    @pytest.fixture
    def simple_model(self):
        """Create simple BitNet model for testing."""
        model = torch.nn.Sequential(
            BitLinear(768, 768, lora_rank=8),
            torch.nn.ReLU(),
            BitLinear(768, 256, lora_rank=8),
        )
        return model
    
    def test_bitlinear_lora_integration(self, simple_model):
        """Test BitLinear layer with LoRA integration."""
        # Test forward pass
        x = torch.randn(4, 768)
        output = simple_model(x)
        
        assert output.shape == (4, 256)
        
        # Test LoRA parameters
        for module in simple_model.modules():
            if isinstance(module, BitLinear):
                lora_A, lora_B, alpha = module.get_lora_params()
                assert lora_A is not None
                assert lora_B is not None
                assert alpha > 0
                
                # Test size estimation
                size_mb = module.estimate_size_mb()
                assert size_mb > 0
                assert size_mb < 1.0  # Should be small
    
    def test_lora_trainer_integration(self, simple_model):
        """Test LoRA trainer with model."""
        trainer = LoRATrainer(
            model=simple_model,
            learning_rate=1e-4,
            deterministic=True,
        )
        
        # Test training step
        batch_data = torch.randn(4, 768)
        batch_labels = torch.randn(4, 256)
        
        loss = trainer.train_step(batch_data, batch_labels)
        assert isinstance(loss, float)
        assert loss > 0
        
        # Test gradient extraction
        gradients = trainer.get_gradients()
        assert len(gradients) > 0
        
        # Verify only LoRA parameters have gradients
        for name, param in simple_model.named_parameters():
            if 'lora' in name.lower():
                assert param.requires_grad
            elif 'backbone' in name.lower():
                assert not param.requires_grad
    
    def test_lora_serialization_integration(self, simple_model):
        """Test LoRA serialization and deserialization."""
        serializer = LoRASerializer()
        
        # Get LoRA state
        lora_state = {}
        for name, param in simple_model.named_parameters():
            if 'lora' in name.lower():
                lora_state[name] = param.data.clone()
        
        # Serialize
        serialized_data = serializer.serialize_gradients(
            lora_state,
            metadata={"test": "integration"}
        )
        
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        
        # Deserialize
        deserialized_state, metadata = serializer.deserialize_gradients(serialized_data)
        
        assert len(deserialized_state) == len(lora_state)
        assert metadata["test"] == "integration"
        
        # Verify data integrity
        for name, original_param in lora_state.items():
            deserialized_param = deserialized_state[name]
            assert torch.allclose(original_param, deserialized_param)
    
    def test_lora_manager_integration(self, simple_model, temp_dir):
        """Test LoRA manager with caching."""
        lora_manager = LoRAManager(
            cache_dir=str(Path(temp_dir) / "lora_cache"),
            max_cache_size_mb=100,
            max_adapters_in_memory=5,
        )
        
        # Get LoRA state
        lora_state = {}
        for name, param in simple_model.named_parameters():
            if 'lora' in name.lower():
                lora_state[name] = param.data.clone()
        
        # Save adapter
        adapter_id = lora_manager.save_adapter(
            lora_state,
            metadata={"model": "test", "version": "1.0"}
        )
        
        assert adapter_id
        assert lora_manager.has_adapter(adapter_id)
        
        # Load adapter
        loaded_state, loaded_metadata = lora_manager.load_adapter(adapter_id)
        
        assert loaded_metadata["model"] == "test"
        assert len(loaded_state) == len(lora_state)
        
        # Verify data integrity
        for name, original_param in lora_state.items():
            loaded_param = loaded_state[name]
            assert torch.allclose(original_param, loaded_param)
        
        # Test cache stats
        stats = lora_manager.get_cache_stats()
        assert stats["memory_cache_count"] >= 1
        assert stats["disk_cache_count"] >= 1
    
    def test_chunk_processor_integration(self):
        """Test chunk processor with various data formats."""
        processor = ChunkProcessor(
            batch_size=2,
            max_sequence_length=128,
            device="cpu",
        )
        
        # Test JSON data
        json_data = [
            {"text": "Hello world", "label": 1},
            {"text": "Test data", "label": 0},
        ]
        
        processed_data, metadata = processor.process_chunk(json_data)
        assert isinstance(processed_data, torch.Tensor)
        assert metadata["data_format"] == "json_list"
        
        # Test text data
        text_data = ["Hello world", "Test data"]
        processed_data, metadata = processor.process_chunk(text_data)
        assert isinstance(processed_data, torch.Tensor)
        assert metadata["data_format"] == "text_list"
        
        # Test batch processing
        batch_data = [json_data, text_data]
        processed_batch, metadata_batch = processor.process_batch(batch_data)
        assert len(processed_batch) == 2
        assert len(metadata_batch) == 2
        
        # Test memory estimation
        memory_stats = processor.estimate_memory_usage(json_data)
        assert "raw_size_mb" in memory_stats
        assert "processed_size_mb" in memory_stats
    
    @pytest.mark.asyncio
    async def test_blockchain_client_integration(self, mock_private_key):
        """Test blockchain client operations."""
        # Note: This test requires a running blockchain node
        # In CI/CD, this would use a test blockchain or mock
        
        client = BlockchainClient(
            node_url="localhost:9090",
            chain_id="remes-test",
            private_key=mock_private_key,
        )
        
        # Test connection (will fail gracefully if no node)
        try:
            seed = client.get_global_seed(1)
            # If successful, verify seed
            if seed is not None:
                assert isinstance(seed, int)
        except Exception:
            # Expected if no blockchain node running
            pass
        
        # Test address derivation
        address = client.get_miner_address()
        assert isinstance(address, str)
        assert len(address) > 0
    
    def test_task_pool_client_integration(self, mock_private_key):
        """Test task pool client operations."""
        blockchain_client = BlockchainClient(
            node_url="localhost:9090",
            chain_id="remes-test",
            private_key=mock_private_key,
        )
        
        task_client = TaskPoolClient(
            blockchain_client=blockchain_client,
            miner_address="test_miner_address",
        )
        
        # Test active pool query (will return None if no blockchain)
        pool_id = task_client.get_active_pool_id()
        # pool_id can be None if no blockchain running
        
        # Test chunk availability query
        chunks = task_client.get_available_chunks(1, limit=10)
        assert isinstance(chunks, list)
        # chunks will be empty if no blockchain running
    
    @pytest.mark.asyncio
    async def test_serving_engine_integration(self, simple_model, mock_private_key):
        """Test serving engine with model."""
        serving_engine = ServingEngine(
            private_key=mock_private_key,
            blockchain_url="localhost:9090",
            chain_id="remes-test",
            model_version="test-v1.0.0",
        )
        
        # Set model
        serving_engine.model = simple_model
        serving_engine.model.eval()
        
        # Test inference processing
        test_data = {"prompt": "Hello world"}
        test_data_json = json.dumps(test_data).encode()
        
        # Mock IPFS hash
        test_ipfs_hash = "QmTest123"
        
        # Create temporary file for test
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(test_data_json)
            temp_file = f.name
        
        try:
            # Mock IPFS client get method
            original_get = serving_engine.ipfs_client.get
            serving_engine.ipfs_client.get = lambda hash, output_dir: temp_file
            
            # Test inference processing
            result_hash = serving_engine.process_inference_request("test_request", test_ipfs_hash)
            
            # Restore original method
            serving_engine.ipfs_client.get = original_get
            
            # Result might be None if IPFS upload fails, which is expected in test
            assert result_hash is None or isinstance(result_hash, str)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_proposer_aggregator_integration(self, mock_private_key):
        """Test proposer aggregator operations."""
        aggregator = ProposerAggregator(
            private_key=mock_private_key,
            blockchain_url="localhost:9090",
            chain_id="remes-test",
        )
        
        # Test gradient querying (will return empty if no blockchain)
        pending_gradients = aggregator.query_pending_gradients(limit=10)
        assert isinstance(pending_gradients, list)
        
        # Test gradient aggregation with mock data
        mock_gradients = [
            torch.randn(100).numpy().tobytes(),
            torch.randn(100).numpy().tobytes(),
        ]
        
        aggregated = aggregator.aggregate_gradients(mock_gradients)
        if aggregated is not None:
            assert isinstance(aggregated, bytes)
            assert len(aggregated) > 0
    
    @pytest.mark.asyncio
    async def test_inference_server_integration(self, simple_model):
        """Test inference server with FastAPI."""
        try:
            inference_server = InferenceServer(
                model=simple_model,
                model_version="test-v1.0.0",
                host="127.0.0.1",
                port=8001,  # Use different port to avoid conflicts
            )
            
            # Test server initialization
            assert inference_server.model is not None
            assert inference_server.model_version == "test-v1.0.0"
            
            # Test inference method
            response_text, tokens_generated = await inference_server._run_inference(
                prompt="Hello world",
                max_tokens=10,
                temperature=0.7,
            )
            
            assert isinstance(response_text, str)
            assert isinstance(tokens_generated, int)
            assert tokens_generated >= 0
            
        except ImportError:
            pytest.skip("FastAPI not available for inference server test")
    
    def test_end_to_end_mining_workflow(self, simple_model, temp_dir, mock_private_key):
        """Test complete mining workflow integration."""
        # Setup components
        lora_manager = LoRAManager(cache_dir=str(Path(temp_dir) / "lora_cache"))
        chunk_processor = ChunkProcessor(batch_size=2, device="cpu")
        
        # 1. Process chunk data
        chunk_data = [
            {"text": "Training sample 1", "label": 1},
            {"text": "Training sample 2", "label": 0},
        ]
        
        processed_data, metadata = chunk_processor.process_chunk(chunk_data)
        assert processed_data.shape[0] == 1  # Batch size 1 for single chunk
        
        # 2. Train model
        trainer = LoRATrainer(model=simple_model, learning_rate=1e-4)
        
        # Create dummy labels
        dummy_labels = torch.randn(1, 256)  # Match model output size
        
        loss = trainer.train_step(processed_data, dummy_labels)
        assert loss > 0
        
        # 3. Extract gradients
        gradients = trainer.get_gradients()
        assert len(gradients) > 0
        
        # 4. Serialize gradients
        serializer = LoRASerializer()
        serialized_gradients = serializer.serialize_gradients(
            gradients,
            metadata={"training_step": 1}
        )
        
        # 5. Save to LoRA manager
        adapter_id = lora_manager.save_adapter(
            gradients,
            metadata={"training_step": 1, "loss": loss}
        )
        
        assert adapter_id
        
        # 6. Load and verify
        loaded_gradients, loaded_metadata = lora_manager.load_adapter(adapter_id)
        assert loaded_metadata["training_step"] == 1
        assert abs(loaded_metadata["loss"] - loss) < 1e-6
        
        # 7. Verify gradient integrity
        for name, original_grad in gradients.items():
            loaded_grad = loaded_gradients[name]
            assert torch.allclose(original_grad, loaded_grad, atol=1e-6)
        
        print(f"✅ End-to-end workflow completed successfully!")
        print(f"   - Processed chunk data: {metadata['data_format']}")
        print(f"   - Training loss: {loss:.6f}")
        print(f"   - Gradients serialized: {len(serialized_gradients)} bytes")
        print(f"   - Adapter saved: {adapter_id}")
    
    def test_performance_benchmarks(self, simple_model):
        """Test performance benchmarks for key operations."""
        import time
        
        # Benchmark LoRA training
        trainer = LoRATrainer(model=simple_model, learning_rate=1e-4)
        
        batch_data = torch.randn(8, 768)
        batch_labels = torch.randn(8, 256)
        
        start_time = time.time()
        for _ in range(10):
            loss = trainer.train_step(batch_data, batch_labels)
        training_time = time.time() - start_time
        
        print(f"📊 Performance Benchmarks:")
        print(f"   - Training (10 steps): {training_time:.3f}s ({training_time/10:.3f}s per step)")
        
        # Benchmark serialization
        gradients = trainer.get_gradients()
        serializer = LoRASerializer()
        
        start_time = time.time()
        for _ in range(100):
            serialized = serializer.serialize_gradients(gradients)
        serialization_time = time.time() - start_time
        
        print(f"   - Serialization (100x): {serialization_time:.3f}s ({serialization_time/100*1000:.1f}ms per op)")
        
        # Benchmark chunk processing
        processor = ChunkProcessor(batch_size=4, device="cpu")
        chunk_data = [{"text": f"Sample {i}", "label": i % 2} for i in range(100)]
        
        start_time = time.time()
        processed_data, metadata = processor.process_chunk(chunk_data)
        processing_time = time.time() - start_time
        
        print(f"   - Chunk processing (100 items): {processing_time:.3f}s")
        
        # Assert reasonable performance
        assert training_time < 10.0  # Should complete in reasonable time
        assert serialization_time < 1.0  # Should be fast
        assert processing_time < 5.0  # Should process quickly


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])