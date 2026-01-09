"""
RAG-Augmented DoRA Trainer

KRİTİK EKSİKLİK #3 ÇÖZÜMÜ: RAG + DoRA Training Entegrasyonu

Bu modül RAG context'ini DoRA training'e entegre eder:
1. Training batch'i için relevant context çeker
2. Context'i batch'e augment eder
3. Augmented data ile DoRA eğitimi yapar

Kullanım:
    from core.rag_augmented_trainer import RAGAugmentedDoRATrainer
    from core.dora_trainer import DoRATrainer
    from rag.retriever import RAGRetriever
    
    # Initialize components
    dora_trainer = DoRATrainer(model)
    rag_retriever = RAGRetriever()
    
    # Create augmented trainer
    trainer = RAGAugmentedDoRATrainer(
        dora_trainer=dora_trainer,
        rag_retriever=rag_retriever,
        context_top_k=3,
        augmentation_strategy="prepend",
    )
    
    # Train with RAG augmentation
    loss = trainer.train_step(batch)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AugmentedBatch:
    """Batch augmented with RAG context."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    original_texts: List[str]
    contexts: List[List[str]]
    augmented_texts: List[str]


class RAGAugmentedDoRATrainer:
    """
    DoRA trainer with RAG context augmentation.
    
    Combines RAG retrieval with DoRA training for improved
    domain-specific fine-tuning.
    """
    
    def __init__(
        self,
        dora_trainer: "DoRATrainer",
        rag_retriever: Optional["RAGRetriever"] = None,
        tokenizer: Optional[Any] = None,
        context_top_k: int = 3,
        augmentation_strategy: str = "prepend",
        context_separator: str = "\n\n---\n\n",
        max_context_length: int = 512,
        max_total_length: int = 2048,
        enable_rag: bool = True,
    ):
        """
        Initialize RAG-augmented DoRA trainer.
        
        Args:
            dora_trainer: Base DoRA trainer instance
            rag_retriever: RAG retriever for context lookup
            tokenizer: Tokenizer for re-tokenizing augmented texts
            context_top_k: Number of context documents to retrieve
            augmentation_strategy: How to augment ("prepend", "append", "interleave")
            context_separator: Separator between context and input
            max_context_length: Maximum context length in tokens
            max_total_length: Maximum total sequence length
            enable_rag: Whether to enable RAG augmentation
        """
        self.dora_trainer = dora_trainer
        self.rag_retriever = rag_retriever
        self.tokenizer = tokenizer
        self.context_top_k = context_top_k
        self.augmentation_strategy = augmentation_strategy
        self.context_separator = context_separator
        self.max_context_length = max_context_length
        self.max_total_length = max_total_length
        self.enable_rag = enable_rag and rag_retriever is not None
        
        # Statistics
        self._total_steps = 0
        self._rag_augmented_steps = 0
        self._avg_context_relevance = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Context cache for efficiency
        self._context_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._cache_max_size = 1000
        
        if self.enable_rag:
            logger.info(
                f"RAG-augmented training enabled: "
                f"top_k={context_top_k}, strategy={augmentation_strategy}, "
                f"max_context={max_context_length}, max_total={max_total_length}"
            )
        else:
            logger.info("RAG augmentation disabled, using standard DoRA training")

    def get_context_for_batch(
        self,
        texts: List[str],
    ) -> List[List[Tuple[str, float]]]:
        """
        Retrieve relevant context for each text in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (context, score) tuples for each text
        """
        if not self.enable_rag or self.rag_retriever is None:
            return [[] for _ in texts]
        
        all_contexts = []
        
        for text in texts:
            # Check cache first
            cache_key = text[:200]  # Use first 200 chars as cache key
            if cache_key in self._context_cache:
                self._cache_hits += 1
                all_contexts.append(self._context_cache[cache_key])
                continue
            
            self._cache_misses += 1
            
            try:
                # Search for relevant documents
                results = self.rag_retriever.search(
                    query=text,
                    top_k=self.context_top_k,
                )
                
                # Extract content and scores
                contexts = [
                    (result.content, result.score)
                    for result in results
                ]
                
                # Update cache
                if len(self._context_cache) < self._cache_max_size:
                    self._context_cache[cache_key] = contexts
                
                all_contexts.append(contexts)
                
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                all_contexts.append([])
        
        return all_contexts
    
    def augment_batch(
        self,
        texts: List[str],
        contexts: List[List[Tuple[str, float]]],
    ) -> List[str]:
        """
        Augment texts with retrieved context.
        
        Args:
            texts: Original input texts
            contexts: Retrieved contexts for each text
            
        Returns:
            Augmented texts
        """
        augmented = []
        
        for text, text_contexts in zip(texts, contexts):
            if not text_contexts:
                augmented.append(text)
                continue
            
            # Extract context strings (ignore scores)
            context_strs = [ctx for ctx, _ in text_contexts]
            
            # Truncate context if too long
            combined_context = self.context_separator.join(context_strs)
            if len(combined_context) > self.max_context_length:
                combined_context = combined_context[:self.max_context_length] + "..."
            
            # Apply augmentation strategy
            if self.augmentation_strategy == "prepend":
                # Context before input
                augmented_text = f"Context:\n{combined_context}{self.context_separator}Input:\n{text}"
            
            elif self.augmentation_strategy == "append":
                # Context after input
                augmented_text = f"Input:\n{text}{self.context_separator}Context:\n{combined_context}"
            
            elif self.augmentation_strategy == "interleave":
                # Interleave context with input (for longer texts)
                augmented_text = f"[Context: {combined_context}]\n{text}"
            
            else:
                # Default: prepend
                augmented_text = f"{combined_context}{self.context_separator}{text}"
            
            augmented.append(augmented_text)
        
        return augmented
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform one training step with RAG augmentation.
        
        Args:
            batch: Training batch (input_ids, attention_mask, labels)
            texts: Original texts (for RAG retrieval)
            
        Returns:
            Training metrics including loss
        """
        self._total_steps += 1
        
        # If RAG is disabled or no texts provided, use standard training
        if not self.enable_rag or texts is None:
            return self.dora_trainer.train_step(batch)
        
        try:
            # 1. Get RAG context for batch
            contexts = self.get_context_for_batch(texts)
            
            # 2. Check if any context was retrieved
            has_context = any(len(ctx) > 0 for ctx in contexts)
            
            if has_context:
                self._rag_augmented_steps += 1
                
                # 3. Augment texts with context
                augmented_texts = self.augment_batch(texts, contexts)
                
                # 4. Re-tokenize augmented texts
                # Note: This requires access to tokenizer
                # In practice, you'd pass tokenizer to __init__
                augmented_batch = self._tokenize_texts(augmented_texts, batch)
                
                # 5. Train with augmented batch
                result = self.dora_trainer.train_step(augmented_batch)
                
                # 6. Track context relevance
                avg_score = sum(
                    sum(score for _, score in ctx) / len(ctx)
                    for ctx in contexts if ctx
                ) / sum(1 for ctx in contexts if ctx)
                self._avg_context_relevance = (
                    self._avg_context_relevance * 0.99 + avg_score * 0.01
                )
                
                result["rag_augmented"] = True
                result["avg_context_relevance"] = avg_score
                
            else:
                # No context found, use standard training
                result = self.dora_trainer.train_step(batch)
                result["rag_augmented"] = False
            
            return result
            
        except Exception as e:
            logger.warning(f"RAG augmentation failed, falling back to standard: {e}")
            return self.dora_trainer.train_step(batch)
    
    def _tokenize_texts(
        self,
        texts: List[str],
        original_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize augmented texts using the provided tokenizer.
        
        Args:
            texts: Augmented text strings
            original_batch: Original batch for fallback and device info
            
        Returns:
            Tokenized batch ready for training
        """
        if self.tokenizer is None:
            # Fallback to original batch if no tokenizer
            logger.warning("No tokenizer provided, using original batch")
            return original_batch
        
        try:
            # Get device from original batch
            device = original_batch["input_ids"].device
            
            # Tokenize augmented texts
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_total_length,
                return_tensors="pt",
            )
            
            # Move to same device as original batch
            tokenized_batch = {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
            }
            
            # Create labels (same as input_ids for causal LM)
            # Shift labels for next-token prediction
            tokenized_batch["labels"] = tokenized_batch["input_ids"].clone()
            
            # Mask padding tokens in labels (-100 is ignored by CrossEntropyLoss)
            tokenized_batch["labels"][tokenized_batch["attention_mask"] == 0] = -100
            
            return tokenized_batch
            
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using original batch")
            return original_batch
    
    def train_epoch(
        self,
        dataloader: "DataLoader",
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Train for one epoch with RAG augmentation.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Epoch metrics
        """
        total_loss = 0.0
        total_steps = 0
        rag_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract texts if available
            texts = batch.pop("texts", None)
            
            # Train step
            result = self.train_step(batch, texts)
            
            total_loss += result.get("loss", 0.0)
            total_steps += 1
            
            if result.get("rag_augmented", False):
                rag_steps += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch}, Step {batch_idx}: "
                    f"loss={result.get('loss', 0):.4f}, "
                    f"rag_rate={rag_steps/total_steps:.2%}"
                )
        
        return {
            "epoch": epoch,
            "avg_loss": total_loss / total_steps if total_steps > 0 else 0,
            "total_steps": total_steps,
            "rag_augmented_steps": rag_steps,
            "rag_augmentation_rate": rag_steps / total_steps if total_steps > 0 else 0,
            "avg_context_relevance": self._avg_context_relevance,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )
        
        return {
            "total_steps": self._total_steps,
            "rag_augmented_steps": self._rag_augmented_steps,
            "rag_augmentation_rate": (
                self._rag_augmented_steps / self._total_steps
                if self._total_steps > 0 else 0
            ),
            "avg_context_relevance": self._avg_context_relevance,
            "rag_enabled": self.enable_rag,
            "context_top_k": self.context_top_k,
            "augmentation_strategy": self.augmentation_strategy,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._context_cache),
            "has_tokenizer": self.tokenizer is not None,
        }
    
    def clear_cache(self):
        """Clear the context cache."""
        self._context_cache.clear()
        logger.info("Context cache cleared")
    
    # Delegate other methods to base trainer
    def __getattr__(self, name):
        """Delegate unknown attributes to base trainer."""
        return getattr(self.dora_trainer, name)


def create_rag_augmented_trainer(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    rag_retriever: Optional["RAGRetriever"] = None,
    learning_rate: float = 1e-4,
    context_top_k: int = 3,
    augmentation_strategy: str = "prepend",
    max_context_length: int = 512,
    max_total_length: int = 2048,
    **kwargs,
) -> RAGAugmentedDoRATrainer:
    """
    Factory function to create RAG-augmented DoRA trainer.
    
    Args:
        model: Model with DoRA layers
        tokenizer: Tokenizer for re-tokenizing augmented texts
        rag_retriever: RAG retriever instance
        learning_rate: Learning rate for DoRA training
        context_top_k: Number of context documents
        augmentation_strategy: Augmentation strategy
        max_context_length: Maximum context length in tokens
        max_total_length: Maximum total sequence length
        **kwargs: Additional arguments for DoRATrainer
        
    Returns:
        Configured RAGAugmentedDoRATrainer
    """
    from core.dora_trainer import DoRATrainer
    
    # Create base DoRA trainer
    dora_trainer = DoRATrainer(
        model=model,
        learning_rate=learning_rate,
        **kwargs,
    )
    
    # Wrap with RAG augmentation
    return RAGAugmentedDoRATrainer(
        dora_trainer=dora_trainer,
        rag_retriever=rag_retriever,
        tokenizer=tokenizer,
        context_top_k=context_top_k,
        augmentation_strategy=augmentation_strategy,
        max_context_length=max_context_length,
        max_total_length=max_total_length,
    )
