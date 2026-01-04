"""
Tests for Router Module.
"""

import pytest
from router.keyword_router import KeywordRouter, RouterResult
from router.vram_adaptive_gating import VRAMAdaptiveGating, ExpertScore


class TestKeywordRouter:
    """Tests for KeywordRouter."""
    
    @pytest.fixture
    def router(self):
        return KeywordRouter()
    
    def test_medical_domain_detection(self, router):
        """Test medical domain keyword detection."""
        query = "Bu hastalık için tedavi var mı?"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "medical_dora" in expert_ids or "turkish_dora" in expert_ids
    
    def test_coding_domain_detection(self, router):
        """Test coding domain keyword detection."""
        query = "Python'da bir fonksiyon nasıl yazılır?"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "coding_dora" in expert_ids
    
    def test_legal_domain_detection(self, router):
        """Test legal domain keyword detection."""
        query = "Bu dava için hangi kanun geçerli?"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "legal_dora" in expert_ids
    
    def test_turkish_language_detection(self, router):
        """Test Turkish language detection."""
        query = "Merhaba, bugün hava nasıl?"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "turkish_dora" in expert_ids
    
    def test_summarization_task_detection(self, router):
        """Test summarization task detection."""
        query = "Bu metni özetle"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "summarization_dora" in expert_ids
    
    def test_translation_task_detection(self, router):
        """Test translation task detection."""
        query = "Translate this to Turkish"
        results = router.route(query)
        
        expert_ids = [r.expert_id for r in results]
        assert "translation_dora" in expert_ids
    
    def test_multiple_matches(self, router):
        """Test query matching multiple experts."""
        query = "Bu hastalığın tedavisi hakkında Türkçe bir özet yaz"
        results = router.route(query)
        
        # Should match medical, turkish, and summarization
        expert_ids = [r.expert_id for r in results]
        assert len(expert_ids) >= 2
    
    def test_no_match(self, router):
        """Test query with no keyword matches."""
        query = "xyz abc 123"
        results = router.route(query)
        
        # Should return empty or low confidence results
        assert len(results) == 0 or all(r.confidence < 0.3 for r in results)
    
    def test_detect_language(self, router):
        """Test language detection helper."""
        assert router.detect_language("Merhaba dünya nasılsın") == "turkish_dora"
        # German with umlaut characters
        assert router.detect_language("Ich möchte Käse und Bröt") == "german_dora"
    
    def test_detect_domain(self, router):
        """Test domain detection helper."""
        assert router.detect_domain("hastalık tedavi") == "medical_dora"
        assert router.detect_domain("python code function") == "coding_dora"
    
    def test_get_all_experts(self, router):
        """Test getting all known experts."""
        experts = router.get_all_experts()
        
        assert "medical_dora" in experts
        assert "turkish_dora" in experts
        assert "summarization_dora" in experts
    
    def test_confidence_ordering(self, router):
        """Test results are ordered by confidence."""
        query = "Python programlama kodu yazılım"
        results = router.route(query)
        
        if len(results) > 1:
            confidences = [r.confidence for r in results]
            assert confidences == sorted(confidences, reverse=True)


class TestVRAMAdaptiveGating:
    """Tests for VRAMAdaptiveGating."""
    
    @pytest.fixture
    def gating(self):
        return VRAMAdaptiveGating()
    
    def test_max_experts_property(self, gating):
        """Test max experts is set correctly."""
        assert gating.max_experts in [1, 2, 3]
    
    def test_select_single_expert(self, gating):
        """Test selecting single expert."""
        scores = [
            ExpertScore("medical_dora", 0.9, "keyword"),
        ]
        
        selected = gating.select(scores)
        assert len(selected) >= 1
        assert selected[0][0] == "medical_dora"
    
    def test_select_multiple_experts(self, gating):
        """Test selecting multiple experts."""
        scores = [
            ExpertScore("medical_dora", 0.9, "keyword"),
            ExpertScore("turkish_dora", 0.8, "keyword"),
            ExpertScore("coding_dora", 0.7, "keyword"),
        ]
        
        selected = gating.select(scores)
        assert len(selected) <= gating.max_experts + 1  # +1 for possible fallback
    
    def test_fallback_on_low_confidence(self, gating):
        """Test fallback when confidence is low."""
        scores = [
            ExpertScore("medical_dora", 0.3, "keyword"),  # Below threshold
        ]
        
        selected = gating.select(scores)
        expert_ids = [eid for eid, _ in selected]
        
        # Should include fallback
        assert "general_dora" in expert_ids
    
    def test_weights_sum_to_one(self, gating):
        """Test weights sum to approximately 1."""
        scores = [
            ExpertScore("medical_dora", 0.9, "keyword"),
            ExpertScore("turkish_dora", 0.8, "keyword"),
        ]
        
        selected = gating.select(scores)
        weights = [w for _, w in selected]
        
        assert abs(sum(weights) - 1.0) < 0.01
    
    def test_empty_scores(self, gating):
        """Test handling empty scores."""
        selected = gating.select([])
        
        # Should return fallback
        assert len(selected) == 1
        assert selected[0][0] == "general_dora"
    
    def test_select_with_budget(self, gating):
        """Test selection with VRAM budget."""
        scores = [
            ExpertScore("medical_dora", 0.9, "keyword"),
            ExpertScore("turkish_dora", 0.8, "keyword"),
            ExpertScore("coding_dora", 0.7, "keyword"),
        ]
        
        sizes = {
            "medical_dora": 50,
            "turkish_dora": 50,
            "coding_dora": 50,
            "general_dora": 50,
        }
        
        # Small budget - should limit selection
        selected = gating.select_with_budget(scores, sizes, vram_budget_mb=60)
        
        total_size = sum(sizes.get(eid, 50) for eid, _ in selected)
        assert total_size <= 60
    
    def test_merge_router_results(self, gating):
        """Test merging results from multiple routers."""
        keyword_results = [
            ("medical_dora", 0.8),
            ("turkish_dora", 0.6),
        ]
        
        semantic_results = [
            ("medical_dora", 0.7),
            ("coding_dora", 0.5),
        ]
        
        merged = gating.merge_router_results(
            keyword_results,
            semantic_results,
            keyword_weight=0.6,
            semantic_weight=0.4,
        )
        
        # medical_dora should have highest score (combined)
        assert merged[0].expert_id == "medical_dora"
        assert merged[0].source == "combined"
    
    def test_get_status(self, gating):
        """Test status reporting."""
        status = gating.get_status()
        
        assert 'vram_gb' in status
        assert 'max_experts' in status
        assert 'tier' in status
        assert status['tier'] in ['low', 'medium', 'high']


class TestExpertScore:
    """Tests for ExpertScore dataclass."""
    
    def test_comparison(self):
        """Test ExpertScore comparison."""
        score1 = ExpertScore("a", 0.8)
        score2 = ExpertScore("b", 0.6)
        
        assert score2 < score1
        assert sorted([score2, score1], reverse=True)[0] == score1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
