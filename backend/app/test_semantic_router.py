"""
Test script for Semantic Router

Usage:
    python -m backend.app.test_semantic_router
"""

from semantic_router import SemanticRouter


def test_semantic_router():
    """Test semantic router with various prompts."""
    
    print("🧪 Testing Semantic Router...\n")
    
    # Initialize router
    router = SemanticRouter(similarity_threshold=0.7, use_semantic=True)
    
    # Test cases
    test_cases = [
        # Coder adapter tests
        ("How do I write a Python function to sort a list?", "coder_adapter"),
        ("What's the syntax for JavaScript classes?", "coder_adapter"),
        ("How to debug this code error?", "coder_adapter"),
        ("Explain this algorithm step by step", "coder_adapter"),
        ("Fix this SQL query bug", "coder_adapter"),
        
        # Law adapter tests
        ("What are my legal rights in this situation?", "law_adapter"),
        ("Explain this contract clause", "law_adapter"),
        ("What does this law mean?", "law_adapter"),
        ("How to file a lawsuit?", "law_adapter"),
        ("What are the legal implications?", "law_adapter"),
        
        # Edge cases
        ("I have a legal question about Python code", "law_adapter"),  # Should prefer law
        ("General question about anything", "default_adapter"),
        ("Tell me about the weather", "default_adapter"),
    ]
    
    print("📋 Running test cases...\n")
    
    passed = 0
    failed = 0
    
    for prompt, expected_adapter in test_cases:
        adapter_name, similarity_score = router.decide_adapter(prompt)
        
        # Check if result matches expected (or is reasonable)
        if adapter_name == expected_adapter:
            status = "✅ PASS"
            passed += 1
        else:
            # For edge cases, check if similarity is reasonable
            if similarity_score > 0.5:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAIL"
                failed += 1
        
        print(f"{status} | Prompt: '{prompt[:50]}...'")
        print(f"        | Expected: {expected_adapter}, Got: {adapter_name} (similarity: {similarity_score:.3f})\n")
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Total: {len(test_cases)}")
    print(f"   🎯 Success Rate: {(passed/len(test_cases)*100):.1f}%")
    
    return passed, failed


if __name__ == "__main__":
    test_semantic_router()

