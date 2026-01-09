"""
Keyword Router for DoRA Expert Selection

Fast, rule-based routing using keyword matching.
Stage 1 of the multi-stage routing pipeline.

Features:
- Regex-based keyword patterns
- Language detection
- Domain classification
- Task type detection
- Sub-millisecond latency
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result from keyword routing."""
    expert_id: str
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)
    category: str = "unknown"  # domain, language, task


class KeywordRouter:
    """
    Rule-based keyword router for fast expert selection.
    
    Uses regex patterns to match keywords and select appropriate experts.
    Designed for sub-millisecond latency.
    """
    
    # Domain expert keywords
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "medical_dora": [
            r"\b(hastalık|tedavi|ilaç|doktor|hastane|sağlık|tıp|ameliyat)\b",
            r"\b(disease|treatment|medicine|doctor|hospital|health|medical|surgery)\b",
            r"\b(symptom|diagnosis|patient|therapy|prescription|clinic)\b",
            r"\b(semptom|tanı|hasta|terapi|reçete|klinik)\b",
        ],
        "legal_dora": [
            r"\b(hukuk|kanun|mahkeme|avukat|dava|yasa|ceza|hak)\b",
            r"\b(law|legal|court|lawyer|case|statute|penalty|right)\b",
            r"\b(contract|lawsuit|attorney|judge|verdict|litigation)\b",
            r"\b(sözleşme|savcı|hakim|karar|anlaşma)\b",
        ],
        "coding_dora": [
            r"\b(kod|programlama|yazılım|hata|debug|fonksiyon|değişken)\b",
            r"\b(code|programming|software|bug|debug|function|variable)\b",
            r"\b(python|javascript|java|rust|golang|typescript|react|vue)\b",
            r"\b(api|database|server|frontend|backend|algorithm|git)\b",
        ],
        "finance_dora": [
            r"\b(finans|yatırım|borsa|hisse|kripto|banka|faiz|kredi)\b",
            r"\b(finance|investment|stock|crypto|bank|interest|credit)\b",
            r"\b(trading|portfolio|dividend|bond|forex|bitcoin|ethereum)\b",
            r"\b(ekonomi|enflasyon|döviz|altın|fon)\b",
        ],
        "science_dora": [
            r"\b(bilim|fizik|kimya|biyoloji|matematik|deney|teori)\b",
            r"\b(science|physics|chemistry|biology|math|experiment|theory)\b",
            r"\b(quantum|molecule|atom|cell|equation|hypothesis|research)\b",
            r"\b(kuantum|molekül|hücre|denklem|araştırma)\b",
        ],
        "history_dora": [
            r"\b(tarih|savaş|imparatorluk|devrim|antik|ortaçağ)\b",
            r"\b(history|war|empire|revolution|ancient|medieval)\b",
            r"\b(civilization|dynasty|battle|treaty|archaeology)\b",
            r"\b(uygarlık|hanedan|muharebe|antlaşma|arkeoloji)\b",
        ],
        "education_dora": [
            r"\b(eğitim|öğretim|okul|üniversite|sınav|ders|öğrenci)\b",
            r"\b(education|teaching|school|university|exam|lesson|student)\b",
            r"\b(curriculum|pedagogy|learning|classroom|teacher|course)\b",
            r"\b(müfredat|pedagoji|öğrenme|öğretmen|kurs)\b",
        ],
    }
    
    # Language detection patterns
    LANGUAGE_KEYWORDS: Dict[str, List[str]] = {
        "turkish_dora": [
            r"\b(merhaba|nasıl|nedir|için|ile|bir|bu|şu|ve|veya|ama)\b",
            r"[ğüşıöçĞÜŞİÖÇ]",  # Turkish-specific characters
            r"\b(değil|olarak|gibi|kadar|daha|çok|az|var|yok)\b",
        ],
        "german_dora": [
            r"\b(und|oder|aber|nicht|ist|sind|haben|werden|können)\b",
            r"[äöüßÄÖÜ]",  # German-specific characters
            r"\b(ich|du|er|sie|es|wir|ihr|Sie)\b",
        ],
        "french_dora": [
            r"\b(et|ou|mais|pas|est|sont|avoir|être|pouvoir)\b",
            r"[àâçéèêëîïôùûüÿœæ]",  # French-specific characters
            r"\b(je|tu|il|elle|nous|vous|ils|elles)\b",
        ],
        "spanish_dora": [
            r"\b(y|o|pero|no|es|son|tener|ser|poder)\b",
            r"[áéíóúüñ¿¡]",  # Spanish-specific characters
            r"\b(yo|tú|él|ella|nosotros|vosotros|ellos)\b",
        ],
        "arabic_dora": [
            r"[\u0600-\u06FF]",  # Arabic script
        ],
        "chinese_dora": [
            r"[\u4e00-\u9fff]",  # Chinese characters
        ],
        "japanese_dora": [
            r"[\u3040-\u309f\u30a0-\u30ff]",  # Hiragana and Katakana
        ],
        "korean_dora": [
            r"[\uac00-\ud7af]",  # Korean Hangul
        ],
    }
    
    # Task detection patterns
    TASK_KEYWORDS: Dict[str, List[str]] = {
        "summarization_dora": [
            r"\b(özetle|özet|kısalt|ana fikir|kısa ver)\b",
            r"\b(summarize|summary|brief|tldr|shorten|condense)\b",
        ],
        "translation_dora": [
            r"\b(çevir|tercüme|translate|translation)\b",
            r"\b(türkçe'ye|ingilizce'ye|almanca'ya)\b",
            r"\b(to english|to turkish|to german|to french)\b",
        ],
        "qa_dora": [
            r"\b(soru|cevap|yanıtla|açıkla|nedir|nasıl|neden|ne zaman)\b",
            r"\b(question|answer|explain|what is|how to|why|when)\b",
        ],
        "creative_dora": [
            r"\b(yaz|hikaye|şiir|senaryo|yaratıcı|hayal)\b",
            r"\b(write|story|poem|script|creative|imagine|fiction)\b",
        ],
        "analysis_dora": [
            r"\b(analiz|değerlendir|karşılaştır|incele)\b",
            r"\b(analyze|evaluate|compare|examine|assess|review)\b",
        ],
    }
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize keyword router.
        
        Args:
            confidence_threshold: Minimum confidence to return result
        """
        self.confidence_threshold = confidence_threshold
        
        # Compile regex patterns for performance
        self._domain_patterns = self._compile_patterns(self.DOMAIN_KEYWORDS)
        self._language_patterns = self._compile_patterns(self.LANGUAGE_KEYWORDS)
        self._task_patterns = self._compile_patterns(self.TASK_KEYWORDS)
        
        logger.info("KeywordRouter initialized")
    
    def _compile_patterns(
        self, keywords: Dict[str, List[str]]
    ) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for each expert."""
        compiled = {}
        for expert_id, patterns in keywords.items():
            compiled[expert_id] = [
                re.compile(p, re.IGNORECASE | re.UNICODE)
                for p in patterns
            ]
        return compiled
    
    def route(self, query: str) -> List[RouterResult]:
        """
        Route query to appropriate experts.
        
        Args:
            query: User query text
            
        Returns:
            List of RouterResult sorted by confidence (descending)
        """
        results = []
        
        # Check domain keywords
        for expert_id, patterns in self._domain_patterns.items():
            score, matches = self._match_patterns(query, patterns)
            if score > 0:
                results.append(RouterResult(
                    expert_id=expert_id,
                    confidence=score,
                    matched_keywords=matches,
                    category="domain",
                ))
        
        # Check language keywords
        for expert_id, patterns in self._language_patterns.items():
            score, matches = self._match_patterns(query, patterns)
            if score > 0:
                results.append(RouterResult(
                    expert_id=expert_id,
                    confidence=score,
                    matched_keywords=matches,
                    category="language",
                ))
        
        # Check task keywords
        for expert_id, patterns in self._task_patterns.items():
            score, matches = self._match_patterns(query, patterns)
            if score > 0:
                results.append(RouterResult(
                    expert_id=expert_id,
                    confidence=score,
                    matched_keywords=matches,
                    category="task",
                ))
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Filter by threshold
        results = [r for r in results if r.confidence >= self.confidence_threshold]
        
        return results
    
    def _match_patterns(
        self, text: str, patterns: List[re.Pattern]
    ) -> Tuple[float, List[str]]:
        """
        Match patterns against text and compute score.
        
        Args:
            text: Text to match
            patterns: List of compiled patterns
            
        Returns:
            Tuple of (score, matched_keywords)
        """
        matches = []
        total_matches = 0
        
        for pattern in patterns:
            found = pattern.findall(text)
            if found:
                total_matches += len(found)
                matches.extend(found[:3])  # Limit matches per pattern
        
        # Compute confidence based on match count
        # More matches = higher confidence, with diminishing returns
        if total_matches == 0:
            return 0.0, []
        
        # Logarithmic scaling: 1 match = 0.4, 2 = 0.6, 5 = 0.8, 10+ = 0.95
        import math
        confidence = min(0.95, 0.3 + 0.2 * math.log2(total_matches + 1))
        
        return confidence, matches[:5]  # Return top 5 matches
    
    def detect_language(self, query: str) -> Optional[str]:
        """
        Detect primary language of query.
        
        Args:
            query: Query text
            
        Returns:
            Language expert ID or None
        """
        best_score = 0.0
        best_lang = None
        
        for expert_id, patterns in self._language_patterns.items():
            score, _ = self._match_patterns(query, patterns)
            if score > best_score:
                best_score = score
                best_lang = expert_id
        
        return best_lang if best_score > 0.2 else None
    
    def detect_domain(self, query: str) -> Optional[str]:
        """
        Detect primary domain of query.
        
        Args:
            query: Query text
            
        Returns:
            Domain expert ID or None
        """
        best_score = 0.0
        best_domain = None
        
        for expert_id, patterns in self._domain_patterns.items():
            score, _ = self._match_patterns(query, patterns)
            if score > best_score:
                best_score = score
                best_domain = expert_id
        
        return best_domain if best_score > 0.3 else None
    
    def get_all_experts(self) -> Set[str]:
        """Get all known expert IDs."""
        experts = set()
        experts.update(self.DOMAIN_KEYWORDS.keys())
        experts.update(self.LANGUAGE_KEYWORDS.keys())
        experts.update(self.TASK_KEYWORDS.keys())
        return experts
