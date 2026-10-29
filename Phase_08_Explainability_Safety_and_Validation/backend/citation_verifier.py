import re
from typing import List, Dict, Any

class CitationVerifier:
    def __init__(self, min_token_overlap: float = 0.35):
        self.min_token_overlap = min_token_overlap

    def extract_citations(self, response_text: str) -> List[str]:
        if not response_text:
            return []
        patterns = [r'\[(?:Source:\s*)?([A-Za-z0-9_\-\s]+)\]', r'\((?:Source:\s*)?([A-Za-z0-9_\-\s]+)\)']
        citations = []
        for pat in patterns:
            for match in re.findall(pat, response_text):
                cleaned = match.strip()
                if cleaned and cleaned not in citations:
                    citations.append(cleaned)
        return citations

    def compute_lexical_overlap(self, text_a: str, text_b: str) -> float:
        def tokenize(txt: str) -> set:
            if not txt:
                return set()
            words = re.findall(r'\b\w+\b', txt.lower())
            stopwords = {"the", "a", "an", "and", "or", "is", "in", "at", "to", "for", "with", "on", "patient"}
            return {w for w in words if w not in stopwords and len(w) > 2}

        set_a, set_b = tokenize(text_a), tokenize(text_b)
        return len(set_a.intersection(set_b)) / len(set_a) if set_a and set_b else 0.0

    def verify_response_grounding(self, generated_response: str, retrieved_context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        citations = self.extract_citations(generated_response or "")
        
        # Safely extract text and source handling None values
        context_texts = [str(chunk.get("text") or "") for chunk in retrieved_context_chunks if chunk]
        combined_context = " ".join(context_texts)
        retrieved_sources = {str(chunk.get("source") or "").lower() for chunk in retrieved_context_chunks if chunk}
        
        valid_citations = [c for c in citations if any(src in c.lower() or c.lower() in src for src in retrieved_sources if src)]
        invalid_citations = [c for c in citations if c not in valid_citations]
        
        grounding_score = self.compute_lexical_overlap(generated_response or "", combined_context)
        is_grounded = grounding_score >= self.min_token_overlap and len(invalid_citations) == 0

        warning = None
        if not citations:
            warning = "Missing explicit source citations."
        elif invalid_citations:
            warning = f"Uncertified citations found: {invalid_citations}"
        elif grounding_score < self.min_token_overlap:
            warning = f"Low lexical alignment ({grounding_score:.2f}). Possible hallucination."

        return {
            "is_fully_grounded": is_grounded,
            "has_valid_citations": bool(citations) and not invalid_citations,
            "extracted_citations": citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "grounding_overlap_score": round(grounding_score, 4),
            "hallucination_warning": warning
        }
