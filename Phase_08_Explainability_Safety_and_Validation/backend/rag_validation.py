import requests
import numpy as np
from typing import List, Dict, Any

class RAGRetrievalValidator:
    def __init__(self, api_url: str = "http://127.0.0.1:8002/search"):
        self.api_url = api_url

    def query_phase7_api(self, query: str, source_filter: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        payload = {"query": query, "source_filter": source_filter, "top_k": top_k}
        try:
            response = requests.post(self.api_url, json=payload, timeout=5)
            return response.json() if response.status_code == 200 else []
        except Exception:
            return []

    def evaluate_retrieval_precision(self, queries: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        hits_at_k = 0
        reciprocal_ranks = []
        detailed_results = []

        for item in queries:
            query_text = item["query"]
            expected_manual = item["expected_manual"]
            retrieved_chunks = self.query_phase7_api(query_text, source_filter=item.get("source_filter"), top_k=top_k)

            match_found = False
            rank_hit = 0
            for idx, chunk in enumerate(retrieved_chunks, 1):
                chunk_source = chunk.get("source", "")
                if expected_manual.lower() in chunk_source.lower() or chunk_source.lower() in expected_manual.lower():
                    if not match_found:
                        match_found = True
                        rank_hit = idx
                        reciprocal_ranks.append(1.0 / idx)
                    break
            
            if match_found:
                hits_at_k += 1
            else:
                reciprocal_ranks.append(0.0)

            detailed_results.append({"query_id": item["query_id"], "query": query_text, "hit_at_k": match_found, "rank_hit": rank_hit})

        return {
            "total_queries_tested": len(queries),
            "top_k_evaluated": top_k,
            "precision_at_k": round(hits_at_k / len(queries) if queries else 0.0, 4),
            "mrr_score": round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4),
            "detailed_query_results": detailed_results
        }
