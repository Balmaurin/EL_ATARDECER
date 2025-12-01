
"""
Retrieval plugins for different search strategies.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def raptor_search(base: Path, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search through RAPTOR tree structure for relevant documents.

    Args:
        base: Base directory containing the RAPTOR index
        query: Search query string
        top_k: Number of top results to return

    Returns:
        List of dictionaries containing search results with scores
    """
    rdir = base / "index" / "raptor"
    if not rdir.exists():
        print("RAPTOR index not found")
        return []
    # Load and score all RAPTOR nodes
    files = sorted(list(rdir.glob("*.jsonl")))
    rows = []

    # Process query terms
    query_terms = set(query.lower().split())

    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            summary = obj.get("summary", "").lower()

            # Calculate term overlap score
            score = sum(1 for term in query_terms if term in summary)

            if score > 0:  # Only keep nodes with some relevance
                rows.append((score, obj))

        except json.JSONDecodeError as e:
            print(f"Error reading RAPTOR node {f}: {e}")
            continue

    # Sort by score in descending order
    rows.sort(key=lambda x: -x[0])
    # Select top results
    chosen = []
    max_results = max(top_k, 10)

    for score, obj in rows:
        if len(chosen) >= max_results:
            break

        # Take top 3 chunks from each matching node
        for chunk_id in obj.get("chunk_ids", [])[:3]:
            result = {
                "chunk_id": chunk_id,
                "doc_id": "",  # Could be enhanced with document tracking
                "title": obj.get("parent", ""),
                "text": "",  # Could be populated if needed
                "source": "raptor",
                "score": float(score),
                "quality": 0.5  # Default quality score
            }
            chosen.append(result)

    # Normalize scores if results exist
    if chosen:
        scores = np.array([hit["score"] for hit in chosen])
        min_score = float(scores.min())
        max_score = float(scores.max()) if len(scores) > 1 else (min_score + 1.0)

        # Normalize scores to [0,1] range
        for hit in chosen:
            hit["score"] = (hit["score"] - min_score) / (max_score - min_score + 1e-9)

    # Return only the requested number of results
    return chosen[:top_k]

def graph_search(base: Path, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search through knowledge graph for relevant documents.

    Args:
        base: Base directory containing the graph index
        query: Search query string
        top_k: Number of top results to return

    Returns:
        List of dictionaries containing search results with scores
    """
    # Check if graph index exists
    gdir = base / "index" / "graph"
    if not (gdir / "nodes.parquet").exists():
        print("Graph index not found")
        return []

    # Load graph nodes and process query
    try:
        nodes = pd.read_parquet(gdir / "nodes.parquet")
        mp = pd.read_parquet(base / "index" / "mapping.parquet")
    except Exception as e:
        print(f"Error loading graph index: {e}")
        return []

    # Extract query tokens
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens:
        return []

    # Find matching nodes
    try:
        match_nodes = nodes[
            nodes["id"].str.lower().apply(lambda x: any(tok in x for tok in q_tokens))
        ]["id"].tolist()
    except Exception as e:
        print(f"Error searching nodes: {e}")
        return []

    if not match_nodes:
        return []

    hits = []
    max_per_node = max(1, top_k // max(1, len(match_nodes)))

    # Process top matching nodes
    for node_id in match_nodes[:10]:
        try:
            # Get documents containing the node
            subset = mp[
                mp["text"].str.contains(node_id, case=False, regex=False)
            ].head(max_per_node)

            # Add matching documents to results
            for _, row in subset.iterrows():
                hits.append({
                    "chunk_id": row.get("chunk_id", ""),
                    "doc_id": row.get("doc_id", ""),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                    "source": "graph",
                    "score": 1.0,  # Base relevance score
                    "quality": 0.7  # Graph matches tend to be more precise
                })
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
            continue

    # Normalize scores if we have results
    if hits:
        # Downweight scores based on number of matching nodes
        score_adjustment = 1.0 / len(match_nodes)
        for hit in hits:
            hit["score"] *= score_adjustment

    return hits[:top_k]

