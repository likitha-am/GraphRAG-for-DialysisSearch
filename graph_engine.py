"""
graph_engine.py
Directed multi-relational knowledge graph for dialysis patient data.
Supports: efficient adjacency lookup, typed edges, multi-hop traversal.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Generator


class KnowledgeGraph:
    """
    Directed multi-relational graph.
    Nodes: patient_id, condition, treatment, outcome, risk_factor
    Edge types: HAS_CONDITION, RECEIVED_TREATMENT, HAD_OUTCOME,
                HAS_RISK_FACTOR, SIMILAR_TO, CONDITION_TREATED_BY,
                TREATMENT_LED_TO
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        # adjacency[src] = list of (dst, edge_type, weight)
        self._adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        # reverse index for incoming edges
        self._rev: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

    # ── Node management ──────────────────────────────────────────────────────

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        """Add or update a node."""
        self._nodes[node_id] = {"type": node_type, "id": node_id, **attrs}

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self._nodes.values() if n["type"] == node_type]

    # ── Edge management ──────────────────────────────────────────────────────

    def add_edge(
        self, src: str, dst: str, edge_type: str, weight: float = 1.0
    ) -> None:
        """Add a directed typed edge. Silently skips if either node is missing."""
        if src not in self._nodes or dst not in self._nodes:
            return
        # Avoid duplicate edges of the same type
        existing = {(d, t) for d, t, _ in self._adj[src]}
        if (dst, edge_type) not in existing:
            self._adj[src].append((dst, edge_type, weight))
            self._rev[dst].append((src, edge_type, weight))

    def neighbors(
        self, node_id: str, edge_type: str | None = None
    ) -> list[tuple[str, str, float]]:
        """Return [(dst, edge_type, weight)] for outgoing edges from node_id."""
        edges = self._adj.get(node_id, [])
        if edge_type:
            return [(d, t, w) for d, t, w in edges if t == edge_type]
        return list(edges)

    def reverse_neighbors(
        self, node_id: str, edge_type: str | None = None
    ) -> list[tuple[str, str, float]]:
        """Return [(src, edge_type, weight)] for incoming edges to node_id."""
        edges = self._rev.get(node_id, [])
        if edge_type:
            return [(s, t, w) for s, t, w in edges if t == edge_type]
        return list(edges)

    # ── BFS multi-hop traversal ───────────────────────────────────────────────

    def bfs(
        self,
        start: str,
        max_hops: int = 4,
        allowed_edge_types: set[str] | None = None,
    ) -> Generator[tuple[str, list[str]], None, None]:
        """
        Breadth-first traversal from start node.
        Yields (node_id, path) for each reachable node.
        Prevents cycles via visited set.
        """
        visited: set[str] = {start}
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            yield current, path

            if len(path) >= max_hops + 1:
                continue

            for dst, etype, _ in self._adj.get(current, []):
                if allowed_edge_types and etype not in allowed_edge_types:
                    continue
                if dst not in visited:
                    visited.add(dst)
                    queue.append((dst, path + [dst]))

    def find_paths(
        self,
        start: str,
        end: str,
        max_hops: int = 5,
        allowed_edge_types: set[str] | None = None,
    ) -> list[list[str]]:
        """Find all simple paths from start to end within max_hops."""
        results: list[list[str]] = []
        stack: list[tuple[str, list[str], set[str]]] = [
            (start, [start], {start})
        ]

        while stack:
            current, path, visited = stack.pop()
            if current == end:
                results.append(path)
                continue
            if len(path) > max_hops:
                continue
            for dst, etype, _ in self._adj.get(current, []):
                if allowed_edge_types and etype not in allowed_edge_types:
                    continue
                if dst not in visited:
                    stack.append((dst, path + [dst], visited | {dst}))

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        edge_count = sum(len(v) for v in self._adj.values())
        type_counts: dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            type_counts[n["type"]] += 1
        return {"nodes": len(self._nodes), "edges": edge_count, **dict(type_counts)}


def build_graph(entities: list[dict[str, Any]]) -> KnowledgeGraph:
    """
    Construct the knowledge graph from extracted patient entities.
    Adds: patient nodes, condition/treatment/outcome/risk_factor nodes,
    and all relational edges.
    """
    g = KnowledgeGraph()

    for e in entities:
        pid = e["patient_id"]

        # Patient node
        g.add_node(
            pid,
            node_type="patient",
            age=e.get("age"),
            prior_cardiac_event=e.get("prior_cardiac_event", False),
            outcome=e.get("outcome", "unknown"),
            conditions=e.get("conditions", []),
            treatments=e.get("treatments", []),
        )

        # Condition nodes + edges
        for cond in e.get("conditions", []):
            cid = f"cond:{cond}"
            if not g.get_node(cid):
                g.add_node(cid, node_type="condition", name=cond)
            g.add_edge(pid, cid, "HAS_CONDITION")

        # Treatment nodes + edges
        for tx in e.get("treatments", []):
            tid = f"tx:{tx}"
            if not g.get_node(tid):
                g.add_node(tid, node_type="treatment", name=tx)
            g.add_edge(pid, tid, "RECEIVED_TREATMENT")

        # Outcome node + edge
        outcome = e.get("outcome", "unknown")
        oid = f"outcome:{outcome}"
        if not g.get_node(oid):
            g.add_node(oid, node_type="outcome", name=outcome)
        g.add_edge(pid, oid, "HAD_OUTCOME")

        # Risk factor nodes + edges
        for rf in e.get("risk_factors", []):
            rfid = f"rf:{rf}"
            if not g.get_node(rfid):
                g.add_node(rfid, node_type="risk_factor", name=rf)
            g.add_edge(pid, rfid, "HAS_RISK_FACTOR")

    # Cross-edges: CONDITION_TREATED_BY and TREATMENT_LED_TO
    for e in entities:
        pid = e["patient_id"]
        for cond in e.get("conditions", []):
            for tx in e.get("treatments", []):
                g.add_edge(f"cond:{cond}", f"tx:{tx}", "CONDITION_TREATED_BY")
        for tx in e.get("treatments", []):
            g.add_edge(
                f"tx:{tx}",
                f"outcome:{e.get('outcome','unknown')}",
                "TREATMENT_LED_TO",
                weight=1.0 if e.get("outcome") == "improved" else 0.5,
            )

    return g

