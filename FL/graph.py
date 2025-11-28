"""Graph of Service Providers (SPs). Nodes are SPs; edges represent client relationships."""

from vertex import Vertex
from typing import Dict, List, Any

class Graph:
    def __init__(self) -> None:
        """Initialize an empty graph."""
        self.vertices: Dict[str, Vertex] = {}

    def add_vertex(self, id: str, **meta: Any) -> Vertex:
        """
        Add a vertex to the graph if not already present.

        Args:
            id: Vertex identifier
            **meta: Optional metadata for the vertex
        Returns:
            The Vertex instance
        """
        if id not in self.vertices:
            self.vertices[id] = Vertex(id, **meta)
        return self.vertices[id]

    def add_edge(self, src: str, dst: str) -> None:
        """
        Add an edge from src to dst vertex.

        Args:
            src: Source vertex id
            dst: Destination vertex id
        Raises:
            KeyError: If src or dst vertex is missing
        """
        if src not in self.vertices or dst not in self.vertices:
            raise KeyError(f'Vertex missing: {src} or {dst}')
        self.vertices[src].add_neighbor(self.vertices[dst])

    def get_vertices(self) -> List[Vertex]:
        """Return a list of all vertices in the graph."""
        return list(self.vertices.values())

    def __len__(self) -> int:
        """Return the number of vertices."""
        return len(self.vertices)


'''
"""Graph of Service Providers (SPs). Nodes are SPs; edges represent client relationships."""
from vertex import Vertex

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, id, **meta):
        if id not in self.vertices:
            self.vertices[id] = Vertex(id, **meta)
        return self.vertices[id]

    def add_edge(self, src, dst):
        if src not in self.vertices or dst not in self.vertices:
            raise KeyError('Vertex missing')
        self.vertices[src].addNeighbor(self.vertices[dst])

    def get_vertices(self):
        return list(self.vertices.values())

    def __len__(self):
        return len(self.vertices)
'''