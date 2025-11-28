"""Vertex representing a Service Provider (SP)."""

from typing import Any, List

class Vertex:
    def __init__(self, id: str, **meta: Any) -> None:
        """
        Initialize a vertex with an identifier and optional metadata.

        Args:
            id: Vertex identifier
            **meta: Optional metadata
        """
        self.id: str = id
        self.meta: dict = meta
        self.neighbors: List['Vertex'] = []

    def add_neighbor(self, v: 'Vertex') -> None:
        """
        Add a neighboring vertex.

        Args:
            v: Vertex instance to add as neighbor
        """
        self.neighbors.append(v)

    def __str__(self) -> str:
        """Return string representation of the vertex."""
        return f'Vertex({self.id})'


'''
"""Vertex representing a Service Provider (SP)."""
class Vertex:
    def __init__(self, id, **meta):
        self.id = id
        self.meta = meta
        self.neighbors = []

    def addNeighbor(self, v):
        self.neighbors.append(v)

    def __str__(self):
        return f'Vertex({self.id})'
'''