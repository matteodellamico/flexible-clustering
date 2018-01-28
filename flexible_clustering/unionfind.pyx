import numpy as np
cimport numpy as np

cdef class UnionFind:
    """Union-find algorithm, with link-by-rank and path compression.
    
    See https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
    """

    cdef np.ndarray parents
    cdef np.ndarray ranks

    def __init__(self, n):
        """n is the number of elements."""

        self.parents = np.arange(n, dtype=np.intp)
        self.ranks = np.zeros(n, dtype=np.intp)

    cdef inline np.intp_t _find_root(self, np.intp_t x):
        cdef np.intp_t i = x
        while self.parents[i] != i:
            self.parents[x] = i = self.parents[i]
        return i

    def find_root(self, x):
        """Return a representative for x's set."""

        return self._find_root(x)

    def union(self, x, y):
        "Returns True if x and y were not in the same set."""

        cdef np.intp_t root_x = self._find_root(x)
        cdef np.intp_t root_y = self._find_root(y)

        if root_x == root_y:
            return False

        rank_x, rank_y = self.ranks[root_x], self.ranks[root_y]
        if rank_x <= rank_y:
            if rank_x == rank_y:
                self.ranks[root_x] = rank_y + 1
            self.parents[root_x] = root_y
        else:
            self.parents[root_y] = root_x
        return True
        