# Copyright (c) 2017-2018 Symantec Corporation. All Rights Reserved. 
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
        