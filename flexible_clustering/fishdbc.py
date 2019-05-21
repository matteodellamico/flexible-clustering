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

import heapq

import hdbscan
import numpy as np
import scipy.sparse

from hdbscan import hdbscan_

from . import hnsw
from .unionfind import UnionFind

def hnsw_hdbscan(data, d, m=5, ef=200, m0=None, level_mult=None,
                 heuristic=True, balanced_add=True, **kwargs):
    """Simple implementation for when you don't need incremental updates."""

    n = len(data)
    distance_matrix = scipy.sparse.lil_matrix((n, n))
    
    def decorated_d(i, j):
        res = d(data[i], data[j])
        distance_matrix[i, j] = distance_matrix[j, i] = res
        return res
    
    the_hnsw = hnsw.HNSW(decorated_d, m, ef, m0, level_mult, heuristic)
    add = the_hnsw.balanced_add if balanced_add else the_hnsw.add
    for i in range(len(data)):
        add(i)

    return hdbscan.hdbscan(distance_matrix, metric='precomputed', **kwargs)


class FISHDBC:
    """Flexible Incremental Scalable Hierarchical Density-Based Clustering."""

    def __init__(self, d, min_samples=5, m=5, ef=200, m0=None, level_mult=None,
                 heuristic=True, balanced_add=True, vectorized=False):
        """Setup the algorithm. The only mandatory parameter is d, the
        dissimilarity function. min_samples is passed to hdbscan, and
        the other parameters are all passed to HNSW."""

        self.min_samples = min_samples
        
        self.data = data = []  # the data we're clustering
        
        self._mst_edges = []  # minimum spanning tree.
        # format: a list of (rd, i, j, dist) edges where nodes are
        # data[i] and data[j], dist is the dissimilarity between them, and rd
        # is the reachability distance.

        # (i, j) -> dist: the new candidates for the spanning tree
        # reachability distance will be computed afterwards
        self._new_edges = {}
        
        # for each data[i], _neighbor_heaps[i] contains a heap of
        # (mdist, j) where the data[j] are the min_sample closest distances
        # to i and mdist = -d(data[i], data[j]). Since heapq doesn't
        # currently support max-heaps, we use a min-heap with the
        # negative values of distances.
        self._neighbor_heaps = []

        # caches the distances computed to the last data item inserted
        self._distance_cache = distance_cache = {}


        self.cache_hits = self.cache_misses = 0
        # decorated_d will cache the computed distances in distance_cache.
        if not vectorized:  # d is defined to work on scalars
            def decorated_d(i, j):
                # assert i == len(data) - 1 # 1st argument is the new item
                if j in distance_cache:
                    self.cache_hits += 1
                    return distance_cache[j]
                self.cache_misses += 1
                distance_cache[j] = dist = d(data[i], data[j])
                return dist
        else: # d is defined to work on a scalar and a list
            def decorated_d(i, js):
                # assert i == len(data) - 1 # 1st argument is the new item
                res = [None] * len(js)
                unknown_j, unknown_pos = [], []
                for pos, j in enumerate(js):
                    if j in distance_cache:
                        res[pos] = distance_cache[j]
                    else:
                        unknown_j.append(j)
                        unknown_pos.append(pos)
                if len(unknown_j) > 0:
                    for pos, j, dist in zip(unknown_pos, unknown_j,
                                            d(data[i], unknown_j)):
                        distance_cache[j] = res[pos] = dist
                misses = len(unknown_j)
                self.cache_misses += misses
                self.cache_hits += len(js) - misses
                return res

        # We create the HNSW
        the_hnsw = hnsw.HNSW(decorated_d, m, ef, m0, level_mult, heuristic,
                             vectorized)
        self._hnsw_add = (the_hnsw.balanced_add if balanced_add
                          else the_hnsw.add)

    def __len__(self):
        return len(self.data)
    
    def add(self, elem):
        """Add elem to the data structure."""
        
        data = self.data
        distance_cache = self._distance_cache
        min_samples = self.min_samples
        nh = self._neighbor_heaps
        new_edges = self._new_edges

        minus_infty = -np.infty

        assert distance_cache == {}
        
        idx = len(data)
        data.append(elem)
        # let's start with min_samples values of infinity rather than
        # having to deal with heaps of less than min_samples values
        nh.append([(minus_infty, minus_infty)] * min_samples)

        self._hnsw_add(idx)
        
        for j, dist in distance_cache.items():
            mdist = -dist
            heapq.heappushpop(nh[idx], (mdist, j))
            new_edges[j, idx] = dist

            # also update j's reachability distances
            nh_j = nh[j]
            old_mrd = heapq.heappushpop(nh_j, (mdist, idx))[0]
            new_mrd = nh_j[0][0]
            if old_mrd != new_mrd:
                # i is a new close neighbor for j and j's reachability
                # distance changed
                for md, k in nh_j:
                    if k == idx or k == minus_infty:
                        continue
                    if nh[k][0][0] > old_mrd:
                        # reachability distance between j and k decreased
                        key = (j, k) if j < k else (k, j)
                        new_edges[key] = -min(md, new_mrd)
        distance_cache.clear()

    def update(self, elems, mst_update_rate=100000):
        """Add elements from elems and update the MST.

        To avoid clogging memory, the MST is updated every
        mst_update_rate elements are added.
        """

        for i, elem in enumerate(elems):
            self.add(elem)
            if i % mst_update_rate == 0:
                self.update_mst()
        self.update_mst()

    def update_mst(self):
        """Update the minimum spanning tree."""
        
        new_edges = self._new_edges

        if len(new_edges) == 0:
            return
        
        candidate_edges = self._mst_edges
        nh = self._neighbor_heaps

        candidate_edges.extend((max(dist, -nh[i][0][0], -nh[j][0][0]),
                                i, j, dist)
                               for (i, j), dist in new_edges.items())
        heapq.heapify(candidate_edges)
        
        # Kruskal's algorithm
        self._mst_edges = mst_edges = []
        n = len(self.data)
        needed_edges = n - 1
        uf = UnionFind(n)
        while needed_edges:
            _, i, j, _ = edge = heapq.heappop(candidate_edges)
            if uf.union(i, j):
                mst_edges.append(edge)
                needed_edges -= 1

        new_edges.clear()
    
    def cluster(self, min_cluster_size=None, cluster_selection_method='eom',
                allow_single_cluster=False,
                match_reference_implementation=False):
        """Returns: (labels, probs, stabilities, condensed_tree, slt, mst)."""
        
        if min_cluster_size is None:
            min_cluster_size = self.min_samples
        self.update_mst()
        mst = np.array(self._mst_edges).astype(np.double)
        mst = np.concatenate((mst[:, 1:3], mst[:, 0].reshape(-1, 1)), axis=1)
        slt = hdbscan_.label(mst)
        condensed_tree = hdbscan_.condense_tree(slt, min_cluster_size)
        stability_dict = hdbscan_.compute_stability(condensed_tree)
        lps = hdbscan_.get_clusters(condensed_tree,
                                    stability_dict,
                                    cluster_selection_method,
                                    allow_single_cluster,
                                    match_reference_implementation)
        return lps + (condensed_tree, slt, mst)
