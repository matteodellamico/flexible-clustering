# Copyright © 2017-2018 Symantec Corporation. All Rights Reserved. 
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
from hdbscan import hdbscan_
import numpy as np
import scipy.sparse

from . import hnsw

def hnsw_hdbscan(data, d, m=5, ef=200, m0=None, level_mult=None,
                 heuristic=True, balanced_add=True, **kwargs):

    n = len(data)
    distance_matrix = scipy.sparse.lil_matrix((n, n))
    
    def decorated_d(i, j):
        res = d(data[i], data[j])
        distance_matrix[i, j] = distance_matrix[j, i] = res
        return res
    
    the_hnsw = hnsw.HNSW(decorated_d, m, ef, m0, level_mult, heuristic)
    add = the_hnsw.balanced_add if balanced_add else add
    for i in range(len(data)):
        add(i)

    return hdbscan.hdbscan(distance_matrix, metric='precomputed', **kwargs)

class UnionFind:
    """Union-find algorithm, with link-by-rank and path compression.
    
    See https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
    """
    
    def __init__(self, n):
        """n is the number of elements."""
        self.parents = np.arange(n)
        self.ranks = np.zeros(n)

    def find_root(self, x):
        """Return a representative for x's set."""
        i = x
        parents = self.parents
        while parents[i] != i:
            parents[x] = i = parents[i]
        return i
    
    def union(self, x, y):
        """Returns True if x and y were not in the same set."""
        parents, ranks = self.parents, self.ranks
        root_x, root_y = self.find_root(x), self.find_root(y)
        if root_x == root_y:
            return False
        rank_x, rank_y = ranks[root_x], ranks[root_y]
        if rank_x <= rank_y:
            if rank_x == rank_y:
                ranks[root_x] = rank_y + 1
            parents[root_x] = root_y
        else:
            parents[root_y] = root_x
        return True

class FISHDBC:
    """Flexible Incremental Scalable Hierarchical Density-Based Clustering."""

    def __init__(self, d, min_samples=5, m=5, ef=200, m0=None, level_mult=None,
                 heuristic=True, balanced_add=True, list_distance=False):
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

        # decorated_d will cache the computed distances in distance_cache.
        if not list_distance:  # d is defined to work on scalars
            def decorated_d(i, j):
                assert i == len(data) - 1 # 1st argument is the new item
                try:
                    return distance_cache[j]
                except KeyError:
                    distance_cache[j] = dist = d(data[i], data[j])
                    return dist
        if list_distance: # d is defined to work on a scalar and a list
            def decorated_d(i, js):
                assert i == len(data) - 1 # 1st argument is the new item
                known = []
                unknown_j, unknown_items = [], []
                for pos, j in enumerate(js):
                    k = j in distance_cache
                    known.append(k)
                    if not k:
                        unknown_j.append(j)
                        unknown_items.append(k)
                new_d = d(data[i], unknown_items)
                for j, dist in zip(unknown_j, new_d):
                    distance_cache[j] = dist
                old_d = (distance_cache[j] for j, k in zip(js, known) if k)
                new_d = iter(new_d)

                res = []
                for k in known:
                    if k:
                        res.append(next(old_d))
                    else:
                        res.append(next(new_d))
                return res

        # We create the HNSW
        the_hnsw = hnsw.HNSW(decorated_d, m, ef, m0, level_mult, heuristic,
                             list_distance)
        self._hnsw_add = (the_hnsw.balanced_add if balanced_add
                          else the_hnsw.add)
    
    def add(self, elem):
        """Add elem to the data structure."""
        
        data = self.data
        distance_cache = self._distance_cache
        min_samples = self.min_samples
        nh = self._neighbor_heaps
        new_edges = self._new_edges
        
        idx = len(data)
        data.append(elem)
        # let's start with min_samples values of infinity rather than
        # having to deal with heaps of less than min_samples values
        nh.append([(-np.infty, None)] * min_samples)
        
        self._distance_cache.clear()
        self._hnsw_add(idx)
        
        for j, dist in distance_cache.items():
            mdist = -dist
            heapq.heappushpop(nh[idx], (mdist, j))
            new_edges[idx, j] = dist

            # also update j's reachability distances
            nh_j = nh[j]
            old_mrd = heapq.heappushpop(nh_j, (mdist, idx))[0]
            new_mrd = nh_j[0][0]
            if old_mrd != new_mrd:
                # i is a new close neighbor for j and j's reachability
                # distance changed
                for md, k in nh_j:
                    if k == idx or k is None:
                        continue
                    if nh[k][0][0] > old_mrd:
                        # reachability distance between j and k decreased
                        new_edges[j, k] = -md

        if len(new_edges) >= idx:
            # update the MST when the candidate edges are more than
            # the ones in the MST itself, to avoid O(n^2) behavior
            self._update_mst()

    def _update_mst(self):
        """Update the minimum spanning tree."""
        
        candidate_edges = self._mst_edges
        new_edges = self._new_edges
        nh = self._neighbor_heaps
        
        candidate_edges.extend((max(dist, -nh[i][0][0], -nh[j][0][0]),
                                i, j, dist)
                               for (i, j), dist in new_edges.items())
        heapq.heapify(candidate_edges)
        
        # Kruskal's algorithm
        self._mst_edges = mst_edges = []
        n = len(self.data)
        uf = UnionFind(n)
        n_edges = 0
        while n_edges < n - 1:
            _, i, j, _ = edge = heapq.heappop(candidate_edges)
            if uf.union(i, j):
                mst_edges.append(edge)
                n_edges += 1
        new_edges.clear()
    
    def cluster(self, min_cluster_size=None, cluster_selection_method='eom',
                allow_single_cluster=False,
                match_reference_implementation=False):
        """Returns: (labels, probs, stabilities, condensed_tree, slt, mst)."""
        
        if min_cluster_size is None:
            min_cluster_size = self.min_samples
        self._update_mst()
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
