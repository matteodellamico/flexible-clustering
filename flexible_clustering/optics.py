#!/usr/bin/env python3

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

import re

from bisect import bisect, insort
from itertools import cycle, tee

import numpy as np

from . import pdict

def optics_iter(n, minpts, get_neighbors):
    """get_neighbors(p) yields sorted (distance, node) pairs"""

    if minpts < 2:
        raise ValueError("minpts should be at least 2")

    unprocessed = set(range(n))
    seeds = pdict.priority_dict()
    neighbors_dtype = [('dist', float), ('o', int)]

    for _ in range(n):
        try:
            p, d = seeds.pop_smallest()
        except IndexError:
            p, d = unprocessed.pop(), np.inf
        else:
            unprocessed.remove(p)
        yield p, d
        neighbors = np.fromiter(get_neighbors(p), neighbors_dtype)
        try:
            coredist = neighbors[minpts - 2][0]
        except IndexError:
            continue
        neighbors['dist'][:minpts - 2] = coredist
        for newdist, o in neighbors:
            # assert newdist >= coredist
            if o not in unprocessed:
                continue
            if newdist < seeds.get(o, np.inf):
                seeds[o] = newdist

                
def optics(n, minpts, get_neighbors):
    ordering = np.zeros(n, int)
    reach_hist = np.zeros(n, float)
    for i, (p, d) in enumerate(optics_iter(n, minpts, get_neighbors)):
        ordering[i] = p
        reach_hist[i] = d
    return ordering, reach_hist

    
def extract_clusters(r, minpts, ksi):

    one_minus_ksi = 1 - ksi

    def categorize():
        a, b = tee(r)
        next(b, None)
        for pred, succ in zip(a, b):
            if pred < succ:
                yield 'U' if pred <= one_minus_ksi * succ else 'u'
            elif pred > succ:
                yield 'D' if succ <= one_minus_ksi * pred else 'd'
            else:
                yield '='
        yield 'U'

    down_re = re.compile('(?:D+[d=]{0,' + str(minpts) + '})*D')
    up_re = re.compile('(?:U+[u=]{0,' + str(minpts) + '})*U')
        
    pt_types = ''.join(categorize())
    
    steepdown = []
    
    def update_steepdown(mib):
        idx = bisect(steepdown, [mib / one_minus_ksi, 0, 0])
        del steepdown[:idx]
        for i, (_, _, this_mib) in enumerate(steepdown):
            if mib > this_mib:
                steepdown[i][2] = mib

    clusters = []
    idx = mib = 0
    while idx < len(r):
        mib = max(mib, r[idx])
        down_match = down_re.match(pt_types, idx)
        if down_match:
            update_steepdown(mib)
            insort(steepdown, [r[idx], idx, 0])
            idx = down_match.end()
            mib = r[idx]
            continue
        up_match = up_re.match(pt_types, idx)
        if up_match:
            update_steepdown(mib)
            idx = up_match.end()
            try:
                mib = r[idx]
            except IndexError:
                mib = np.inf
            for sd, start, this_mib in steepdown:
                if this_mib > mib:
                    continue
                if sd * one_minus_ksi >= mib:
                    try:
                        s = next(i - 1 for i in range(start + 1, idx)
                                 if r[i] <= mib)
                    except StopIteration:
                        continue
                    e = idx
                elif sd < mib * one_minus_ksi:
                    s = start
                    try:
                        e = next(i + 1 for i in reversed(range(start, idx))
                                 if r[i] < sd)
                    except StopIteration:
                        continue
                else:
                     s, e = start, idx
                if e - s < minpts:
                    continue
                yield s, e
        else:
            idx += 1
                
        
        # check if we're at the beginning of a steep down area
    #     if r[idx] >= (1 - ksi) * r[idx + 1]:
    #         for eidx in range(idx + 1, len(r)):
    #             if r[eidx] > r[eidx + 1]:
    #                 break


def hierarchy(rh, minpts, ksi):
    
    def cluster_size(ab):
        a, b = ab
        return b - a
    
    clusters = sorted(extract_clusters(rh, minpts, ksi),
                      key=cluster_size, reverse=True)
    levels = []
    for c in clusters:
        for level in levels:
            idx = bisect(level, c)
            if ((idx == 0 or level[idx - 1][1] <= c[0])
                and (idx == len(level) or level[idx][0] >= c[1])):
                level.insert(idx, c)
                break
        else:
            levels.append([c])

    return levels


if __name__ == '__main__':

    from sklearn.datasets import make_blobs

    from plot_optics import do_plot

    N = 100
    MINPTS = 5

    data, labels = make_blobs(N)

    def distance(x, y):
        return np.linalg.norm(x - y)
    
    def get_neighbors(i):
        return sorted((distance(data[i], data[j]), j) for j in range(N)
                      if i != j)

    ordering, rh = optics(N, MINPTS, get_neighbors)

    do_plot(data[ordering], rh, MINPTS, labels[ordering])
