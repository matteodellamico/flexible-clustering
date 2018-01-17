#!/usr/bin/env python3

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

import dbm

from itertools import groupby
from operator import itemgetter
from struct import calcsize, iter_unpack, pack, unpack, unpack_from

from . import optics

from .extsort import extsort
from .hnsw import HNSW

def hnsw_distances_file(data, d, fn, m=5, ef=200, m0=None, level_mult=None,
                        heuristic=True, balanced=True, keyfmt='I', valfmt='f',
                        print_every=None, tmpsize=1024*1024*1024):

    tmpfmt = keyfmt + keyfmt + valfmt
    finalfmt = keyfmt + valfmt

    buf = []
    def decorated_d(i, j):
        dist = d(data[i], data[j])
        buf.append(pack(tmpfmt, i, j, dist))
        buf.append(pack(tmpfmt, j, i, dist))
        return dist
        
    hnsw = HNSW(decorated_d, m, ef, m0, level_mult, heuristic)
    add = hnsw.balanced_add if balanced else hnsw.add
    
    def cachedata():
        for i in range(len(data)):
            if print_every and i % print_every == 0 and i != 0:
                print(i)
            add(i)
            yield from buf
            buf.clear()

    tmpfmt_size = calcsize(tmpfmt)
    tmp_nitems = tmpsize // tmpfmt_size
    
    def dump(buf, f):
        for v in buf:
            f.write(v)

    def load(f):
        return iter(lambda: f.read(tmpfmt_size), b'')

    bindata = extsort(cachedata(), tmp_nitems, dump, load)
    unpacked = (unpack(tmpfmt, v) for v in bindata)
    grouped = groupby(unpacked, itemgetter(0))

    itemgetter1 = itemgetter(1)
    itemgetter2 = itemgetter(2)
    
    with dbm.open(fn, 'n') as db:
        for k, group in grouped:
            group = groupby(group, itemgetter1)
            group = sorted((next(g) for _, g in group), key=itemgetter2)
            group = (pack(finalfmt, idx, dist) for _, idx, dist in group)
            db[pack(keyfmt, k)] = b''.join(group)

    hnsw.distance = d
    return hnsw

def db_neighbors(db, keyfmt='I', valfmt='f'):
    fmt = keyfmt + valfmt
    def get_neighbors(i):
        return list((d, idx)
                    for idx, d in iter_unpack(fmt, db[pack(keyfmt, i)]))
    return get_neighbors

def hnsw_distances(data, d, m=5, ef=200, m0=None, level_mult=None,
                   heuristic=True, balanced=True, fmt='fI', print_every=None):

    distances = [bytearray() for _ in range(len(data))]
    def decorated_d(i, j):
        dist = d(data[i], data[j])
        distances[i].append(pack(fmt, dist, j))
        distances[j].extend(pack(fmt, dist, i))
        return dist
    
    hnsw = HNSW(decorated_d, m, ef, m0, level_mult, heuristic)
    add = hnsw.balanced_add if balanced else hnsw.add
    for i in range(len(data)):
        if print_every and i % print_every == 0 and i != 0:
            print(i)
        add(i)
    hnsw.distance = d

    size = calcsize(fmt)
    fmt0 = fmt[0]
    def sort_elem(dd):
        # assert len(dd) % size == 0
        def key(j):
            return unpack_from(fmt0, dd, size * j)
        ddsorted = bytearray()
        for j in sorted(range(len(dd) // size), key=key):
            start = size * j
            ddsorted.extend(dd[start:start+size])
        #assert ([d for d, _ in iter_unpack(fmt, ddsorted)]
        #        == sorted(d for d, _ in iter_unpack(fmt, ddsorted)))
        return ddsorted

    return hnsw, [sort_elem(dd) for dd in distances]


def get_get_neighbors(distances, fmt='fI'):
    def get_neighbors(i):
        return iter_unpack(fmt, distances[i])
    return get_neighbors


def compute_optics(data, d, minpts, m=5, ef=200, m0=None, level_mult=None,
                   heuristic=True, balanced=True, fmt='fI'):
    distances = hnsw_distances(data, d, m, ef, m0, level_mult, heuristic,
                               balanced, fmt)[1]
    print(sum(map(len, distances)) // 2 // calcsize(fmt))
    for i, dd in enumerate(distances):
        print(i, list(iter_unpack(fmt, dd))[:5])
    return optics.optics(len(data), minpts, get_get_neighbors(distances, fmt))

def optics_from_hnsw(hnsw, minpts):
    
    def get_neighbors(idx):
        return sorted((d, n) for n, d in hnsw[idx])
    
    return optics.optics(len(hnsw.data), minpts, get_neighbors)

if __name__ == '__main__':

    import argparse
    
    from numpy.linalg import norm
    
    from sklearn.datasets import make_blobs
    from plot_optics import do_plot

    parser = argparse.ArgumentParser()
    parser.add_argument('--nitems', type=int, default=100)
    parser.add_argument('--minpts', type=int, default=5)
    args = parser.parse_args()
    
    data, labels = make_blobs(args.nitems)

    def distance(x, y):
        return norm(x - y)

    hnsw = HNSW(distance)
    for p in data:
        hnsw.add(p)
    ordering, rh = optics_from_hnsw(hnsw, args.minpts)

    do_plot(data[ordering], rh, args.minpts, labels[ordering])
