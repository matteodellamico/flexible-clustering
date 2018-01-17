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

import argparse
import collections
import itertools

import numpy as np
import numpy.linalg
import sklearn.datasets
import matplotlib.pyplot as plt

from flexible_clustering import FISHDBC
    
parser = argparse.ArgumentParser(description="""
Show an example of running FISHDBC.
This will plot points that are naturally clustered and added incrementally,
and then loop through all the hierarchical clusters recognized by the
algorithm.

Original clusters are shown in different colors while each cluster found by
FISHDBC is shown in red; press a key or click the mouse button to loop through
clusters.""")
parser.add_argument('--nitems', type=int, default=200,
                    help="Number of items (default 200).")
parser.add_argument('--niters', type=int, default=4,
                    help="Clusters are shown in NITERS stage while being "
                    "added incrementally (default 4).")
parser.add_argument('--centers', type=int, default=5,
                    help="Number of centers for the clusters generated "
                    "(default 5).")
args = parser.parse_args()

data, labels = sklearn.datasets.make_blobs(args.nitems,
                                           centers=args.centers)
x, y = data[:, 0], data[:, 1]

def distance(x, y):
    return numpy.linalg.norm(x - y)
    
fishdbc = FISHDBC(distance)

plt.figure(figsize=(9, 9))
plt.gca().set_aspect('equal')

for points in np.split(data, args.niters):
    for point in points:
        fishdbc.add(point)
    nknown = len(fishdbc.data)
    _, _, _, ctree, _, _ = fishdbc.cluster()
    clusters = collections.defaultdict(set)
    for parent, child, lambda_val, child_size in ctree[::-1]:
        if child_size == 1:
            clusters[parent].add(child)
        else:
            assert len(clusters[child]) == child_size
            clusters[parent].update(clusters[child])
    clusters = sorted(clusters.items())
    xknown, yknown = x[:nknown], y[:nknown]
    plt.scatter(xknown, yknown,
                c=['rgbcmyk'[l % 7] for l in labels], linewidth=0)
    plt.show(block=False)
    for _, cluster in clusters:
        plt.waitforbuttonpress()
        plt.gca().clear()
        c = ['kr'[i in cluster] for i in range(args.nitems)]
        plt.scatter(xknown, yknown, c=c, linewidth=0)
        plt.draw()
