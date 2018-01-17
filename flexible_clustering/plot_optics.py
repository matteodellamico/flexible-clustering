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


import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle
from matplotlib import gridspec

from .optics import extract_clusters

def do_plot(data, rh, minpts=5, labels=None, ksi=0.03, title='OPTICS'):

    fig = plt.figure(figsize=(9, 12))
    fig.canvas.set_window_title(title)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    def plot(color, title):
        plt.subplot(gs[0])
        plt.gca().clear()
        plt.title(title)
        plt.scatter(data[:, 0], data[:, 1], c=color, linewidth=0, s=40)
        plt.gca().set_aspect('equal')

        plt.subplot(gs[1])
        plt.gca().clear()
        plt.bar(range(len(data)), rh, color=color, linewidth=0)
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        ylim = ax.get_ylim()
        inftys = np.where(rh == np.inf)[0]
        plt.bar(inftys, [ylim[1]] * len(inftys),
                color=[color[i] for i in inftys],
                linewidth=0)
        plt.ylim(ylim)
        plt.grid()
        plt.ylabel("Reachability distance")
        
    
    if labels is not None:
        color_s = 'rgbcmyk'
        lcs = len(color_s)
        color = [color_s[l % lcs] for l in labels]
    else:
        color = 'k'
    
    plot(color, "Generated points")
    plt.tight_layout()
    plt.show(block=False)
    
    for start, end in cycle(extract_clusters(rh, minpts, ksi)):
        
        plt.waitforbuttonpress()

        color = ['k'] * len(data)
        for i in range(start, end):
            color[i] = 'r'

        plot(color, '{}-{} cluster'.format(start, end))
        plt.draw()
