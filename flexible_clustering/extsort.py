#!/usr/bin/env python3

import heapq
import itertools
import struct
import tempfile

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

def extsort(data, tmp_nitems, dump, load, **kwargs):
    
    # Create temporary sorted files
    dataiter = iter(data)
    tmpfiles = []    
    while True:
        buf = sorted(itertools.islice(dataiter, tmp_nitems), **kwargs)
        if not buf: # we're done
            break
        f = tempfile.TemporaryFile()
        dump(buf, f)
        tmpfiles.append(f)

    # returns the merging of the files
    for f in tmpfiles:
        f.seek(0)
    yield from heapq.merge(*(load(f) for f in tmpfiles), **kwargs)
    for f in tmpfiles:
        f.close()

def struct_extsort(fmt, data, tmpsize, **kwargs):

    size = struct.calcsize(fmt)
    
    def dump(buf, f):
        for values in buf:
            f.write(struct.pack(fmt, *values))
    
    def load(f):
        for s in iter(lambda: f.read(size), b''):
            yield struct.unpack(fmt, s)

    return extsort(data, tmpsize, dump, load, **kwargs)

if __name__ == '__main__':
    
    RANGE = 10000000 # 10M
    NITEMS = 1000000 # 1M
    TMPSIZE = 10000 # 10K
    
    import random

    data = [(random.randrange(RANGE),) for _ in range(NITEMS)]
    sorted_data = sorted(data)
    extsorted_data = list(struct_extsort('i', data, TMPSIZE))
    assert extsorted_data == sorted_data
    
