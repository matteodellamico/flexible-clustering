Flexible clustering
===================

A project for scalable hierachical clustering, thanks to a Flexible,
Incremental, Scalable, Hierarchical Density-Based Clustering
algorithms (FISHDBC, for the friends).

Dependencies
------------

* Python 3
* hdbscan: https://github.com/scikit-learn-contrib/hdbscan
* scipy: https://www.scipy.org/


Installation
------------

    python3 setup.py install

A projects allowing scalable hierarchical clustering, thanks to an
approximated version of OPTICS, on arbitrary data and distance measures.

Quickstart
----------

Look at the HDBSCAN documentation for the meaning of the return values
of the `cluster` method.  There are plenty of configuration options,
inherited by HNSWs and HDBSCAN, but the only compulsory argument is a
dissimilarity function between arbitrary data elements::

    import flexible_clustering
    
    clusterer = flexible_clustering.FISHDBC(my_dissimilarity)
    for elem in my_data:
        clusterer.add(elem)
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()

    for elem in some_new_data: # support cheap incremental clustering
        clusterer.add(elem)
    # new clustering according to the newly available data
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()


Demo/Example
------------

Look at the fishdbc_example.py file for something more (it requires
matplotlib to be run).

Want More Info?
---------------

Send me an email at matteo_dellamico@symantec.com. I'll improve the
docs as and if people use this.
    
Author
------

Matteo Dell'Amico

Copyright
---------

BSD 3-clause; see the LICENSE file.
