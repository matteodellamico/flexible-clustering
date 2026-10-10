This project is now obsolete
============================

Head over to https://gitlab.com/bobtables/fishdbc for a new implementation of the same idea that is orders of magnitude faster.

Flexible clustering
===================

A project for scalable hierachical clustering, thanks to a Flexible,
Incremental, Scalable, Hierarchical Density-Based Clustering
algorithms (FISHDBC, for the friends).

This package lets you use an arbitrary dissimilarity function you write (or reuse from somebody else's work!) to cluster
your data.

Please see the paper at https://arxiv.org/abs/1910.07283

Dependencies
------------

* Python 3
* Cython
* hdbscan: https://github.com/scikit-learn-contrib/hdbscan
* scipy: https://www.scipy.org/


Installation
------------

    python3 setup.py install

Quickstart
----------

There are plenty of configuration options, inherited by HNSWs and HDBSCAN,
but the only compulsory argument is a dissimilarity function between arbitrary
data elements::

    import flexible_clustering
    
    clusterer = flexible_clustering.FISHDBC(my_dissimilarity)
    for elem in my_data:
        clusterer.add(elem)
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()

    for elem in some_new_data: # support cheap incremental clustering
        clusterer.add(elem)
    # new clustering according to the newly available data
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()

Make sure to run everything from *outside* the source directory, to
avoid confusing Python path.

Return Values
-------------

As documented in the `HDBSCAN source code <https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/hdbscan_.html>`_:

labels : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.

probabilities : ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are assigned
        0.

cluster_persistence : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.

single_linkage_tree : ndarray, shape (n_samples - 1, 4)
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

min_spanning_tree : ndarray, shape (n_samples - 1, 3)
        The minimum spanning as an edgelist.

Demo/Example
------------

Look at the fishdbc_example.py file for something more (it requires
matplotlib to be run).

Want More Info?
---------------

Send me an email at `della@linux.it`. I'll improve the
docs as and if people use this.
    
Author
------

Matteo Dell'Amico

Copyright
---------

BSD 3-clause; see the LICENSE file.
