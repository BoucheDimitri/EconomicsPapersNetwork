import pandas as pd
import numpy as np
import importlib
import networkx as nx
import CitNet.GraphCN as GraphCN
import scipy.sparse as sparse
import os

importlib.reload(GraphCN)


def topic_query(attrs_df, query_string, search_in=("title", "keywords")):
    """
    Return indexes corresponding to a given query

    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_string: (str) The query string
    :param search_in: (tuple) The columns to include for the query

    :return: list : the list of result indexes, one per column in search_in
    """
    inds = []
    for col in search_in:
        bool_ind = attrs_df[col].str.contains(query_string, regex=False)
        bool_ind = bool_ind.replace(np.nan, False)
        inds.append(attrs_df[bool_ind].index)
    return inds


def indexlist_inter(indexlist):
    """
    Intersection of all elements of a list of indexes

    :param indexlist: (list) : the list of pandas indexes (pandas.indexes.range.RangeIndex)

    :return: The intersected index ((pandas.indexes.range.RangeIndex)
    """
    first = True
    for ind in indexlist:
        if first:
            inter = ind
            first = False
        else:
            inter = inter.intersection(ind)
    return inter


def indexlist_union(indexlist):
    """
    Union of all elements of a list of indexes

    :param indexlist: (list) : the list of pandas indexes (pandas.indexes.range.RangeIndex)

    :return: The union index (pandas.indexes.range.RangeIndex)
    """
    first = True
    for ind in indexlist:
        if first:
            inter = ind
            first = False
        else:
            inter = inter.union(ind)
    return inter


def topic_subgraph_root(attrs_df, query_string, search_in=("title", "keywords"), how="union"):
    """
    Root nodes for building a subgraph relevant to a topic based query

    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_string: (str) The query string
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: the index of the nodes (pandas.indexes.range.RangeIndex)
    """
    inds = topic_query(attrs_df, query_string, search_in)
    if how == "inter":
        return indexlist_inter(inds)
    else:
        return indexlist_union(inds)


def similarity_subgraph_root(nodes_list, graph):
    """
    Root nodes for building a subgraph relevant to similiraty query

    :param nodes_list: the list of articles of our similar to request
    :param graph: (networkx.classes.digraph.DiGraph) the graph

    :return list of root nodes for our similarity request
    """
    root_nodes = []
    for node in nodes_list:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        root_nodes += successors + predecessors
    return list(set(root_nodes))


def expand_root(root_nodes, graph, d):
    """
    Expand root nodes by including their successors and some of their predecessors (d to be exact)

    :param root_nodes: (list-like) the roots nodes
    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?

    :return: the expanded nodes list (list).
    """
    nodes = []
    all_nodes = set(graph.nodes)
    root_nodes = set(root_nodes).intersection(all_nodes)
    for node in root_nodes:
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        if len(predecessors) >= d:
            np.random.shuffle(predecessors)
            new_nodes = set(successors + predecessors[0: d])
        else:
            new_nodes = set(successors + predecessors)
        nodes += new_nodes
    return nodes


def topic_query_subgraph(graph, d, attrs_df, query_string, search_in=("title", "keywords"), how="union"):
    """
    Find expanded subgraph for a query

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?
    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_string: (str) The query string
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: (networkx.classes.digraph.DiGraph) the expanded subgraph for the query
    """
    root_nodes = topic_subgraph_root(attrs_df, query_string, search_in, how)
    expanded = expand_root(root_nodes, graph, d)
    return graph.subgraph(expanded)


def similarity_query_subgraph(nodes_list, graph, d):
    """
    Find expanded subgraph for a query

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?
    :param nodes_list: the list of articles of our similar to request

    :return: (networkx.classes.digraph.DiGraph) the expanded subgraph for the similarity query
    """
    root_nodes = similarity_subgraph_root(nodes_list, graph)
    expanded = expand_root(root_nodes, graph, d)
    return graph.subgraph(expanded)


def iterate_hubs_auths(subgraph, k=20):
    """
    Compute hubs and authorities coefficients the iterative way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param k: number of iterations

    :return: x, y, nodes respectively vector of authorities coefs, hubs coefs and nodes ordering used for computations
    """
    nodes = list(subgraph)
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes) #.asfptype()
    n = len(nodes)
    y = np.ones((n, ))
    x = np.ones((n, ))
    for i in range(0, k):
        x = sparse.csr_matrix.dot(sparse.csr_matrix.transpose(A), y)
        y = sparse.csr_matrix.dot(A, x)
        x *= (1 / np.linalg.norm(x))
        y *= (1 / np.linalg.norm(y))
    return x, y, nodes


def sort_nodes(xy, nodes_list):
    """
    Return list of nodes sorted by authority coefs (xy = authorities coefs) or hubs coef (xy = hubs coefs)

    :param xy: authorities coefs vector or hubs coefs vector
    :param nodes_list: list of actual nodes in bijection with the range index of xy

    :return: nodes from nodes list sorted by authorities coefs or hubs coefs depending on what xy is.
    """
    xind = np.argsort(xy)
    return np.array(nodes_list)[xind]


def compute_authorities(subgraph, nauths=1):
    """
    Compute authorities coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param nauths: number of principal vectors wanted

    :return: xstar, nodes : respectively eigen vector stacked as columns and nodes ordering used for computations
    """
    nodes = list(subgraph.nodes)
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes).asfptype()
    AT = sparse.csr_matrix.transpose(A)
    ATA = sparse.csr_matrix.dot(AT, A)
    accept = False
    # Sometimes the sparse eigen solver does not converge to the right solution
    # Yielding a principal vector with very very small (order 1e-18), all negative components
    # When this is the case the result is rejected and the solver is launched again
    while not accept :
        w, xstar = sparse.linalg.eigs(ATA, k=nauths, which="LM")
        xstar = np.real(xstar)
        xstar[np.abs(xstar) < 1e-10] = 0
        accept = True
        for i in range(0, nauths):
            if np.all(xstar[:, i] == 0):
                accept = False
    return xstar, nodes


def compute_hubs(subgraph, nhubs=1, thresh=1e-10):
    """
    Compute hubs coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param nauths: number of principal vectors wanted

    :return: xstar, nodes : respectively eigen vectors stacked as columns and nodes ordering used for computations
    """
    nodes = list(subgraph.nodes)
    nodes.sort()
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes).asfptype()
    AT = sparse.csr_matrix.transpose(A)
    AAT = sparse.csr_matrix.dot(A, AT)
    accept = False
    # Sometimes the sparse eigen solver does not converge to the right solution
    # Yielding a principal vector with very very small (order 1e-18), all negative components
    # When this is the case the result is rejected and the solver is launched again
    while not accept :
        w, ystar = sparse.linalg.eigs(AAT, k=nhubs, which="LM")
        ystar = np.real(ystar)
        ystar[np.abs(ystar) < 1e-10] = 0
        accept =True
        for i in range(0, nhubs):
            if np.all(ystar[:, i] == 0):
                accept = False
    return ystar, nodes


def non_principal_auths(x, y, c)
    x_plus_inds = np.argwhere(x > 0)
    y_plus_inds = np.argwhere(y > 0)







# Path to the data
path = os.getcwd() + "/Tables/"

# Load attributes for terms matching
attrs = pd.read_csv(path + "attrs_nos.csv")

# Load refs and cites edges dataframes
cits_edgesdf = pd.read_csv(path + "cits_edges.csv")
refs_edgesdf = pd.read_csv(path + "refs_edges.csv")

# Convert them to list of edges
cits_edges = GraphCN.edgesdf_to_edgeslist(cits_edgesdf)
refs_edges = GraphCN.edgesdf_to_edgeslist(refs_edgesdf)

# Stack refs and cits edges together
all_edges = cits_edges + refs_edges

# Construct nx.DiGraph from stacked edges (refs + cits)
cits_refs_graph = nx.DiGraph(all_edges)


# ***************TEST ON TOPIC QUERY*********************************

# Create the expanded subgraph on which to perform the algo
d = 1000
qstring = "agency problems"
subtest_topic = topic_query_subgraph(cits_refs_graph, d, attrs, qstring)

# compute hubs and authorities in an iterative fashion
x, y, nodes = iterate_hubs_auths(subtest_topic, k=1000)

# compute authorities in the eigen vector search fasion
auths_eig, nodes_test = compute_authorities(subtest_topic, nauths=1)
hubs_eig, nodes_test = compute_hubs(subtest_topic, nhubs=1)

# nodes sorted by authority coef
top_auths_topic = sort_nodes(x, nodes)
print(top_auths_topic)

# nodes sorted by "hubness" coef
top_hubs_topic = sort_nodes(y, nodes)
print(top_hubs_topic)


# ***************TEST ON SIMILARITY QUERY*******************************

# Create the expanded subgraph on which to perform the algo
d = 1000
pages = [3168]
subtest_similarity = similarity_query_subgraph(pages, cits_refs_graph, d)

# compute hubs and authorities in an iterative fashion
x, y, nodes = iterate_hubs_auths(subtest_similarity, k=20)

# nodes sorted by authority coef
top_auths_similarity = sort_nodes(x, nodes)
print(top_auths_similarity)

# nodes sorted by "hubness" coef
top_hubs_similarity = sort_nodes(y, nodes)
print(top_hubs_similarity)




auths_eig, nodes_test = compute_authorities(subtest_topic, nauths=3)