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


def subgraph_root(attrs_df, query_string, search_in=("title", "keywords"), how="union"):
    """
    Root nodes for the subgraph generation of Hubs and Authorities

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


def query_subgraph(graph, d, attrs_df, query_string, search_in=("title", "keywords"), how="union"):
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
    root_nodes = subgraph_root(attrs_df, query_string, search_in, how)
    expanded = expand_root(root_nodes, graph, d)
    return graph.subgraph(expanded)


def compute_authorities(subgraph):
    nodes = list(subgraph)
    nodes.sort()
    print(nodes)
    A = nx.to_scipy_sparse_matrix(subgraph, nodelist=nodes).asfptype()
    AT = sparse.csr_matrix.transpose(A)
    ATA = sparse.csr_matrix.dot(AT, A)
    w, xstar = sparse.linalg.eigs(ATA, k=1)
    xstar = np.real(xstar).reshape((xstar.shape[0], ))
    print(xstar[:10])
    indsort = np.argsort(xstar)[::-1]
    nodes = np.array(nodes)
    return nodes[indsort]
    # return np.real(xstar)







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

d = 10000
qstring = "behavioral"
subtest = query_subgraph(cits_refs_graph, d, attrs, qstring)

# search_inds = topic_query(attrs, "monetary policy")
# inter_inds = indexlist_inter(search_inds)
# union_inds = indexlist_union(search_inds)

# root = subgraph_root(attrs, "monetary policy")
# expanded = expand_root(root, cits_refs_graph, 400)

authorities = compute_authorities(subtest)
print(max(authorities))
