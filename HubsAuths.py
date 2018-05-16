import pandas as pd
import numpy as np
import importlib
import networkx as nx
import CitNet.GraphCN as GraphCN
import scipy.sparse as sparse
import seaborn as sns
import os
import time
import matplotlib.pyplot as plt

importlib.reload(GraphCN)


def topic_query(attrs_df, query_list, search_in=("title", "keywords")):
    """
    Return indexes corresponding to a given query

    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list: list of keywords
    :param search_in: (tuple) The columns to include for the query

    :return: list : the list of result indexes, one per column in search_in
    """
    inds = []
    for col in search_in:
        first = True
        for query in query_list:
            bool_ind = attrs_df[col].str.contains(query, regex=False)
            bool_ind = bool_ind.replace(np.nan, False)
            if first:
                inds_query = attrs_df[bool_ind].index
                first = False
            else:
                inds_query = inds_query.intersection(attrs_df[bool_ind].index)
        inds.append(inds_query)
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


def topic_subgraph_root(attrs_df, query_list, search_in=("title", "keywords"), how="union"):
    """
    Root nodes for building a subgraph relevant to a topic based query

    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list (list) : list of keywords
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: the index of the nodes (pandas.indexes.range.RangeIndex)
    """
    inds = topic_query(attrs_df, query_list, search_in)
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


def topic_query_subgraph(graph, d, attrs_df, query_list, search_in=("title", "keywords"), how="union"):
    """
    Find expanded subgraph for a query

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param d: how many predecessors to include at most ?
    :param attrs_df: (pandas.core.frame.DataFrame) Dataframe on which to perform the query
    :param query_list: (list) List of keywords
    :param search_in: (tuple) The columns to include for the query
    :param how: (str) How to join the indexes in the list ?

    :return: (networkx.classes.digraph.DiGraph) the expanded subgraph for the query
    """
    root_nodes = topic_subgraph_root(attrs_df, query_list, search_in, how)
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
    results_df = pd.DataFrame(index=nodes)
    results_df["xauth_0"] = x
    results_df["xhubs_0"] = y
    return results_df


def compute_authorities(subgraph, neigs=1):
    """
    Compute authorities coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted

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
        w, xstar = sparse.linalg.eigs(ATA, k=neigs, which="LM")
        xstar = np.real(xstar)
        xstar[np.abs(xstar) < 1e-10] = 0
        accept = True
        for i in range(0, neigs):
            if np.all(xstar[:, i] <= 0):
                accept = False
    return xstar, nodes


def compute_hubs(subgraph, neigs=1):
    """
    Compute hubs coefficients the eigen vectors way

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted

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
        w, ystar = sparse.linalg.eigs(AAT, k=neigs, which="LM")
        ystar = np.real(ystar)
        ystar[np.abs(ystar) < 1e-10] = 0
        accept = True
        for i in range(0, neigs):
            if np.all(ystar[:, i] <= 0):
                accept = False
    return ystar, nodes


def hubs_authorities_eigen(subgraph, neigs=1):
    """
    Wraps the result from compute_hubs and compute_authorities functions in a dataframe

    :param subgraph: a subgraph (networkx.classes.digraph.DiGraph)
    :param neigs: number of principal vectors wanted

    :return: Dataframe containing the principal vectors, indexed with nodes
    """
    xstar, nodes = compute_authorities(subgraph, neigs)
    ystar, nodes = compute_hubs(subgraph, neigs)
    results_df = pd.DataFrame(index=nodes)
    for i in range(0, neigs):
        results_df["xauth_" + str(i)] = xstar[:, i]
        results_df["xhub_" + str(i)] = ystar[:, i]
    return results_df


def non_principal_authorities(eigs_vecs_df, c):
    """
    Find c top authorities from set of non principal eigen vectors

    :param eigs_vecs_df: DataFrame, should have 2 columns : the i-th eigen vector of ATA and the i-th eigen vector of AAT
    :param c: Number of authorities to retrieve

    :return: Top c authorities
    """
    x = eigs_vecs_df.iloc[0].as_matrix()
    y = eigs_vecs_df.iloc[1].as_matrix()
    xy = np.concatenate((x, y))
    conc_index = eigs_vecs_df.index.append(eigs_vecs_df.index)
    cmax_inds = np.argsort(xy)[::-1][:c]
    cmax = conc_index[cmax_inds]
    return cmax


def non_principal_hubs(eigs_vecs_df, c):
    """
    Find c top hubs from set of non principal eigen vectors

    :param eigs_vecs_df: DataFrame, should have 2 columns : the i-th eigen vector of ATA and the i-th eigen vector of AAT
    :param c: Number of authorities to retrieve

    :return: Top c hubs
    """
    x = eigs_vecs_df.iloc[0].as_matrix()
    y = eigs_vecs_df.iloc[1].as_matrix()
    xy = np.concatenate((x, y))
    conc_index = eigs_vecs_df.index.append(eigs_vecs_df.index)
    cmax_inds = np.argsort(xy)[:c]
    cmax = conc_index[cmax_inds]
    return cmax


def get_citations_ranking(graph, nodes=None, drop_zeros=True):
    """
    Return a series of number of citations ranks indexed by the nodes

    :param graph: (networkx.classes.digraph.DiGraph) the graph
    :param nodes: (list) nodes to rank

    :return: pandas Series of ranking indexed by the nodes
    """
    if nodes :
        indegrees = dict(graph.in_degree(nodes))
    else:
        indegrees = dict(graph.in_degree())
    ncits_df = pd.DataFrame.from_dict(data=indegrees, orient="index")
    ncits_df.rename(columns={0: "ncits"}, inplace=True)
    if drop_zeros:
        ncits_df = ncits_df[ncits_df["ncits"] != 0]
    ncits_df.sort_values(inplace=True, by="ncits", ascending=False)
    ncits_df["rank"] = range(0, ncits_df.shape[0])
    ncits_df.sort_index(inplace=True)
    return ncits_df["rank"]


def get_zero_cits_nodes(graph):
    indegrees = dict(graph.in_degree())
    ncits_df = pd.DataFrame.from_dict(data=indegrees, orient="index")
    ncits_df.rename(columns={0: "ncits"}, inplace=True)
    return ncits_df[ncits_df["ncits"] == 0].index


def get_top_autorities(hubs_auths_df, graph=None, drop_zeroscits=False):
    if not drop_zeroscits:
        top_auths = hubs_auths_df.sort_values(by="xauth_0", ascending=False)["xauth_0"]
    else:
        cits_nodes = hubs_auths_df.index.difference(get_zero_cits_nodes(graph))
        top_auths = hubs_auths_df.loc[cits_nodes, :].sort_values(by="xauth_0", ascending=False)["xauth_0"]
    auths_ranks = pd.Series(index=top_auths.index, data=range(0, len(top_auths)))
    auths_ranks.sort_index(inplace=True)
    return auths_ranks


def plot_hubs_authorities(subgraph,
                          auths_rank,
                          hubs_rank,
                          kauths=5,
                          khubs=5,
                          layout=nx.spring_layout,
                          other_authorities=None,
                          other_hubs=None):
    pos = layout(subgraph)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(auths_rank[kauths:]),
                           node_color='C0',
                           node_size=75)
                           #alpha=0.8)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(auths_rank[:kauths]),
                           node_color='C1',
                           node_size=150,
                           label="Top authorities")
                           #alpha=0.8)
    nx.draw_networkx_nodes(subgraph, pos,
                           nodelist=list(hubs_rank[:khubs]),
                           node_color='C2',
                           node_size=150,
                           label="Top hubs")
                           #alpha=0.8)
    if other_authorities:
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=other_authorities,
                               node_color='C3',
                               node_size=150,
                               label="Non principal authorities (2nd largest eigen value)")
    if other_hubs:
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=other_hubs,
                               node_color='C4',
                               node_size=150,
                               label="Non principal hubs (2nd largest eigen value)")
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
    plt.legend()




# *************** LOAD DATA AND CONSTRUCT GRAPH ********************
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
query_list = ["asymmetry", "trading"]
subtest_topic = topic_query_subgraph(cits_refs_graph, d, attrs, query_list)

# compute hubs and authorities in an iterative fashion
hubs_auths_df = iterate_hubs_auths(subtest_topic, k=1000)

# compute authorities in the eigen vector search fashion
hubs_auths_eig = hubs_authorities_eigen(subtest_topic, neigs=1)

# nodes sorted by authority coef
top_auths_topic = hubs_auths_df.sort_values(by="xauth_0", ascending=False).index
print(top_auths_topic)

# nodes sorted by "hubness" coef
top_hubs_topic = hubs_auths_df.sort_values(by="xhubs_0", ascending=False).index
print(top_hubs_topic)

# Draw
plot_hubs_authorities(subtest_topic, top_auths_topic, top_hubs_topic)

# Test of 1st non principal authorities and hubs
# c = 5
#first_eigs = hubs_auths_eig.loc[:, ["xauth_1", "xhub_1"]]
#top_auths1 = non_principal_authorities(first_eigs, c)
#top_hubs1 = non_principal_hubs(first_eigs, c)
#first_eigs.iloc[top_auths1, :]
# plot_hubs_authorities(subtest_topic, top_auths_topic, top_hubs_topic)#, other_authorities=list(top_auths1))




# ***************TEST ON SIMILARITY QUERY*******************************

# Create the expanded subgraph on which to perform the algo
d = 300
pages = []
subtest_similarity = similarity_query_subgraph(pages, cits_refs_graph, d)

# Compute hubs and authorities using eigenvector approach
hubs_auths_sim = hubs_authorities_eigen(subtest_similarity, neigs=1)

# Top auths of principal vector
top_auths_sim = hubs_auths_sim.sort_values(by="xauth_0", ascending=False).index

# Top hubs of principal vector
top_hubs_sim = hubs_auths_sim.sort_values(by="xhub_0", ascending=False).index

# Draw
plot_hubs_authorities(subtest_similarity, top_auths_sim, top_hubs_sim)




# ***************TEST ON WHOLE GRAPH*********************************

start = time.clock()
hubs_auths_whole = hubs_authorities_eigen(cits_refs_graph, neigs=5)
end = time.clock()
print(end - start)

# nodes sorted by authority coef
top_auths_whole = hubs_auths_whole.sort_values(by="xauth_0", ascending=False).index
print(top_auths_whole)

# nodes sorted by "hubness" coef
top_hubs_whole = hubs_auths_whole.sort_values(by="xhub_0", ascending=False).index
print(top_hubs_whole)

# Correlation plot between ncitations and authority score
cits_ranks = get_citations_ranking(cits_refs_graph, drop_zeros=True)
auths_ranks = get_top_autorities(hubs_auths_whole, cits_refs_graph, drop_zeroscits=True)
df = pd.DataFrame(columns=["cits_rank", "authority_rank"])
df["cits_rank"] = cits_ranks
df["authority_rank"] = auths_ranks
sns.jointplot("cits_rank", "authority_rank", df, kind="kde")
# xmin, xmax = plt.xlim()
# # plt.xlim(xmax, xmin)
# ymin, ymax = plt.ylim()
# plt.ylim(ymax, ymin)
