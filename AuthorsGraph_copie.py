# Data management
import pandas as pd
# Network
import networkx as nx
# Sparse
import scipy.sparse
# Utils
import os
# UserDefined
from CitNet import GraphCN, Utils


#####################################################################
# GRAPH AUTHORS
#
# Input: attrs_nos.csv
# Output: AdjMat_Auth.npz - Sparse weighted adjacency mat of
#                           co-authorship
#####################################################################

#####################################################################
# Path to the data
path = os.path.join(os.getcwd(), "Tables")


#####################################################################
#####################################################################
# Section 1. Edges
#####################################################################
#####################################################################


#####################################################################
#####################################################################
# Load the data
attrs_nos = pd.read_csv(path + "/attrs_nos.csv",
                        encoding="ISO-8859-1",
                        index_col=0)
# Pre-process : Interpret "author_nos" column as a list of numbers of authors
attrs_nos["authors_nos"] = attrs_nos["authors_nos"].apply(Utils.str_to_list)

#####################################################################
#####################################################################
# Get list of edges
auths_nos = attrs_nos["authors_nos"].copy()
edges_list = GraphCN.get_edges_list(auths_nos)
# Sort list of edges
s_edges_list = GraphCN.sort_edges(edges_list)
# Get weighted edges
nx_dict = GraphCN.weighted_edges_list(s_edges_list)
# Nb: prints might be useful to fully understand what happens
# print(edges_list[:10], "\n",
#  s_edges_list[:10],"\n",
#  {k: v for k,v in nx_dict.items() if k<100})
del edges_list, s_edges_list

#####################################################################
#####################################################################
# Section 2. Nodes
#####################################################################
#####################################################################

#####################################################################
#####################################################################
# Get list of nodes (all authors, no duplicates)
nodes_list = GraphCN.get_nodes_list(auths_nos)
del auths_nos

#####################################################################
#####################################################################
# Section 2. Graph and Adjacency matrix
#####################################################################
#####################################################################

#####################################################################
#####################################################################
# Create graphs
# Only from edges (ie w/o single authors)
authors_graph = nx.Graph(nx_dict)
# Add the nodes that have no edges (ie w single authors)
authors_graph.add_nodes_from(nodes_list)
# NB: these so-called "dangling" nodes will be excluded from PageRank

#####################################################################
#####################################################################
# Get adjacency matrix
# Extract the adjacency matrix as a scipy sparse matrix /!\ Weighted
adjacency_matrix = nx.to_scipy_sparse_matrix(authors_graph)
# print(adjacency_matrix[:10])
# save as .npz
scipy.sparse.save_npz(path + '/AdjMat_Auth.npz', adjacency_matrix)
# scipy.sparse.load_npz(path + '/AdjMat_Auth.npz') # to load


#####################################################################
#####################################################################
# Output : AdjMat_Auth.npz - Sparse weighted adjacency mat of
#                            co-authorship
#####################################################################
#####################################################################
