import pandas as pd
import numpy as np
import importlib
import networkx as nx
import CitNet.GraphCN as GraphCN
import os

importlib.reload(GraphCN)


def topic_query(attrs_df, query_string, search_in=("title", "keywords"), how="union"):
    inds = []
    for col in search_in:
        inds.append(attrs_df[attrs_df[col].str.contains(query_string, regex=False)].index)

def set_root():
    return 0


def expand_root():
    return 0





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



