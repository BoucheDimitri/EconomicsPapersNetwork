# Data management
import pandas as pd
import numpy as np

# Network
import networkx as nx

# Data
import itertools
from collections import Counter

# Graphes
import matplotlib.pyplot as plt

# Utilitaires
import os


def str_to_list(x):
    """
    Interpret strings of the form "['int1', 'int2']" as the list [int1, int2]

    Params:
        x (str) : the string to interpret
    Returns:
        list : the output list
    """
    x = x.replace("[", "")
    x = x.replace("]", "")
    splitted = x.split(", ")
    no_list = [int(i) for i in splitted]
    return no_list


def get_edges_list(auths_nums):
    """
    Get list of edges between authors from a series of list of authors
    (each entry of the series are the authors of a given article).

    Params:
        auths_nums (pandas.core.series.Series) : the series of list of authors

    Returns:
        list : a list of tuples with possibly redundant tuples

    WARNING: Does not retain articles with only one author
    """
    edges_list = []
    auths_nums_reduced = auths_nums[auths_nums.apply(lambda x: len(x)) > 1]
    for auth_list in auths_nums_reduced:
        combinations = list(itertools.combinations(auth_list, 2))
        edges_list += combinations
    return edges_list


def sort_edges(edges_list):
    """
    For all tuples in edges_list, put the lowest number in the first place and then sort the whole list.

    Params:
        edges_list (list) : list of tuples representing the authors pairs

    Returns:
        list : a list of tuples after the two sorting steps described above

    """
    sorted_edges_list = []
    for edge in edges_list:
        edge_l = list(edge)
        edge_l.sort()
        sorted_edges_list.append(tuple(edge_l))
    sorted_edges_list.sort()
    return sorted_edges_list


def weighted_edges_list(sorted_edges_list):
    """
    Convert list of authors pairs to a dict of dict to pass to the networkx.Graph constructor.
    Count the duplicates and store them in the "weights" attributes of the dictionnary so that
    an edge that appears k times in sorted_edges_list will get a weight of k in the graph.

    Params:
        list : sorted list of tuples

    Returns:
        dict : a dict of dict, for a given tuple (author1, author2), entry of the form :
        {author1 : {author2: {'weight': n_collaborations(author1, author2)}}}
    """
    counter_dict = dict(Counter(sorted_edges_list)) # {(auth1, auth2): #co_auth, ...}
    nx_dict = dict()
    for key in counter_dict.keys(): # reformat / keys= [(auth1, auth2), ...]
        nx_dict[key[0]] = {key[1]: {'weight': counter_dict[key]}} # format for nx
    return nx_dict


def get_nodes_list(auths_nums):
    """
    Get the different authors from the series of list of authors.
    WARNING: we add non connected authors to the graph afterwards.

    Params:
        auths_nums (pandas.core.series.Series) : the series of list of authors.

    Returns:
        list : the list of authors.
    """
    concat = []
    for auth_list in auths_nums:
        concat +=  auth_list
    return list(set(concat))


def count_CoA(auths_nums):
    """
    Get the different authors from the series of list of authors.
    Nb: we take non connected authors into account.

    Params:
        auths_nums (pandas.core.series.Series) : the series of list of authors.
    Returns:
        Counter : author_id : nbr of authored articles
    """
    concat = []
    for auth_list in auths_nums:
        concat +=  auth_list
    concat.sort()
    return Counter(concat)


# To look for a given author
def search_auth(auth_name):
    resp_auth=auth_coresp.index[auth_coresp.uniformat.str.contains(auth_name)]
    for i in resp_auth:
        print(auth_coresp.uniformat[i],C_CoA[i])


# Path to the data
path = "C://Users//Dimitri//Desktop//ENSAE3A//EconomicsPapersNetwork//Tables//"
# path= os.path.join(os.getcwd(),"Tables")

# Load the data
attrs_nos = pd.read_csv(path + "/attrs_nos.csv", encoding = "ISO-8859-1")

# Print the data
attrs_nos.head()

print("Elements of the {0} authors_nos are {1} with '{2}' as 0th element". \
      format(type(attrs_nos.authors_nos), type(attrs_nos.authors_nos[0]), attrs_nos.authors_nos[0][0]))

# Interpret "author_nos" column as a list of numbers of authors
attrs_nos["authors_nos"] = attrs_nos["authors_nos"].apply(str_to_list)

attrs_nos.authors_nos.head()

print("Elements of the {0} authors_nos are now a {1} of {2}". \
      format(type(attrs_nos.authors_nos), type(attrs_nos.authors_nos[0]), type(attrs_nos.authors_nos[0][0])))

# Get list of edges
auths_nos = attrs_nos["authors_nos"].copy()
edges_list = get_edges_list(auths_nos)
# Print it
print(edges_list[:100])

# Sort list of edges
s_edges_list = sort_edges(edges_list)
# Print the result
print(s_edges_list[:100])

nx_dict = weighted_edges_list(s_edges_list)
print({k:v for k,v in nx_dict.items() if k<100})

# Get nodes list
nodes_list = get_nodes_list(auths_nos)

# Create graphs only from edges
authors_graph = nx.Graph(nx_dict)
# Add the nodes that have no edges
authors_graph.add_nodes_from(nodes_list)
# Comment CV: these so-called" dangling nodes will be excluded from PageRank

# Extract the adjacency matrix as a scipy sparse matrix (won't fit into the RAM as numpy matrix)
adjacency_matrix = nx.to_scipy_sparse_matrix(authors_graph)
# Print it
print(adjacency_matrix[:10])


# STATS DESC ON AUTHORS
C_CoA=count_CoA(auths_nos)

# Deciles
for i in np.linspace(10,100,10):
    print(str(i),":", np.percentile(list(C_CoA.values()),i))

# Percentiles (90-100)
for i in np.linspace(90,100,11):
    print(str(i),":", np.percentile(list(C_CoA.values()),i))

# Hist of coauthorship
plt.hist(list(C_CoA.values()), bins=max(C_CoA.values()))
plt.xlabel("Number of co-authored articles")
plt.ylabel("Authors")

# Top coauthors
auth_coresp=pd.read_csv(path + "/authors.csv", encoding = "ISO-8859-1")
auth_coresp.head()

# List authors who authored at least "tsh" articles
tsh=60
top_CoA=list({k: v for k, v in C_CoA.items() if v > tsh}.keys())
auth_coresp.loc[top_CoA]

search_auth("Stiglitz")