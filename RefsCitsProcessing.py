# Data management
import pandas as pd
# Utilitaires
import time
import os
# UserDefined module
from CitNet import Utils
from CitNet import GraphCN

#####################################################################
# GRAPH CITATIONS
#
# Input: refs.csv
#        cits.csv
#        attrs.csv
# Output:cits_ids.csv
#        refs_ids.csv
#####################################################################

#####################################################################
# Path to the data
path = os.path.join(os.getcwd(), "Tables")

#####################################################################
#####################################################################
# Load the data
refs = pd.read_csv(path + "refs.csv")
cits = pd.read_csv(path + "cits.csv", header=None)
attrs = pd.read_csv(path + "attrs_nos.csv", encoding="ISO-8859-1")

#####################################################################
#####################################################################
# Section 1. Preprocessing of urls
#####################################################################
#####################################################################

#####################################################################
#####################################################################
# Pre processing of cits to uniformize format with refs.csv
cits["listed"] = cits[0].apply(Utils.str_to_list)
cits["referred_to"] = cits["listed"].apply(lambda x: x[0])
cits["referring"] = cits["listed"].apply(lambda x: x[1])
cits = cits[["referred_to", "referring"]]
# Uniformize the format of the urls
to_remove = "https://ideas.repec.org/a/"
parse_url_ideas = lambda x: Utils.parse_url(x, to_remove)
attrs["url_id"] = attrs["url"].apply(parse_url_ideas)

#####################################################################
#####################################################################
# Section 2. Matching urls index (of attrs)
#####################################################################
#####################################################################

#####################################################################
#####################################################################
# Get the series which index are the articles number and which field are their urls
id_series = attrs["url_id"]
# Match article id for all refs (TAKES APPROX 7 1/2 HOURS !)
start = time.clock()
modified_refs = GraphCN.match_articles(refs, id_series)
modified_refs.to_csv(path + "/refs_id.csv")  # refs_edges.csv ?
end = time.clock()
print(end - start)
# Match article id for all cits (TAKES APPROX 3 1/2 HOURS !)
start = time.clock()
modified_cits = GraphCN.match_articles(cits, id_series, begin_with="referring")
modified_cits.to_csv(path + "/cits_id.csv")  # cits_edges.csv ?
end = time.clock()
print(end - start)

#####################################################################
#####################################################################
# Output:cits_ids.csv
#        refs_ids.csv
#####################################################################
#####################################################################
