import pandas as pd
import time
from CitNet import DisambName as DN
import os

#####################################################################
# NAME DISAMBIGUATION
#
# Input: attrs.csv
# Output: authors.csv - dataframe of disambiguated names based on
#                       names in attrs.authors
# Output: attrs_nos.csv - attrs.csv + author_nos corresponding to
#                         disambiguated names in authors.csv
#####################################################################


#####################################################################
# Path to the data
path = os.path.join(os.getcwd(), "Tables")


#####################################################################
#####################################################################
# Section 1
# Input: attrs.csv
#####################################################################
#####################################################################
# Load attributes data
attrs = pd.read_csv(path + "/attrs.csv")


#####################################################################
#####################################################################
# Parse authors from string to list (inplace)
attrs["authors_list"] = attrs["authors"].apply(DN.authors_parser)
# Stack all authors in a list (wo duplicates)
authors = []
for author in attrs["authors_list"]:
    authors += author
authors = list(set(authors))


#####################################################################
#####################################################################
# Create a dataframe for authors
df_authors = pd.DataFrame(authors, columns=["original"])
# Create a column with authors in a uniform format
df_authors["uniformat"] = df_authors["original"].apply(DN.uniformize_names)


#####################################################################
#####################################################################
# Reduced version of df_authors for faster testing
# df_authors_reduced = df_authors.iloc[10000: 12000, :].copy()
# cleaned = map_authors(df_authors_reduced, 0.12)


#####################################################################
#####################################################################
# Start mapping authors (finding equivalent ones)
start = time.clock()
cleaned = DN.map_authors(df_authors, 0.12)
end = time.clock()
print(end - start)
# Filter out the cases where multiple authors points to one
print(cleaned[cleaned.equivalent.apply(lambda x: len(x)) > 1])
print("len was {0}, it is now {1}".format(len(df_authors), len(cleaned)))

#####################################################################
#####################################################################
# Sort the resulting modified authors dataframe by "uniformat" and reset
# its index
cleaned = cleaned.sort_values(by="uniformat")
cleaned = cleaned.reset_index(drop=True)
# Save to csv
cleaned.to_csv(path + "/authors.csv", index=False)

#####################################################################
#####################################################################
# Output: authors.csv
#####################################################################
#####################################################################


#####################################################################
#####################################################################
# Section 2
# Input: authors.csv
#####################################################################
#####################################################################
# Reload the data
cleaned_cop = pd.read_csv(path + "/authors.csv", encoding="ISO-8859-1")

#####################################################################
#####################################################################
# Pre-process the data
cleaned_cop["equivalent"] = cleaned_cop["equivalent"].apply(DN.str_to_list)
cleaned_bis = DN.unpack_authors_list(cleaned_cop)
# Find authors indexes for each paper in attrs (WARNING : TAKES A LITTLE MORE THAN AN HOUR)
maxs_eq = max(cleaned_cop.equivalent.apply(lambda x: len(x)))
eqs_cols = ["equivalent_" + str(i) for i in range(0, maxs_eq)]
start = time.clock()
attrs["authors_nos"] = attrs["authors_list"].apply(
    lambda x: DN.author_corresp(cleaned_cop, eqs_cols, x))
end = time.clock()
print(end - start)

#####################################################################
#####################################################################
# Save to csv
attrs.to_csv(path + "/attrs_nos.csv")

#####################################################################
#####################################################################
# Output: attrs_nos.csv
#####################################################################
#####################################################################
