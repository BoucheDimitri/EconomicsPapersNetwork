import pandas as pd
import numpy as np
import copy
import itertools
# Package for Levenstein string distance
import jellyfish
import time


def authors_parser(authors_string, sep=";"):
    authors = authors_string.split("; ")
    for i in range(0, len(authors)):
        if authors[i][0] == " ":
            authors[i] = authors[i][1:]
    return authors


def normalized_edit_distance(str1, str2):
    dist = jellyfish.levenshtein_distance(str1, str2)
    try:
        return dist / max(len(str1), len(str2))
    except ZeroDivisionError:
        return 0


def uniformize_names(str1):
    if "," in str1:
        splitted = str1.split(", ")
        if len(splitted) > 1:
            str1bis = splitted[1] + " " + splitted[0]
        else:
            str1bis = str1
        return str1bis
    else:
        return str1


def map_authors(auths_df, thresh):
    cleaned_df = pd.DataFrame(columns=["original", "uniformat", "equivalent"])
    for i in auths_df.index:
        author = auths_df.loc[i, "uniformat"]
        author_original = auths_df.loc[i, "original"]
        splitted = author.split(" ")
        ind_search = cleaned_df[cleaned_df["uniformat"].str.contains(
            splitted[-1], regex=False)].index
        exists_similar = False
        for ind in ind_search:
            unisplit = cleaned_df["uniformat"][ind].split(" ")
            dist_last = normalized_edit_distance(unisplit[-1],
                                                 splitted[-1])
            first_test = dist_last <= thresh
            if first_test:
                dist = normalized_edit_distance(cleaned_df["uniformat"][ind],
                                                author)
                dist_test = dist <= thresh
                if dist_test:
                    exists_similar = True
                    cleaned_df["equivalent"][ind].append(author_original)
                else:
                    len_test = (len(splitted[0]) == 1) or (
                        len(unisplit[0]) == 1)
                    if not len_test:
                        dist2 = normalized_edit_distance(
                            unisplit[0], splitted[0])
                        if dist2 <= thresh:
                            exists_similar = True
                            cleaned_df["equivalent"][ind].append(
                                author_original)
                        elif ((splitted[0][1] == ".") or (unisplit[0][1] == ".")) and (splitted[0][0] == unisplit[0][0]):
                            exists_similar = True
                            cleaned_df["equivalent"][ind].append(
                                author_original)
                    else:
                        if splitted[0][0] == unisplit[0][0]:
                            exists_similar = True
                            cleaned_df["equivalent"][ind].append(
                                author_original)
        if not exists_similar:
            df = pd.DataFrame(
                columns=[
                    "original",
                    "uniformat",
                    "equivalent"],
                index=[0])
            df.set_value(0, "original", author_original)
            df.set_value(0, "uniformat", author)
            df.set_value(0, "equivalent", [author_original])
            cleaned_df = cleaned_df.append(df, ignore_index=True)
        if i % 1000 == 0:
            print(i)
    return cleaned_df


def unpack_authors_list(authors_df):
    n_eqs = authors_df["equivalent"].apply(lambda x: len(x))
    max_eqs = n_eqs.max()
    for i in range(0, max_eqs):
        authors_df["equivalent_" + str(i)] = np.nan
    for i in authors_df.index:
        for j in range(0, n_eqs[i]):
            authors_df.set_value(i, "equivalent_" +
                                 str(j), authors_df.loc[i, "equivalent"][j])
    return authors_df


def author_corresp(authors_df, eqs_cols, author_list):
    n_eqs = len(eqs_cols)
    authors_nos = []
    for author in author_list:
        found = False
        i = 0
        while (i <= n_eqs) and (not found):
            search = authors_df[eqs_cols[i]].str.contains(author, regex=False)
            ind_search = search[search].index
            if len(ind_search) >= 1:
                found = True
                authors_nos.append(ind_search[0])
            i += 1
    return authors_nos


def search_auth(def_auths, auth_name):
    query_auth = def_auths[def_auths.uniformat.str.contains(auth_name)]
    return query_auth


# Path to the data
path = "C://Users//Dimitri//Desktop//ENSAE3A//EconomicsPapersNetwork//Data//"

# Load attributes data
attrs = pd.read_csv(path + "attrs_sub.csv")

# Parse authors from string to list (inplace)
attrs["authors_list"] = attrs["authors"].apply(authors_parser)

# Stack all authors in a list
authors = []
for author in attrs["authors_list"]:
    authors += author
# Remove duplicates
authors = list(set(authors))

# Create a dataframe for authors
df_authors = pd.DataFrame(authors, columns=["original"])

# Create a column with authors in a uniform format
df_authors["uniformat"] = df_authors["original"].apply(uniformize_names)

# Reduced version of df_authors for faster testing
df_authors_reduced = df_authors.iloc[10000: 12000, :].copy()

# Start mapping authors (finding equivalent ones)
start = time.clock()
cleaned = map_authors(df_authors, 0.12)
# cleaned = map_authors(df_authors_reduced, 0.12)
end = time.clock()
print(end - start)

# Filter out the cases where multiple authors points to one
print(cleaned[cleaned.equivalent.apply(lambda x: len(x)) > 1])

# Sort the resulting modified authors dataframe by "uniformat" and reset
# its index
cleaned = cleaned.sort_values(by="uniformat")
cleaned = cleaned.reset_index(drop=True)

# Save the dataframe as csv
cleaned.to_csv(path + "authors.csv", index=False)

# Add columns for 1st authors, 2nd authors, and so on, the irrelevant ones
# for a paper are set to np.nan
eqs_cols = ["equivalent_" + str(i) for i in range(0, 7)]
cleaned = unpack_authors_list(cleaned)

# Find authors indexes for each paper in attrs
start = time.clock()
attrs["authors_nos"] = attrs["authors_list"].apply(
    lambda x: author_corresp(cleaned, eqs_cols, x))
end = time.clock()
print(end - start)

# Save to csv
attrs.to_csv(path + "attrs_nos.csv")
