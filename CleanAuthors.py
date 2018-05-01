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


def author_comparison(potential_match, author, thresh=0.1):
    """
    Function to determine (approximately) if two authors are likely to be the same

    Params:
        potential_match (str) : potential match to compare author to
        author (str) : author to match
        thresh (float) : between 0 and 1, the normalized Levensthein distance tolerance
    Returns:
        bool : True if is likely to be a match and False otherwise
    """
    potential_split = potential_match.split(" ")
    author_split = author.split(" ")
    dist_last = normalized_edit_distance(potential_split[-1],
                                         author_split[-1])
    # First, check that last string in the name (hopefully, the last name) is
    # coherent
    first_test = dist_last <= thresh
    similar = False
    if first_test:
        # Indicator for the case where the first string in the name is just one
        # character
        len_test = (len(author_split[0]) == 1) or (
            len(potential_split[0]) == 1)
        # Indicator for the case where the two have the same lenghts and more than two elements each,
        # in that case we can (we do) test on the first character of the second
        # string (the "middle name") as well
        same_len_test = (len(author_split) >= 3) and (
            len(author_split) == len(potential_split))
        # Levensthein distance between the first strings (the first name
        # hopefully)
        dist2 = normalized_edit_distance(
            author_split[0], potential_split[0])
        # Middle test is active only when same_len_test is True and the length
        # of both names is > 1
        if (len(author_split) > 1) and (
                len(potential_split) > 1) and same_len_test:
            middle_test = author_split[1][0] == potential_split[1][0]
        else:
            middle_test = True
        # Case where the first string is just on character
        if len_test:
            similar = (potential_split[0][0] ==
                       author_split[0][0]) and middle_test
        # General test on first using Levensthein distance
        elif dist2 <= thresh and middle_test:
            similar = True
        # Case where the first string is just one character followed by a "."
        elif ((potential_split[0][1] == ".") or (author_split[0][1] == ".")) and (
                potential_split[0][0] == author_split[0][0]) and middle_test:
            similar = True
    return similar


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
            potential_match = cleaned_df["uniformat"][ind]
            similar = author_comparison(potential_match, author, thresh)
            if similar:
                cleaned_df["equivalent"][ind].append(
                    author_original)
                exists_similar = True
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
        authors_df["equivalent_" + str(i)] = ""
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
    for auth_list in query_auth.equivalent :
        print(auth_list)


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

cleaned_cop = cleaned.copy()
# eqs_cols = ["equivalent_" + str(i) for i in range(0, max_equivalence)]
cleaned_cop = unpack_authors_list(cleaned_cop)

# Find authors indexes for each paper in attrs (WARNING : TAKES A LITTLE MORE THAN AN HOUR)
maxs_eq = max(cleaned_cop.equivalent.apply(lambda x: len(x)))
eqs_cols = ["equivalent_" + str(i) for i in range(0, maxs_eq)]
start = time.clock()
attrs["authors_nos"] = attrs["authors_list"].apply(
    lambda x: author_corresp(cleaned_cop, eqs_cols, x))
end = time.clock()
print(end - start)

# Save to csv
attrs.to_csv(path + "attrs_nos.csv")
