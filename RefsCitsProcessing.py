# Data management
import numpy as np
import pandas as pd

# utilitaires
import time
import os

# Graphes
import matplotlib.pyplot as plt


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
    x = x.replace("'", "")
    splitted = x.split(", ")
    no_list = [str(i) for i in splitted]
    return no_list


def parse_url(url, to_remove):
    """
    Remove a given sequence from a string

    Params:
        url (str) : the original url.
        to_remove(str) : the string to remove
    Returns:
        str: url with string removed
    """
    return url.replace(to_remove, "")


def article_matching(url_id_series, to_match):
    """
    Match url_ids from url_id_series to each element of to_match
    (concretely : transforms urls into their article_id from attrs)

    Params:
        url_id_series (pandas.core.series.Series) : series which index is the article number and which field is its url
        to_match (pandas.core.series.Series) : series of url to match to articles numbers in url_id_series

    Returns:
        pandas.core.series.Series : to_match with urls replaced by articles numbers from url_id_series
    """
    to_match_copy = to_match.copy()
    uniques = set(to_match_copy.values)
    for url_id in uniques:
        matchs_inds = url_id_series[url_id_series == url_id].index
        if len(matchs_inds) == 1:
            num = matchs_inds[0]
        else:
            num = np.nan
        to_match_copy[to_match_copy == url_id] = num
    return to_match_copy


def match_articles(refs_df, id_series, begin_with="referred_to"):
    """
    Match all articles in the references (or citations) dataframe.
    Matching is done first for the article to which the reference is since they are not all in our database,
    we thus remove the deadlinks before matching the articles from which the reference originates
    which spares a lot of useless computations.

    Params:
        pandas.core.frame.DataFrame

    """
    if begin_with == "referring":
        col1 = "referring"
        col2 = "referred_to"
    else:
        col1 = "referred_to"
        col2 = "referring"
    refs_df_copy = refs_df.copy()
    refs_df_copy.sort_values(by=col1, inplace=True)
    refs_df_copy[col1] = article_matching(id_series, refs_df_copy[col1])
    refs_df_copy.dropna(axis=0, how="any", inplace=True)
    refs_df_copy.sort_values(by=col2, inplace=True)
    refs_df_copy[col2] = article_matching(id_series, refs_df_copy[col2])
    return refs_df_copy





#path = "~//Bureau//ENSAE//NetworkData//CitationsNetwork//"
path= os.getcwd()

refs = pd.read_csv(path + "refs.csv")
cits = pd.read_csv(path + "cits.csv", header=None)
attrs = pd.read_csv(path + "attrs_nos.csv", encoding = "ISO-8859-1")

# Pre processing of cits to uniformize format with refs.csv
cits["listed"] = cits[0].apply(str_to_list)
cits["referred_to"] = cits["listed"].apply(lambda x: x[0])
cits["referring"] = cits["listed"].apply(lambda x: x[1])
cits = cits[["referred_to", "referring"]]

# Uniformize the format of the urls
to_remove = "https://ideas.repec.org/a/"
parse_url_ideas = lambda x: parse_url(x, to_remove)
attrs["url_id"] = attrs["url"].apply(parse_url_ideas)

# Get the series which index are the articles number and which field are their urls
id_series = attrs["url_id"]

# Match article id for all refs (TAKES APPROX 7 1/2 HOURS !)
start = time.clock()
modified_refs = match_articles(refs, id_series)
modified_refs.to_csv(path + "refs_id.csv")
end = time.clock()
print(end - start)

# Match article id for all cits (TAKES APPROX 3 1/2 HOURS !)
start = time.clock()
modified_cits = match_articles(cits, id_series, begin_with="referring")
modified_cits.to_csv(path + "cits_id.csv")
end = time.clock()
print(end - start)
