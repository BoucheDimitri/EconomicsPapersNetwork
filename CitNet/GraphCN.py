#!python
# -*-coding:utf-8 -*

"""This module provides tools for building network graph"""

import itertools
from collections import Counter
import numpy as np


def get_edges_list(auths_nums):
    """
    Get list of edges between authors from a series of list of authors
    (each entry of the series are the authors of a given article).

    :param :
        auths_nums (pandas.core.series.Series) : the series of list of authors
    :return :
        list : a list of tuples with possibly redundant tuples

    WARNING: Filters out articles with only one author
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

    :param :
        edges_list (list) : list of tuples representing the authors pairs
    :return :
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
    Count the duplicates and store them in the "weights" attributes of the dictionary so that
    an edge that appears k times in sorted_edges_list will get a weight of k in the graph.

    :param :
        list : sorted list of tuples

    :return :
        dict : a dict of dict, for a given tuple (author1, author2), entry of the form :
        {author1 : {author2: {'weight': n_collaborations(author1, author2)}}}
    """
    counter_dict = dict(Counter(sorted_edges_list))  # {(auth1, auth2): #co_auth, ...}
    nx_dict = dict()
    for key in counter_dict.keys():  # reformat / keys= [(auth1, auth2), ...]
        nx_dict[key[0]] = {key[1]: {'weight': counter_dict[key]}}  # format for nx
    return nx_dict


def get_nodes_list(auths_nums):
    """
    Get the different authors from the series of list of authors.
    WARNING: we add non connected authors to the graph afterwards.

    :param :
        auths_nums (pandas.core.series.Series) : the series of list of authors.

    :return :
        list : the list of authors.
    """
    concat = []
    for auth_list in auths_nums:
        concat += auth_list
    return list(set(concat))  # set() avoid duplicates


def article_matching(url_id_series, to_match):
    """
    Match url_ids from url_id_series to each element of to_match
    (concretely : transforms urls into their article_id from attrs)

    :param url_id_series: (pandas.core.series.Series)
     series which index is the article number and which field is its url
    :param to_match: (pandas.core.series.Series)
     series of url to match to articles numbers in url_id_series

    :return : (pandas.core.series.Series) to_match with urls replaced
    by articles numbers from url_id_series
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

    :param refs_df: (pandas.core.frame.DataFrame)
    :param id_series:
    :param begin_with:
    :return:
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
