#!python
# -*-coding:utf-8 -*

"""This module provides tools for name disambiguation network graph"""

import pandas as pd
import jellyfish


def authors_parser(authors_string, sep=";"):
    """
    This function gets a str ("auth1;auth2;...") and returns a list
    of authors ["auth1", "auth2", ...]

    :param authors_string:
    :param sep: separator operator (default: ";")
    :return:
    """
    authors = authors_string.split(sep)
    authors = list(filter(None, authors))
    for i in range(0, len(authors)):
        if authors[i][0] == " ":
            authors[i] = authors[i][1:]
    return authors


def normalized_edit_distance(str1, str2):
    """
    This function returns the normalized Levenshtein distance
    between 2 strings

    :param str1: string
    :param str2: string
    :return: distLev(str1,str2)/max(len(str1),str(2))
    """
    dist = jellyfish.levenshtein_distance(str1, str2)
    try:
        return dist / max(len(str1), len(str2))
    except ZeroDivisionError:
        return 0


def uniformize_names(str1):
    """
    This function turns a name string "Name, Surname" into
    "Surname Name"

    :param str1:
    :return:
    """
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
    Function to determine (approximately) whether two authors are likely to be the same

    :param potential_match: (str) potential match to compare author to
    :param author: (str) author to match
    :param thresh: (float) between 0 and 1, the normalized Levensthein distance tolerance
    :return: (bool) True if is likely to be a match and False otherwise
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
        # Indicator for the case where the first string in the name is just one character
        len_test = (len(author_split[0]) == 1) or (
            len(potential_split[0]) == 1)
        # Indicator for the case where the two have the same lenghts and more than two elements each,
        # in that case we can (we do) test on the first character of the second string ("middle name") as well
        same_len_test = (len(author_split) >= 3) and (
            len(author_split) == len(potential_split))
        # Levensthein distance between the first strings (the first name hopefully)
        dist2 = normalized_edit_distance(
            author_split[0], potential_split[0])
        # Middle test is active only when same_len_test is True and the length of both names is > 1
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
    """
    This functions looks for authors with "close enough" names where similarity is defined
    by the author_comparison function with treshold 'tresh'. It returns a dataframe with
    an additional var containing a list of "equivalent" names.

    :param auths_df: (pandas.core.frame.DataFrame) with "original" and "uniformat" columns
    (see unformize_names)
    :param thresh: (float) in (0;1) (see autho_comparison)
    :return:
    """
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


def str_to_list(x):
    """
    Interpret strings of the form "['auth1', 'auth2']" as the list ['auth1', 'auth2']

    :param x: (str) string to interpret
    :return y: (list) the output list
    """
    x = x.replace("[", "")
    x = x.replace("]", "")
    splitted = x.split("', ")
    y = [z.replace("'", "") for z in splitted]
    return y


def unpack_authors_list(authors_df):
    """

    :param authors_df:
    :return:
    """
    n_eqs = authors_df["equivalent"].apply(lambda x: len(x))
    max_eqs = n_eqs.max()
    for i in range(0, max_eqs):
        authors_df["equivalent_" + str(i)] = ""
    for i in authors_df.index:
        for j in range(0, n_eqs[i]):
            authors_df.set_value(i, "equivalent_" + str(j), authors_df.loc[i, "equivalent"][j])
    return authors_df


def author_corresp(authors_df, eqs_cols, author_list):
    """

    :param authors_df:
    :param eqs_cols:
    :param author_list:
    :return:
    """
    n_eqs = len(eqs_cols)
    authors_nos = []
    for author in author_list:
        found = False
        i = 0
        while (i < n_eqs) and (not found):
            search = authors_df[eqs_cols[i]].str.contains(author, regex=False)
            ind_search = search[search].index
            if len(ind_search) >= 1:
                found = True
                authors_nos.append(ind_search[0])
            i += 1
    return authors_nos
