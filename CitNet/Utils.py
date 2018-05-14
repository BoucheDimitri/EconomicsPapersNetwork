#!python
# -*-coding:utf-8 -*

"""This module provides various tools for processing"""

def str_to_list(x):
    """
    Turns strings of the form "['int1', 'int2']" into
    a list of integers [int1, int2]

    :param x:(str) the string to be interpreted
    :return: (list) the output list
    """
    x = x.replace("[", "")
    x = x.replace("]", "")
    splitted = x.split(", ")
    if splitted[0] == "":
        no_list = []
    else:
        no_list = [int(i) for i in splitted]
    return no_list


def parse_url(url, to_remove):
    """
    Removes a given sequence from a string

    :param url: (str) the original url
    :param to_remove: (str) the string to remove
    :return: (str) url with to_remove removed
    """
    return url.replace(to_remove, "")
