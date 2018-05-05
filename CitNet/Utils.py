#!python
# -*-coding:utf-8 -*

"""This module provides various tools for processing"""

def str_to_list(x):
    """
    Turns strings of the form "['int1', 'int2']" into
    a list of integers [int1, int2]

    :param x:
        x (str) : the string to interpret
    :return:
        list : the output list
    """
    x = x.replace("[", "")
    x = x.replace("]", "")
    splitted = x.split(", ")
    if splitted[0] == "":
        no_list = []
    else:
        no_list = [int(i) for i in splitted]
    return no_list