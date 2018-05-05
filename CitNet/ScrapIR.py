#!python
# -*-coding:utf-8 -*



"""This module provides tools to scrap articles from Ideas Repec"""


from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError

import pandas as pd
import numpy as np

from progressbar import ProgressBar


def get_refs(eja, root="https://ideas.repec.org/a/"):
    """
    This function returns the references from the 
    specified article (format eja: editor/journal/article)
    
    :param eja:  format editor/journal/article
    :param root: default "https://ideas.repec.org/a/"
    :return: [[id_art, id_ref]...] (np.array)
    """
    
    url= root + eja
    try:
        html=urlopen(url)
    except HTTPError as e:
        return None
    
    bsObj=BeautifulSoup(html,"lxml")
    try:
        ref=bsObj.find("div", {"aria-labelledby":"refs-tab" })\
        .find("input").attrs["value"].split("#")
    except AttributeError as e:
        return None
    
    ref=pd.Series(ref).apply(lambda x:\
                           [eja,x.split(":")[1] + "/" + \
                           x.split(":")[2] + "/" + \
                           ''.join(x.split(":")[3:]) + ".html"]).values
    return ref


def get_cits(eja, root="https://ideas.repec.org/a/"):
    """
    This function returns the citations pointing to the 
    specified article (format eja: editor/journal/article)

    :param eja:  format editor/journal/article
    :param root: default "https://ideas.repec.org/a/"
    :return: [[id_art, id_cit]...] (np.array)
    """
    url= root + eja
    try:
        html=urlopen(url)
    except HTTPError as e:
        return None
    
    bsObj=BeautifulSoup(html,"lxml")
    try:
        ref=bsObj.find("div", {"aria-labelledby":"cites-tab" })\
        .find("input").attrs["value"].split("#")
    except AttributeError as e:
        return None
    
    ref=pd.Series(ref).apply(lambda x:\
                           [eja ,x.split(":")[1] + "/" + \
                           x.split(":")[2] + "/" + \
                           ''.join(x.split(":")[3:]) + ".html"]).values
    return ref


def get_stack(itr_list, meth):
    """
    This function piles-up the refences/citations from/pointing
    to the sequence of specified articles (format eja: 
    editor/journal/article)

    :param itr_list:  list of eja (potentially eja[i:j])
    :param meth:"ref" for references
                "cit" for citations
    :return:[id_art1, id_cit1]
            [id_art1, id_cit2]
            ...
            [id_artn, id_citp]] (np.array)
    """
    
    valid={"cit","ref"}
    if meth not in valid:
        raise ValueError("results: meth must be one of %r." % valid)
        
    if meth=="cit":
        cits=np.empty(1)
        pbar = ProgressBar()
        for eja in pbar(itr_list):
        #for eja in itr_list[i:j]:
            if get_cits(eja) is not None:
                    cits=np.concatenate((cits,get_cits(eja)), axis=0)
        return cits[1:]
    
    if meth=="ref":
        refs=np.empty(1)
        pbar = ProgressBar()
        for eja in pbar(itr_list):
        #for eja in itr_list[i:j]:
            if get_refs(eja) is not None:
                    try :
                        refs_copy = np.copy(refs)
                        refs=np.concatenate((refs,get_refs(eja)), axis=0)
                    except ValueError:
                        refs = refs_copy
        return refs[1:]


def get_attrs(eja, root="https://ideas.repec.org/a/"):
    """
    This function returns the attributes of interest from the 
    specified article (format editor/journal/article). 
    
    :param eja: format editor/journal/article
    :param root: default "https://ideas.repec.org/a/"
    :return: url, title, authors, date, jel_code, keywords
    """
    

    url= root + eja
    try:
        html=urlopen(url)
    except HTTPError as e:
        return None
    
    bsObj=BeautifulSoup(html,"lxml")
    try:
        title=bsObj.find("meta", {"name":"citation_title"}).attrs["content"]
    except AttributeError as e:
        title=np.nan
    try:
        authors=bsObj.find("meta", {"name":"citation_authors"}).attrs["content"]
    except AttributeError as e:
        authors=np.nan
    try:
        date=bsObj.find("meta", {"name":"date"}).attrs["content"]
    except AttributeError as e:
        date=np.nan
    try:
        jel_code=bsObj.find("meta", {"name":"jel_code"}).attrs["content"]
    except AttributeError as e:
        jel_code=np.nan
    try:
        keywords=bsObj.find("meta", {"name":"keywords"}).attrs["content"]
    except AttributeError as e:
        keywords=np.nan
    
    return url, title, authors, date, jel_code, keywords    