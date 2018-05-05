from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError

import pandas as pd
import numpy as np

from progressbar import ProgressBar

from CitNet import ScrapIR

# 1. Get the set of url adresses of interest

# 1.0 Set root
root = "https://ideas.repec.org/a/"

# 1.1 Get editor list
ed_list = []
pbar = ProgressBar()

html = urlopen(root)
bsObj = BeautifulSoup(html, "lxml")

for ed in pbar(bsObj.findAll({"a": "href"})):
    if len(ed.attrs["href"]) == 4:
        ed_list += [ed.attrs["href"]]


#  1.2 Get journal list
journ_list = []
edjourn_list = []
pbar = ProgressBar()

for ed in pbar(ed_list):
    html = urlopen(root + ed)
    bsObj = BeautifulSoup(html, "lxml")

    for journ in bsObj.findAll({"a": "href"}):
        if len(journ.attrs["href"]) == 7:
            journ_list += [journ.attrs["href"]]
            edjourn_list += [ed + journ.attrs["href"]]

#  1.3 Get article list
art_list = []
edjournart_list = []
pbar = ProgressBar()
i = 0

for edj in pbar(edjourn_list):
    try:
        html = urlopen(root + edj)
    except HTTPError as e:
        i += 1

    bsObj = BeautifulSoup(html, "lxml")
    for art in bsObj.findAll({"a": "href"}):
        if ".html" in art.attrs["href"]:
            art_list += [art.attrs["href"]]
            edjournart_list += [edj + art.attrs["href"]]

# ed_list ["ed1/","ed2/", ...]
# journ_list ["journ1/","journ2/", ...]
# art_list ["art1.htm","art2.html", ...]
# edjourn_list ["ed1/journ1/","ed1/journ2", ...]
# edjournart_list ["ed1/journ1/art1.html", ...]

# 1.4 Save eja list (and del the others)
pd.Series(edjournart_list).to_csv('Tables/edjournart_list.csv', index=False, header=False)
del ed_list, journ_list, art_list, edjourn_list
del edjournart_list

# 2. Most renowned journals
# 2.0 Get Rank_j


def get_rankj(lb, url="https://ideas.repec.org/top/top.journals.all.html"):
    """
    This function returns the lb_th most important journals.

    :param lb: rank lower bound in the subset
    :param url: url of the ranking page
    :return: np.ar ["ed/journ", ...]
    """
    html = urlopen(url)
    bsObj = BeautifulSoup(html, "lxml")
    rankj_list = []
    i = 0

    while len(rankj_list) <= lb:
        i += 1
        try:
            rankj_list += [bsObj.find("div", {"aria-labelledby": "ranking-tab"}).findAll("a")[i].attrs["name"]]
            # print(i)
        except (AttributeError, KeyError):
            pass

    return pd.Series(rankj_list).apply(lambda x: x.split(":")[1] + "/" + x.split(":")[2]).values

# 2.1 Top journals (np.ar)
TopJourn_ar = get_rankj(30)

# 2.2 Subset of articles in top journals
edjournart_db = pd.read_csv('Tables/edjournart_list.csv', header=None, names=["eja"])
# !rm "Tables/edjournart_list.csv"

TopEja = np.empty(1).flatten()
for i in range(len(TopJourn_ar)):
    TopEja = np.concatenate([TopEja, np.array(np.where(edjournart_db.eja.str.contains(TopJourn_ar[i]) == True)).flatten()])
TopEja = TopEja[1:].astype(int)  # Indexes (!)

pd.Series(edjournart_db.values[TopEja].flatten()).to_csv("Tables/eja.csv", index=False, header=False)

# 3. Parse Articles
# 3.0 Load list of eja (np.array)
eja_ar = pd.read_csv('Tables/eja.csv', header=None).values.flatten()
# !rm "Tables/eja.csv"

# 3.1  Get attributes

attrs = []
for eja in eja_ar[:100]:  # Should be ran on full eja
    attrs += [ScrapIR.get_attrs(eja)]

db_attrs = pd.DataFrame(attrs, columns=["url", "title", "authors", "date", "jel_code", "keywords"])
db_attrs["editor"] = db_attrs.url.str.split("/").apply(lambda x: x[4])
db_attrs["journal"] = db_attrs.url.str.split("/").apply(lambda x: x[5])
db_attrs["article_id"] = db_attrs.url.str.split("/").apply(lambda x: x[-1])
# db_attrs["year"] = pd.DatetimeIndex(db_attrs.date).year
#db_attrs.to_csv("Tables/attrs")

# 3.2 Get refs and cits

cit_ar = ScrapIR.get_stack(eja_ar[:100], "cit")
ref_ar = ScrapIR.get_stack(eja_ar[:100], "ref")
# Should be ran on full eja

# pd.DataFrame(cit_ar).to_csv("Tables/cits", index=False, header=False)
# pd.DataFrame(ref_ar).to_csv("Tables/refs", index=False, header=False)


# At the end of the day, the 3 output of interest are :
# attrs
# refs
# cits
