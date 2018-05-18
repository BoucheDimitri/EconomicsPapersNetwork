# EconomicsPapersNetwork
Authors : Dimitri Bouche & Cyril Verluise

* **Brief presentation:** In this repository, we implement tools to scrap, process, and exploit a relationnal database of Economics articles (containing both contextual informations on the articles and references and citations between articles). The final objective being comparing the performances of different search algorithms that uses graph structure such as HITS (Hubs and Authorities), PageRank and some derivatives of those algorithms*.

## Getting started

Most of our visual results can be seen in the Notebooks 'DescStat.ipynb' and 'Ranking.ipynb'. They both leverage the processed database and the other part of the codes, so they are kind of the finised product.

## Structure

1. Python Notebooks (see get started section)


2. Python Files

      * The folder **CitNet** contains all the functions that we wrote for pre processing of the database and construction of the graph       wrappeed up as a Python module.

      * 'AuthorsGraph.py' uses tools from our **CitNet** module as well as processed data tablesto construct the graph of authors             collaboration.

      * 'CleanAuthors.py' uses tools from our **CitNet** module to parse authors from original data and perform desambiguitation of           authors

      * 'DbScrap.py' uses tools from our **CitNet** module to scrap the RePec database

      * 'HubsAuths.py' contains our functions for the implementation of the HITS algorithm as well as example of use and visualization         at the end of the file.

      * 'PageRank.py' contains our implementation of the PageRank algorithm.

      * 'RefsCitsProcessing.py' uses tools from our **CitNet** module to construct the graph of citations & references from our raw           scrapped database

3 . Csv Files

    * The folder **Tables** contains the processed data that can be used easily to construct the collaboration graphs, the citations and     references graph and relate them to the attributes of the articles. 



