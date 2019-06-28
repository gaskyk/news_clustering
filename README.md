# Clustering news articles

## Overview

This code will allow you to collect short news articles from the Guardian RSS feed, then cluster these according to their content.

The code is exploratory to enable me to understand the steps involved and is not a finished product.

The steps in the process are:
- Get RSS feeds from the Guardian website
- Use doc2vec to numerically represent the articles
- Use t-SNE to reduce the dimension of the document vectors and to plot clusters of similar articles
- Cluster similar articles together based on the doc2vec vectors

## How do I use this project?

The following packages are required:
- feedparser
- re
- string
- nltk
- gensim
- numpy
- sklearn
- pandas
- seaborn