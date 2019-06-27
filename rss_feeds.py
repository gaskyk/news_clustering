"""
Get RSS feeds from BBC news website

Requirements
:requires: feedparser
:requires: re
:requires: string
:requires: nltk
:requires: gensim
:requires: numpy
:requires: sklearn
:requires: pandas

"""

# Import packages
import feedparser
import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def get_rss_feed(url):
    """
    Function to get RSS feed from a chosen news service
    :param url: URL of RSS news feed
    :type url: str
    :return: d
    :rtype: feedparser.FeedParserDict
    """
    d = feedparser.parse(url)
    return d

guardian_feed = 'http://www.theguardian.com/uk/rss'
d = get_rss_feed(guardian_feed)

# Combine headlines and descriptions into one string
headlines = []
for i in d['entries']:
    headlines.append(i['summary'])

# Combine headlines and descriptions into one string
summaries = []
for i in d['entries']:
    temp = i['title'] + " " + i['summary']
    summaries.append(temp)

# Summaries data looks messy as it contains html tags
# p, a, li tags, ul, em, strong, span
def clean_rss_feed(text):
    """
    Function to remove a handful of known html tags (p,
    a, li, ul, em, strong, span and a 'continue reading'
    phrase at the end of each feed
    :param text: A list of RSS feed strings
    :type text: list of strings
    :return cleaned_text: A cleaned list of RSS feed strings
    :rtype cleaned_text: list of strings
    """
    # Remove HTML tags
    tags_removed = [re.sub("</?(a|p|li|ul|em|strong|span|time|br)[^>]*>", " ", i) for i in text]
    # Remove continue reading phrase
    cleaned_text = [re.sub("Continue reading...", "", i) for i in tags_removed]
    return cleaned_text

headlines = clean_rss_feed(headlines)
feeds_no_html = clean_rss_feed(summaries)

def removeNonAscii(text):
    """
    Function to remove non-ascii characters
    :param text: A list of strings which may contain non-ascii characters
    :type text: list of strings
    :return rtext: A list of strings which won't contain non-ascii characters
    :rtype rtext: list of strings
    """
    text = [i.encode('ascii','ignore') for i in text]
    rtext = [i.decode("utf-8") for i in text]
    return rtext

feeds_no_html = removeNonAscii(feeds_no_html)

# Save list of strings to a text file
with open('guardian_rss.txt', 'w') as f:
    for item in feeds_no_html:
        f.write("%s\n" % item)

# Read in saved list of strings
with open('guardian_rss.txt', 'r') as f:
    feeds_no_html = f.readlines()

def remove_punct_lower_case(text):
    """
    Function to remove punctuation and convert text to lower case
    in a list of strings
    :param text: A list of RSS feeds
    :type text: list of strings
    :return cleaned_text: A list of RSS feeds in lower case and without punctuation
    :rtype cleaned_text: list of strings
    """
    text_no_punct = [''.join(c for c in s if c not in punctuation) for s in text]
    cleaned_text = [i.lower() for i in text_no_punct]
    return cleaned_text

word_tokens = [word_tokenize(sentence) for sentence in feeds_no_html]


def remove_stopwords(my_string):
    """
    Function to remove stop words from a string
    :param text: A sentence
    :type text: string
    :return sentence_no_stopwords: A sentence containing no stop words
    :rtype sentence_no_stopwords: string
    """
    stop_words = set(stopwords.words('english'))
    sentence_no_stopwords = [w for w in my_string if not w in stop_words]
    return sentence_no_stopwords

word_tokens = [remove_stopwords(i) for i in word_tokens]

# Create TaggedDocument required as input for Doc2vec and run model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(word_tokens)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# Can see individual vectors
print(model.docvecs[1])

doc_vectors = np.array(model.docvecs.vectors_docs)

# Run Kmeans clustering

km_model = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1)
km_model.fit(doc_vectors)

# Kmeans labels
labels = km_model.labels_.tolist()

# Merge cluster labels and headlines to pandas dataframe
clusters_headlines = pd.DataFrame(
    {'cluster_labels': labels,
     'headlines': headlines
    })

# Save as CSV
clusters_headlines.to_csv("clusters_headlines.csv", encoding='utf-8')

