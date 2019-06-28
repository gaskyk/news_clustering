"""
Cluster news articles based on collected data
=============================================

- Get RSS feeds from the Guardian
- Use doc2vec to numerically represent the articles
- Use t-SNE to reduce the dimension of the document vectors and to plot clusters of similar articles
- Cluster similar articles together based on the doc2vec vectors

Requirements
:requires: feedparser
:requires: re
:requires: string
:requires: nltk
:requires: gensim
:requires: numpy
:requires: sklearn
:requires: pandas
:requires: seaborn

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

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

"""
GET RSS FEED, CLEAN TEXT THEN SAVE TO TXT FILE
"""

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

# Headlines
headlines = []
for i in d['entries']:
    headlines.append(i['summary'])

# Combine headlines and descriptions into one string
summaries = []
for i in d['entries']:
    temp = i['title'] + " " + i['summary']
    summaries.append(temp)

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

def remove_non_ascii(text):
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

headlines = remove_non_ascii(headlines)
feeds_no_html = remove_non_ascii(feeds_no_html)

def save_to_txt(my_list, name_of_file):
    """
    Function to save a list to a text file
    :param my_list: List to save to file
    :type my_list: list
    :param name_of_file: Name of file
    :type name_of_file: str
    """
    with open(name_of_file, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)

save_to_txt(headlines, 'guardian_headlines_rss.txt')
save_to_txt(feeds_no_html, 'guardian_rss.txt')

"""
TOKENISE SENTENCES THEN USE DOC2VEC FOR NUMERICAL REPRESENTATION OF ARTICLES
"""

# Read feeds from text file
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
model = Doc2Vec(documents, vector_size=10, window=2, min_count=2, workers=4)

# Can see individual vectors
print(model.docvecs[1])

doc_vectors = np.array(model.docvecs.vectors_docs)

"""
DIMENSION REDUCTION USING T-SNE
"""

tsne_on_docvecs = TSNE(n_components=2).fit_transform(doc_vectors)

# Plot t-SNE output
tsne_plot_data = pd.DataFrame({'x':tsne_on_docvecs[:,0],
                               'y':tsne_on_docvecs[:,1]})
sns.scatterplot(tsne_plot_data["x"], tsne_plot_data["y"]).plot()

"""
CLUSTER THE ARTICLES USING K-MEANS
"""

# Run Kmeans clustering with different cluster sizes
sum_of_squared_distances = []
for i in range(1,16):
    km_model = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1)
    km_model.fit(doc_vectors)
    sum_of_squared_distances.append(km_model.inertia_)

# Create elbow plot to decide optimal number of clusters
elbow_plot_input = pd.DataFrame(
    {'clusters': list(range(1,16)),
     'sum_of_squared_distances': sum_of_squared_distances
    })
sns.lineplot(x="clusters", y="sum_of_squared_distances", data=elbow_plot_input)

# Kmeans labels
labels = km_model.labels_.tolist()

# Merge cluster labels and headlines to pandas dataframe
clusters_headlines = pd.DataFrame(
    {'cluster_labels': labels,
     'headlines': headlines
    })

# Save as CSV
clusters_headlines.to_csv("clusters_headlines.csv", encoding='utf-8')

