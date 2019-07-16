"""
Cluster news articles based on collected data
=============================================

- Get RSS feeds from the Guardian
- Use doc2vec to numerically represent the articles
- Use t-SNE to reduce the dimension of the document vectors and to plot clusters of similar articles
- Cluster similar articles together based on the doc2vec vectors

Requirements
:requires: string
:requires: nltk
:requires: gensim
:requires: numpy
:requires: sklearn
:requires: pandas
:requires: seaborn

"""

# Import packages
import collect_rss_feeds as collect
from sklearn.datasets import fetch_20newsgroups

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

"""
GET RSS FEED, CLEAN TEXT THEN SAVE TO TXT FILE
"""

collect.collect_rss_feeds('http://www.theguardian.com/uk/rss')

"""
GET 20 NEWSGROUPS DATASET AS RSS FEEDS DON'T CONTAIN ENOUGH DATA
"""

def get_clean_20_newsgroups(categories=None):
    """
    Function to get the 20 newsgroups dataset
    Can pass optional argument categories as a list of categories
    Cleaning simply replaces \n (new paragraph) with space
    :param categories: List of categories to get from 20 newsgroups
    :type categories: list of strings
    :return newsgroups_data: The training set for the 20 newsgroups dataset
    :rtype newsgroups_data: list of strings
    """
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                          categories=categories)
    newsgroups_data = newsgroups_train.data
    newsgroups_data = [i.replace('\n', ' ') for i in newsgroups_data]
    return newsgroups_data

newsgroups = get_clean_20_newsgroups(categories=['sci.space'])

"""
TOKENISE SENTENCES THEN USE DOC2VEC FOR NUMERICAL REPRESENTATION OF ARTICLES
"""

# Read feeds and headlines from text file
with open('guardian_rss.txt', 'r') as f:
    feeds = f.readlines()
with open('guardian_headlines_rss.txt', 'r') as f:
    headlines = f.readlines()

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

word_tokens = [word_tokenize(sentence) for sentence in newsgroups]

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

def run_doc2vec_model(words):
    """
    Function to run the doc2vec model from a lists of words (tokens).
    Output is a
    :param words: Lists of words or tokens
    :type words: list of list of words
    :return doc_vectors: Document vectors, a 10xn shaped array,
    where n is the number of documents
    :rtype doc_vectors: numpy.ndarray
    """
    # Create TaggedDocument required as input for Doc2vec
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(words)]
    # Run model
    model = Doc2Vec(documents, vector_size=10, window=2, min_count=2, workers=4)
    # Output vectors as a numpy array
    doc_vectors = np.array(model.docvecs.vectors_docs)
    return doc_vectors

doc_vectors = run_doc2vec_model(word_tokens)

"""
DIMENSION REDUCTION USING T-SNE
"""

def tsne_plot_and_csv(document_vectors, articles, csv_output):
    """
    Function to reduce dimensions of document vectors to two
    dimensions (for plotting). Function then plots output
    and outputs CSV including details of nearest neighbour
    :param document_vectors: Document vectors from doc2vec output
    :type document_vectors: numpy.array
    :param articles: List of articles or headlines summarising articles
    :type articles: list of strings
    :param csv_output: Name of file for output to CSV
    :type csv_output: str
    """
    # Run t-SNE
    tsne_on_docvecs = TSNE(n_components=2).fit_transform(document_vectors)
    # Find nearest neighbour headline
    tree = KDTree(tsne_on_docvecs)
    nearest_dist, nearest_ind = tree.query(tsne_on_docvecs, k=2)  # k=2 nearest neighbors where k1 = identity
    nearest_dist = nearest_dist[:, 1].tolist()
    nearest_ind = nearest_ind[:, 1].tolist()
    nearest_articles = []
    for i in nearest_ind:
        nearest_articles.append(articles[i])
    # Summarise t-SNE output
    tsne_plot_data = pd.DataFrame({'articles': articles,
                                   'nearest_articles': nearest_articles,
                                   'nearest_distance': nearest_dist,
                                   'x':tsne_on_docvecs[:,0],
                                   'y':tsne_on_docvecs[:,1]})
    # Save as CSV
    tsne_plot_data.to_csv(csv_output, encoding='utf-8')
    # Plot t-SNE output
    sns.scatterplot(tsne_plot_data["x"], tsne_plot_data["y"]).plot()

tsne_plot_and_csv(doc_vectors, newsgroups, "TSNE_output.csv")

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

