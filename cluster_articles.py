"""
Cluster news articles based on collected data
=============================================

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


def run_doc2vec(orig_articles):
    """
    Function to pull together all functions below ie.
    - Remove punctuation and convert to lower case
    - Tokenise sentences
    - Remove stop words
    - Run doc2vec algorithm to get a vector per article
    - Run t-SNE algorithm to reduce dimension of vectors and plot
    :param orig_articles: A set of articles eg. RSS feeds or 20 newsgroups data
    :type orig_articles: list of strings
    :return doc_vectors: Document vectors of articles
    :rtype doc_vectors: numpy.ndarray
    """
    articles = _remove_punct_lower_case(orig_articles)
    word_tokens = [word_tokenize(sentence) for sentence in articles]
    word_tokens = [_remove_stopwords(i) for i in word_tokens]
    doc2_vectors = _run_doc2vec_model(word_tokens)
    return doc2_vectors


def _remove_punct_lower_case(text):
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


def _remove_stopwords(my_string):
    """
    Function to remove stop words from a string
    :param my_string: A sentence
    :type my_string: string
    :return sentence_no_stopwords: A sentence containing no stop words
    :rtype sentence_no_stopwords: string
    """
    stop_words = set(stopwords.words('english'))
    sentence_no_stopwords = [w for w in my_string if not w in stop_words]
    return sentence_no_stopwords


def _run_doc2vec_model(words):
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
    document_vectors = np.array(model.docvecs.vectors_docs)
    return document_vectors


def tsne_plot_and_csv(document_vectors, articles, csv_output=False):
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
                                   'x': tsne_on_docvecs[:, 0],
                                   'y': tsne_on_docvecs[:, 1]})
    # Save as CSV
    if csv_output:
        tsne_plot_data.to_csv('TSNE_output.csv', encoding='utf-8')
    # Plot t-SNE output
    sns.scatterplot(tsne_plot_data["x"], tsne_plot_data["y"]).plot()


def create_scree_plot(document_vectors, number_clusters):
    """
    Function to run k-means clustering and show scree / elbow plot for
    selecting suitable number of clusters
    :param document_vectors: Document vectors from doc2vec algorithm
    :type document_vectors: np.array
    :param number_clusters: Maximum number of clusters for scree plot
    :type number_clusters: int
    """
    # Run Kmeans clustering with different cluster sizes
    sum_of_squared_distances = []
    for i in range(1, number_clusters):
        km_model = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1)
        km_model.fit(document_vectors)
        sum_of_squared_distances.append(km_model.inertia_)
    # Create elbow plot to decide optimal number of clusters
    elbow_plot_input = pd.DataFrame(
        {'clusters': list(range(1, number_clusters)),
         'sum_of_squared_distances': sum_of_squared_distances
         })
    sns.lineplot(x="clusters", y="sum_of_squared_distances", data=elbow_plot_input)
