"""
Cluster news articles based on collected data
=============================================

- Get RSS feeds from the Guardian
- Use doc2vec to numerically represent the articles
- Use t-SNE to reduce the dimension of the document vectors and to plot clusters of similar articles
- Cluster similar articles together based on the doc2vec vectors

"""

# Import packages
import collect_rss_feeds as collect
import cluster_articles as cluster

# Main functions
collect.collect_rss_feeds('http://www.theguardian.com/uk/rss')
feeds = collect.get_collected_rss_feeds()
newsgroups = collect.get_clean_20_newsgroups(categories=['sci.space'])
doc_vectors = cluster.run_doc2vec(feeds)
cluster.tsne_plot_and_csv(doc_vectors, feeds, csv_output=True)
cluster.create_scree_plot(doc_vectors, 20)
