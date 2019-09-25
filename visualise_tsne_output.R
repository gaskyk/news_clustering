##########################################
##                                      ##
##  Plotly scatter plot of tSNE output  ##
##  Author: gaskyk                      ##
##  Date: 25-Sep-19                     ##
##                                      ##
##########################################

# Required packages
library(tidyverse)
library(plotly)

# Import data
tsne <- readr::read_csv("~/DataScience/news_clustering/TSNE_output.csv")

# Create scatter plot
p <- plotly::plot_ly(data = tsne, x = ~x, y = ~y,
                     type='scatter',
                     mode='markers',
                     # Hover text:
                     text = ~articles,
                     hoverinfo='text')
p



