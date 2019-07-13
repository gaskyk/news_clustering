"""
Collect RSS feeds
=================

- Get RSS feeds
- Clean feeds
- Save to text files

Requirements
:requires: feedparser
:requires: re

"""

# Import packages
import feedparser
import re


def collect_rss_feeds(url):
    """
    Function to combine other functions in this file
    Collect, clean, then save RSS feeds
    :param: URL of RSS feed
    :return: Append headlines and summaries to two text files
    """
    d = get_rss_feed(url)

    # Headlines
    headlines = []
    for i in d['entries']:
        headlines.append(i['summary'])
    headlines = clean_rss_feed(headlines)
    headlines = remove_non_ascii(headlines)
    save_to_txt(headlines, 'guardian_headlines_rss.txt')

    # Summaries - combines headlines and rest of text
    summaries = []
    for i in d['entries']:
        temp = i['title'] + " " + i['summary']
        summaries.append(temp)
    summaries = clean_rss_feed(summaries)
    summaries = remove_non_ascii(summaries)
    save_to_txt(summaries, 'guardian_rss.txt')


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


def remove_non_ascii(text):
    """
    Function to remove non-ascii characters
    :param text: A list of strings which may contain non-ascii characters
    :type text: list of strings
    :return rtext: A list of strings which won't contain non-ascii characters
    :rtype rtext: list of strings
    """
    text = [i.encode('ascii', 'ignore') for i in text]
    rtext = [i.decode("utf-8") for i in text]
    return rtext


def save_to_txt(my_list, name_of_file):
    """
    Function to save a list to a text file
    :param my_list: List to save to file
    :type my_list: list
    :param name_of_file: Name of file
    :type name_of_file: str
    """
    with open(name_of_file, 'a') as f:
        for item in my_list:
            f.write("%s\n" % item)
