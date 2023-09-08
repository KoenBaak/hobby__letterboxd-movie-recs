import requests
from bs4 import BeautifulSoup
import multiprocessing
import functools


def find_tmdbid(movie_link):
    response = requests.get("https://letterboxd.com/film/{}".format(movie_link))
    soup = BeautifulSoup(response.content, "lxml")
    return int(soup.select("body")[0]["data-tmdb-id"])


def _get_ratings_from_link(link):
    page = requests.get(link)
    if page.status_code == 404:
        return 404
    soup = BeautifulSoup(page.content, "lxml")
    poster_containers = soup.select("li.poster-container")
    ratings = {}
    for pc in poster_containers:
        film_link = pc.select("div.poster")[0]["data-target-link"].split("/")[2]
        rating_element = pc.select("span.rating")
        if not rating_element:
            continue
        rating = int(rating_element[0]["class"][3].split("-")[1])
        ratings[film_link] = rating
    return ratings


def _get_ratings_multiprocessing(username, processes):
    pool = multiprocessing.Pool(processes)
    link = lambda i: "https://letterboxd.com/{}/films/page/{}".format(username, i)
    results = []
    n = 1
    while True:
        urls = [link(i) for i in range(n, n + 2 * processes)]
        results += pool.map(_get_ratings_from_link, urls)
        if 404 in results:
            return 404
        if {} in results:
            break
        n += 2 * processes
    pool.close()
    return functools.reduce(lambda x, y: {**x, **y}, results)


def _get_ratings(username):
    link = lambda i: "https://letterboxd.com/{}/films/page/{}".format(username, i)
    ratings = {}
    i = 1
    while True:
        url = link(i)
        new = _get_ratings_from_link(url)
        if new == 404:
            return 404
        if not new:
            break
        ratings = {**ratings, **new}
        i += 1
    return ratings


def get_ratings(username, processes=0):
    if processes > 0:
        return _get_ratings_multiprocessing(username, processes)
    return _get_ratings(username)
