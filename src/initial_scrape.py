"""
This file is not tested on GitHub upload in 2023

- It was originally used to create the dataset for the model training.
- Data is stored as pickles.
- On project upload, the raw dataset was converted to a feather file.
"""
import requests
from bs4 import BeautifulSoup
import multiprocessing
import functools
import pickle

MIN_RATED = 30
MAX_WATCHED = 3200


def get_ratings_from_link(link):
    page = requests.get(link)
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


def get_ratings(username, pool=None):
    if pool is None:
        pool = multiprocessing.Pool(6)
    link = lambda i: "https://letterboxd.com/{}/films/page/{}".format(username, i)
    results = []
    done = False
    n = 1
    while not done:
        urls = [link(i) for i in range(n, n + 10)]
        results += pool.map(get_ratings_from_link, urls)
        if {} in results:
            done = True
        n += 10
    return functools.reduce(lambda x, y: {**x, **y}, results)


def get_usernames_from_link(link):
    usernames = set([])
    page = requests.get(link)
    soup = BeautifulSoup(page.content, "lxml")
    pieces = soup.select("a.has-icon.icon-16.icon-watched")
    for p in pieces:
        if int(p.getText().replace(",", "")) <= MAX_WATCHED:
            uname = p["href"].split("/")[1]
            usernames.add(uname)
    return usernames


def get_username_list(pool):
    link = lambda i: "https://letterboxd.com/people/popular/page/{}".format(i)
    results = pool.map(get_usernames_from_link, (link(i) for i in range(1, 129)))
    link = lambda i: "https://letterboxd.com/people/popular/this/week/page/{}".format(i)
    results += pool.map(get_usernames_from_link, (link(i) for i in range(1, 129)))
    link = lambda i: "https://letterboxd.com/people/popular/this/month/page/{}".format(
        i
    )
    results += pool.map(get_usernames_from_link, (link(i) for i in range(1, 129)))
    link = lambda i: "https://letterboxd.com/people/popular/this/year/page/{}".format(i)
    results += pool.map(get_usernames_from_link, (link(i) for i in range(1, 129)))
    return functools.reduce(lambda x, y: x | y, results)


def make_data(directory_path: str = "/pkl_data"):
    import os

    _, _, files = next(os.walk(directory_path))
    files = set([f.split(".")[0] for f in files])

    pool = multiprocessing.Pool(6)
    print("fetching usernames")
    usernames = get_username_list(pool)
    N = len(usernames)
    print(N, "users found")

    for c, username in enumerate(usernames):
        if username in files:
            continue
        data = get_ratings(username, pool)
        print("{} % done".format((c + 1) * 100 / N), end="\r")
        if len(data) > MIN_RATED:
            with open("{}/{}.pkl".format(directory_path, username), "wb") as f:
                pickle.dump(data, f)
    pool.close()
