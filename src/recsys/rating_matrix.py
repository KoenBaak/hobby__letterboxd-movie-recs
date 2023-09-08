import json
import os
import pickle
from copy import deepcopy

from bidict import bidict

from recsys.user import User
from recsys.movie import Movie
import letterboxd_scrape as scrape


class RatingMatrix:
    def __init__(self):
        self._data = {}
        self._ratedby = {}
        self._nusers = 0
        self._nmovies = 0
        self._uid = bidict()
        self._mid = bidict()
        self._cached_gm = None

    def add_user_data(self, username, data):
        if username not in self._uid:
            self._uid[username] = self._nusers
            self._nusers += 1
        uid = self._uid[username]
        for movie, rating in data.items():
            if movie not in self._mid:
                self._mid[movie] = self._nmovies
                self._nmovies += 1
            mid = self._mid[movie]
            if not mid in self._ratedby:
                self._ratedby[mid] = []
            if uid not in self._data:
                self._data[uid] = {}
            self._data[uid][mid] = rating
            self._ratedby[mid].append(uid)

    @staticmethod
    def read_pickle_folder(path, verbose=True):
        _, _, files = next(os.walk(path))
        obj = RatingMatrix()
        N = len(files)
        for i, filename in enumerate(files):
            with open("{}/{}".format(path, filename), "rb") as f:
                uname = filename.split(".")[0]
                obj.add_user_data(uname, pickle.load(f))
                if verbose:
                    print("reading {:.2f} %".format((i + 1) / N * 100), end="\r")
        if verbose:
            print("\n", end="")
        return obj

    @classmethod
    def read_json(cls, path: str) -> "RatingMatrix":
        # this function was made in 2023 to facilitate creation from json
        # instead of pickle
        with open(path, "r") as f:
            data = json.load(f)
        rm = cls()
        usernames = data["users"]
        movielinks = data["movies"]
        for key, value in data["ratings"].items():
            rm.add_user_data(
                username=usernames[int(key)],
                data={movielinks[int(mid)]: r for mid, r in value.items()},
            )
        return rm

    def to_json(self, path: str):
        # this function was made in 2023 to facilitate writing to json
        # instead of pickle
        movies = [None] * len(self._mid.inverse)
        users = [None] * len(self._uid.inverse)
        for idx, m in self._mid.inverse.items():
            movies[idx] = m
        for idx, u in self._uid.inverse.items():
            users[idx] = u
        data = {
            "movies": movies,
            "users": users,
            "ratings": self._data,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def scrape_from_usernames(usernames):
        obj = RatingMatrix()
        for u in usernames:
            obj.add_user_data(u, scrape.get_ratings(u))
        return obj

    def _add_datapoint(self, username, moviename, rating):
        if username not in self._uid:
            self._uid[username] = self._nusers
            self._nusers += 1
        uid = self._uid[username]
        if moviename not in self._mid:
            self._mid[moviename] = self._nmovies
            self._nmovies += 1
        mid = self._mid[moviename]
        if not mid in self._ratedby:
            self._ratedby[mid] = []
        self._ratedby[mid].append(uid)
        if not uid in self._data:
            self._data[uid] = {}
        self._data[uid][mid] = rating

    @staticmethod
    def read_csv(filename, sep="::"):
        rm = RatingMatrix()
        with open(filename, "r") as f:
            for line in f.readlines():
                uid, mid, rating, _ = line.split(sep)
                rm._add_datapoint(
                    "user{}".format(uid), "movie{}".format(mid), int(rating)
                )
        return rm

    def votes(self, movies=None):
        if type(movies) == str:
            moviesit = [self._mid[movies]]
        elif movies is None:
            moviesit = iter(self._mid.inverse)
        else:
            moviesit = (self._mid[movie] for movie in movies)
        result = {}
        for mid in moviesit:
            result[mid] = len(self._ratedby[mid])
        return result

    def remove_movie(self, movie):
        mid = movie
        if type(movie) == str:
            mid = self._mid[movie]
        if mid not in self._mid.inverse:
            raise ValueError("Movie not found")
        self._remove_mid(mid)

    def _remove_mid(self, mid):
        del self._mid.inverse[mid]
        for oldmid in range(mid + 1, self._nmovies):
            movie = self._mid.inverse[oldmid]
            newmid = oldmid - 1
            self._mid[movie] = newmid
            for uid in self._ratedby[oldmid]:
                self._data[uid][newmid] = self._data[uid][oldmid]
                del self._data[uid][oldmid]
            self._ratedby[newmid] = self._ratedby[oldmid]
            del self._ratedby[oldmid]
        self._nmovies -= 1

    def _remove_multiple_mids(self):
        removed = 0
        try:
            while True:
                to_remove = yield
                del self._mid.inverse[to_remove]
                for uid in self._ratedby[to_remove]:
                    del self._data[uid][to_remove]
                del self._ratedby[to_remove]
                removed += 1
        except GeneratorExit:
            newmids = []
            for i in range(self._nmovies):
                if i in self._mid.inverse:
                    if newmids:
                        newmid = newmids.pop(0)
                        movie = self._mid.inverse[i]
                        self._mid[movie] = newmid
                        for uid in self._ratedby[i]:
                            self._data[uid][newmid] = self._data[uid][i]
                            del self._data[uid][i]
                        self._ratedby[newmid] = self._ratedby[i]
                        del self._ratedby[i]
                        newmids.append(i)
                    continue
                newmids.append(i)
            self._nmovies -= removed

    def global_mean(self, reset_cache=False):
        if self._cached_gm is not None and not reset_cache:
            return self._cached_gm
        gm = 0
        i = 0
        for uid, d in self._data.items():
            for mid, r in d.items():
                gm += r
                i += 1
        self._cached_gm = gm / i
        return self._cached_gm

    def copy(self):
        copy = RatingMatrix()
        copy._data = deepcopy(self._data)
        copy._mid = self._mid.copy()
        copy._uid = self._uid.copy()
        copy._ratedby = self._ratedby.copy()
        copy._nusers = self._nusers
        copy._nmovies = self._nmovies
        copy._cached_gm = self._cached_gm
        return copy

    def filter(self, topviewed=None, minvotes=None, maxvotes=None, inplace=False):
        if inplace:
            frm = self
        else:
            frm = self.copy()
        v = frm.votes()

        if minvotes is not None:
            cor = frm._remove_multiple_mids()
            cor.__next__()
            for mid in range(self._nmovies):
                if v[mid] < minvotes:
                    cor.send(mid)
            cor.close()

        """
        if topviewed is not None:
            v = frm.votes()
            s = sorted(frm._mid.inverse.keys(), key = lambda x: v[x], reverse=True)[:topviewed]
            print(len(s))
            to_remove = [frm._mid.inverse[mid] for mid in range(frm._nmovies) if mid not in s]
            print(len(to_remove))
            for movie in to_remove:    
                frm._remove_mid(frm._mid[movie])
        
        if minvotes is not None:
            i = 0
            while i < frm._nmovies:
                if Movie(frm, i).votes() >= minvotes:
                    i += 1
                else:
                    frm._remove_mid(i)
        
        if maxvotes is not None:
            i = 0
            while i < frm._nmovies:
                if Movie(frm, i).votes() <= maxvotes:
                    i += 1
                else:
                    frm._remove_mid(i)
        """
        if not inplace:
            return frm

    def get_user(self, arg):
        if type(arg) == str:
            uid = self._uid[arg]
        else:
            uid = arg
        return User(self, uid)

    def get_movie(self, arg):
        if type(arg) == str:
            mid = self._mid[arg]
        else:
            mid = arg
        return Movie(self, mid)

    def sparsity(self):
        v = self.votes()
        return sum(v[x] for x in range(self._nmovies)) / (self._nusers * self._nmovies)

    def rescale(self, targetscale=5):
        for uid in self._data:
            for mid in self._data[uid]:
                self._data[uid][mid] = self._data[uid][mid] / 2

    @property
    def top(self):
        pairs = [
            (self._mid.inverse[mid], Movie(self, mid).global_rating())
            for mid in self._mid.inverse
        ]
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def print_top(self, arg):
        if type(arg) != slice:
            arg = slice(0, arg, None)
        rjust_mid = max(len(name) for name in self._mid)
        print("-" * (15 + rjust_mid))
        for i, v in enumerate(self.top[arg]):
            print(
                "{}".format(arg.start + i + 1).ljust(6)
                + v[0].ljust(rjust_mid + 5)
                + "{:.2f}".format(v[1])
            )
        print("-" * (15 + rjust_mid))
