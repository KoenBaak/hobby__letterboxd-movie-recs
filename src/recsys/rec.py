import numpy as np
from random import sample
import math
from timeit import default_timer

from recsys.foreigner import Foreigner

from recsys.fast_methods import _fast_predict, _fast_train, _fast_validation_metrics


class RecSys:
    def __init__(self, rm, nfactors=150, lr=0.001, reg=0.02):
        self._rm = rm
        self._nusers = rm._nusers
        self._nmovies = rm._nmovies
        self._foreigners = set()
        self._nf = nfactors
        self._test = set()
        self._U = None
        self._M = None
        self._user_bias = None
        self._movie_bias = None
        self._lr = lr
        self._reg = reg

    def set_foreigners(self, perc):
        self._foreigners = set(sample(range(self._nusers), int(perc * self._nusers)))

    def set_testset(self, perc):
        aux = []
        for uid in self._rm._data:
            if not uid in self._foreigners:
                aux += [(uid, mid) for mid in self._rm._data[uid]]
        self._test = set(sample(aux, int(perc * len(aux))))

    def _predict(self, uid, mid):
        return _fast_predict(
            self._user_bias[uid],
            self._movie_bias[mid],
            self._U[uid],
            self._M[mid],
            self._rm.global_mean(),
            self._nf,
        )

    def initialize(self, mean=0.0, sd=0.01):
        self._U = np.random.normal(mean, sd, (self._nusers, self._nf))
        self._M = np.random.normal(mean, sd, (self._nmovies, self._nf))
        self._user_bias = np.zeros(self._nusers)
        self._movie_bias = np.zeros(self._nmovies)

    def trainset(self):
        for uid in range(self._nusers):
            if uid in self._foreigners:
                continue
            for mid, rating in self._rm._data.get(uid, {}).items():
                if (uid, mid) in self._test:
                    continue
                yield uid, mid, rating

    def testset(self):
        for uid, mid in self._test:
            yield uid, mid, self._rm._data[uid][mid]

    def numpy_trainset(self):
        return np.array(list(self.trainset()))

    def numpy_testset(self):
        return np.array(list(self.testset()))

    def train(self, epochs=200, shuffle=False, verbose=True, time=True):
        if self._U is None:
            if verbose:
                print("initializing")
            self.initialize()

        if time:
            start = default_timer()

        trainset = self.numpy_trainset()
        self._U, self._M, self._user_bias, self._movie_bias = _fast_train(
            epochs,
            trainset,
            self._user_bias,
            self._movie_bias,
            self._U,
            self._M,
            self._rm.global_mean(),
            self._lr,
            self._reg,
            shuffle,
            verbose,
        )

        if time:
            return default_timer() - start

    def validate(self, verbose=True):
        testset = self.numpy_testset()
        N = testset.shape[0]
        rmse, mae, mse = _fast_validation_metrics(
            self.numpy_testset(),
            self._U,
            self._M,
            self._user_bias,
            self._movie_bias,
            self._rm.global_mean(),
            self._nf,
        )
        if verbose:
            print("N    :", N)
            print("rmse :", rmse)
            print("mae  :", mae)
        return rmse, mae, mse

    def _movie_cosine_similarity(self, mid1, mid2):
        vector1 = self._M[mid1]
        vector2 = self._M[mid2]

        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )

    def similar_movies(self, moviename, show_top=10):
        mid = self._rm._mid.get(moviename, False)
        if not mid:
            raise ValueError("Movie {} not in data".format(moviename))
        movie_similarity = []
        for omid in range(self._nmovies):
            if mid != omid:
                name = self._rm._mid.inverse[omid]
                movie_similarity.append(
                    (name, self._movie_cosine_similarity(mid, omid))
                )
        movie_similarity = sorted(movie_similarity, key=lambda x: x[1], reverse=True)[
            :show_top
        ]

        name_length = max(len(x[0]) for x in movie_similarity)
        print("name".ljust(name_length + 3), "cosine similarity")
        print(
            "-------------------------------------------------------------------------------"
        )
        for x in movie_similarity:
            print(x[0].ljust(name_length + 3), f"{x[1]:.2f}")

    def inspect_feature(self, n, top=10):
        aux = np.hstack((np.asmatrix(range(self._rm._nmovies)).transpose(), self._M))
        aux = np.array(aux)
        aux = aux[aux[:, n].argsort()][::-1]
        result = [(self._rm._mid.inverse[int(x[0])], x[n]) for x in aux]
        name_length = max(len(x[0]) for x in result)
        print("name".ljust(name_length + 3), "value")
        print(
            "-------------------------------------------------------------------------------"
        )
        for x in result[:top]:
            print(x[0].ljust(name_length + 3), f"{x[1]:.2f}")
        print("                  ....                       ")
        for x in result[-1 * top :]:
            print(x[0].ljust(name_length + 3), f"{x[1]:.2f}")

    def foreigner_test(self, epochs=200, verbose=True):
        N = len(self._foreigners)
        total = 0
        fs = [Foreigner.from_ratingmatrix(self, uid) for uid in self._foreigners]
        if verbose:
            print("foreigners made")
        total_obs = set()
        for i, f in enumerate(fs):
            for movie in f._data:
                total_obs.add((i, self._rm._mid[movie]))
        M = len(total_obs)
        testset = set(sample(list(total_obs), int(0.1 * M)))
        K = len(testset)
        if verbose:
            print("testset made")
        for i, mid in testset:
            del fs[i]._data[self._rm._mid.inverse[mid]]
        for f in fs:
            f.initialize()
            f.train(verbose=False)
        if verbose:
            print("foreigners trained")
        for i, mid in testset:
            total += (
                self._rm._data[self._rm._uid[fs[i]._username]][mid] - fs[i].predict(mid)
            ) ** 2
        return math.sqrt(total / K)
        for uid in self._foreigners:
            f = Foreigner.from_ratingmatrix(self, uid)
            f.initialize()
            f.train(epochs=epochs, verbose=False)
            rmse, mae, mse = f.validate(verbose=False)
            total += rmse
            if verbose:
                print("foreigner", uid, "rmse:", rmse)
        return total / N
