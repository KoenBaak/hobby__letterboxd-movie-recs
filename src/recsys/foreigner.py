from letterboxd_scrape import get_ratings
import numpy as np
from timeit import default_timer

from recsys.fast_methods import (
    _fast_predict,
    _fast_foreign_train,
    _fast_foreign_validation,
)


class Foreigner:
    def __init__(self, rc, username, do_scrape=False):
        self._rc = rc
        self._rm = rc._rm
        self._username = username
        self._data = None
        self._U = None
        self._bias = None
        if do_scrape:
            self.scrape()

    @staticmethod
    def from_ratingmatrix(rc, uid):
        name = rc._rm._uid.inverse[uid]
        f = Foreigner(rc, name, do_scrape=False)
        f._data = {}
        for mid, r in rc._rm._data.get(uid, {}).items():
            f._data[rc._rm._mid.inverse[mid]] = r
        return f

    @property
    def username(self):
        return self._username

    def scrape(self):
        self._data = get_ratings(self._username)
        # for movie in self._data:
        #    self._data[movie] = self._data[movie] / 2

    def ratings(self):
        return self._data

    def _trainset(self):
        for movie in self._data:
            if not movie in self._rm._mid:
                continue
            mid = self._rm._mid[movie]
            yield mid, self._data[movie]

    def numpy_trainset(self):
        return np.array(list(self._trainset()))

    def initialize(self, mean=0, sd=0.1):
        self._bias = np.random.normal(mean, sd)
        self._U = np.random.normal(mean, sd, self._rc._nf)

    def train(self, epochs=200, shuffle=False, verbose=True, time=True):
        if self._U is None:
            if verbose:
                print("initializing")
            self.initialize()

        if time:
            start = default_timer()

        if self._data:
            trainset = self.numpy_trainset()
            self._bias, self._U = _fast_foreign_train(
                epochs,
                trainset,
                self._bias,
                self._rc._movie_bias,
                self._U,
                self._rc._M,
                self._rm.global_mean(),
                self._rc._lr,
                self._rc._reg,
                shuffle,
                verbose,
            )

        if time:
            return default_timer() - start

    def validate(self, verbose=False):
        if self._data:
            dataset = self.numpy_trainset()
            rmse, mae, mse = _fast_foreign_validation(
                dataset,
                self._U,
                self._rc._M,
                self._bias,
                self._rc._movie_bias,
                self._rm.global_mean(),
                self._rc._nf,
            )
            if verbose:
                print("N    :", dataset.shape[0])
                print("rmse :", rmse)
                print("mae  :", mae)
            return rmse, mae, mse
        return 0, 0, 0

    def predict(self, mid):
        return _fast_predict(
            self._bias,
            self._rc._movie_bias[mid],
            self._U,
            self._rc._M[mid],
            self._rm.global_mean(),
            self._rc._nf,
        )

    def _summary_gen(self):
        for movie in self._data:
            if not movie in self._rm._mid:
                continue
            mid = self._rm._mid[movie]
            true = self._data[movie]
            pred = self.predict(mid)
            yield movie, mid, true, pred, abs(true - pred)

    def _recommendation_gen(self):
        for mid in range(self._rm._nmovies):
            movie = self._rm._mid.inverse[mid]
            if movie in self._data:
                continue
            yield movie, mid, self.predict(mid)

    def summary(self, print_all=False, nrecs=15):
        summary_data = list(self._summary_gen())
        summary_data = sorted(summary_data, key=lambda x: x[4])
        name_length = max(len(x[0]) for x in summary_data)
        N = len(summary_data)
        rmse = sum(x[4] ** 2 for x in summary_data) / N
        mae = sum(x[4] for x in summary_data) / N

        recommendations = sorted(
            list(self._recommendation_gen()), key=lambda x: x[2], reverse=True
        )

        YELLOW = lambda x: "\033[93m" + x + "\033[0m"  # for nice terminal printing
        RED = lambda x: "\033[31m" + x + "\033[0m"
        GREEN = lambda x: "\033[32m" + x + "\033[0m"

        print(
            RED(
                "-------------------------------------------------------------------------------"
            )
        )
        print("name                                  :", self._username)
        print("movies rated in recommendation system :", N)
        print("rmse                                  :", rmse)
        print("mae                                   :", mae)
        print(
            "-------------------------------------------------------------------------------"
        )
        print(GREEN("           Predictions for already rated movies"))
        print(
            "name".ljust(name_length + 3),
            "actual rating",
            " pred rating ",
            "absolute error",
        )
        print(
            "-------------------------------------------------------------------------------"
        )
        if print_all:
            for x in summary_data:
                print(
                    x[0].ljust(name_length + 3),
                    f"{x[2]:.2f}".ljust(13),
                    f"{x[3]:.2f}".ljust(13),
                    f"{x[4]:.2f}",
                )
        else:
            for x in summary_data[:4]:
                print(
                    x[0].ljust(name_length + 3),
                    f"{x[2]:.2f}".ljust(13),
                    f"{x[3]:.2f}".ljust(13),
                    f"{x[4]:.2f}",
                )
            print("                              ...")
            for x in summary_data[-4:]:
                print(
                    x[0].ljust(name_length + 3),
                    f"{x[2]:.2f}".ljust(13),
                    f"{x[3]:.2f}".ljust(13),
                    f"{x[4]:.2f}",
                )
        print(
            "-------------------------------------------------------------------------------"
        )
        print(GREEN("                    Recommendations"))
        print("name".ljust(name_length + 3), " pred rating ")
        print(
            "-------------------------------------------------------------------------------"
        )
        for x in recommendations[:nrecs]:
            print(x[0].ljust(name_length + 3), f"{x[2]:.2f}")
        print(
            RED(
                "-------------------------------------------------------------------------------"
            )
        )

    def consensus_compare(self, ascending=True, show_top=10):
        result = []
        for movie in self._data:
            if movie not in self._rm._mid:
                continue
            mid = self._rm._mid[movie]
            my_rating = self._data[movie]
            mo = self._rm.get_movie(mid)
            mean = mo.mean()
            gr = mo.global_rating()
            result.append((movie, my_rating, mean, my_rating - mean, gr))
        result = sorted(result, key=lambda x: abs(x[3]), reverse=ascending)[:show_top]

        YELLOW = lambda x: "\033[93m" + x + "\033[0m"  # for nice terminal printing
        RED = lambda x: "\033[31m" + x + "\033[0m"
        GREEN = lambda x: "\033[32m" + x + "\033[0m"
        name_length = max(len(x[0]) for x in result)
        print(
            "name".ljust(name_length + 3),
            GREEN("user rating"),
            YELLOW("  global mean  "),
            RED("deviation"),
        )
        print(
            "-------------------------------------------------------------------------------"
        )
        for x in result:
            print(
                x[0].ljust(name_length + 3),
                GREEN(f"{x[1]:.2f}".ljust(15)),
                YELLOW(f"{x[2]:.2f}".ljust(11)),
                RED(f"{x[3]:.2f}"),
            )
