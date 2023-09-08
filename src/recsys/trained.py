import pickle
import json
import numpy as np
from recsys.fast_methods import _fast_foreign_train, _fast_cv


class TrainedModel:
    def __init__(self, recsys=None):
        self._movie_indices = {}
        self._M = None
        self._b = None
        self._lr = None
        self._reg = None
        self._global_mean = None
        if recsys is not None:
            self._M = recsys._M.copy()
            self._b = recsys._movie_bias.copy()
            self._movie_indices = dict(recsys._rm._mid)
            self._lr = recsys._lr
            self._reg = recsys._reg
            self._global_mean = recsys._rm.global_mean()

    @classmethod
    def read_json(cls, path: str) -> "TrainedModel":
        # this function was made in 2023 to facilitate creation from json
        # instead of pickle
        with open(path, "r") as f:
            data = json.load(f)
        model = cls()
        model._movie_indices = data["indices"]
        model._M = np.array(data["matrix"])
        model._b = np.array(data["bias"])
        model._lr = data["lr"]
        model._reg = data["reg"]
        model._global_mean = data["gm"]
        return model

    @staticmethod
    def load_pkl(filename):
        f = open(filename, "rb")
        d = pickle.load(f)
        f.close()
        model = TrainedModel()
        model._movie_indices = d["indices"]
        model._M = d["matrix"]
        model._b = d["bias"]
        model._lr = d["lr"]
        model._reg = d["reg"]
        model._global_mean = d["gm"]
        return model

    def pickle(self, filename):
        d = {}
        d["matrix"] = self._M
        d["indices"] = self._movie_indices
        d["bias"] = self._b
        d["lr"] = self._lr
        d["reg"] = self._reg
        d["gm"] = self._global_mean
        f = open(filename, "wb")
        pickle.dump(d, f)
        f.close()

    @property
    def nf(self):
        return self._M.shape[1]

    def trainset_gen(self, data):
        for movie, rating in data.items():
            if movie in self._movie_indices:
                yield self._movie_indices[movie], rating

    def cv(self, data, folds):
        trainset = np.array(list(self.trainset_gen(data)))
        print(trainset)
        return _fast_cv(
            trainset, self._b, self._M, self._global_mean, folds, self._lr, self._reg
        )

    def train(self, data):
        trainset = np.array(list(self.trainset_gen(data)))
        N = len(trainset)
        bias, U = _fast_foreign_train(
            200,
            trainset,
            np.random.normal(0, 0.1),
            self._b,
            np.random.normal(0, 0.1, self.nf),
            self._M,
            self._global_mean,
            self._lr,
            self._reg,
            False,
            False,
        )
        return N, bias, U

    def recommend(self, bias, U, exclude):
        predictions = self._global_mean + bias + U.dot(self._M.transpose()) + self._b
        predictions = np.minimum(np.maximum(predictions, 1), 10)
        predictions = {
            movie: predictions[self._movie_indices[movie]]
            for movie in self._movie_indices
            if not movie in exclude
        }
        predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return predictions
