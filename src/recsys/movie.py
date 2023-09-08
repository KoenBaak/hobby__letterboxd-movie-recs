class Movie:
    def __init__(self, rm, mid):
        self._rm = rm
        self._id = mid

    @property
    def id(self):
        return self._id

    @property
    def title(self):
        return self._rm._mid.inverse[self._id]

    def ratedby(self):
        return self._rm._ratedby.get(self._id, [])

    def votes(self):
        return len(self.ratedby())

    def ratings(self):
        result = {}
        for uid in self.ratedby():
            result[uid] = self._rm._data[uid][self._id]
        return result

    def mean(self):
        m, i = 0, 0
        for uid, r in self.ratings().items():
            m += r
            i += 1
        return m / i

    def global_rating(self):
        w = len(self.ratedby()) / self._rm._nusers
        return w * self.mean() + (1 - w) * self._rm.global_mean()
