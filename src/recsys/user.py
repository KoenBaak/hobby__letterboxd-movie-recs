class User:
    def __init__(self, rm, uid):
        self._rm = rm
        self._id = uid

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._rm._uid.inverse[self._id]

    def ratings(self):
        return self._rm._data.get(self._id, {})

    def watched(self):
        return len(self.ratings())

    def mean(self):
        m, i = 0, 0
        for mid, r in self.ratings().items():
            m += r
            i += 1
        return m / i
