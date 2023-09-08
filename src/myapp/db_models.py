import os
from myapp import db, ILLEGAL_GENRES
from letterboxd_scrape import find_tmdbid
import tmdbsimple as tmdb

tmdb.API_KEY = os.environ["TMDB_API_KEY"]

movie_genres = db.Table(
    "movie_genres",
    db.Column("movie_link", db.String, db.ForeignKey("movies.letterboxd_link")),
    db.Column("genre_id", db.Integer, db.ForeignKey("genres.tmdb")),
)


movie_directors = db.Table(
    "movie_directors",
    db.Column("movie_link", db.String, db.ForeignKey("movies.letterboxd_link")),
    db.Column("director_id", db.Integer, db.ForeignKey("directors.tmdb")),
)


class Genre(db.Model):
    __tablename__ = "genres"

    tmdb = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    def __repr__(self):
        return "<Genre: {}>".format(self.name)


class Director(db.Model):
    __tablename__ = "directors"

    tmdb = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)

    def __repr__(self):
        return "<Directors: {}>".format(self.name)


class Movie(db.Model):
    __tablename__ = "movies"

    letterboxd_link = db.Column(db.String, primary_key=True)
    tmdb = db.Column(db.Integer, unique=True)
    title = db.Column(db.String, default="")
    original_title = db.Column(db.String, default="")
    poster = db.Column(db.String, default="")
    trailer = db.Column(db.String, default="")
    synopsis = db.Column(db.Text, default="")
    year = db.Column(db.Integer, default=-1)
    ignore_me = db.Column(db.Boolean, default=False)
    rating = db.Column(db.Float, default=-1.0)

    genres = db.relationship(
        "Genre",
        secondary=movie_genres,
        backref=db.backref("movies", lazy="dynamic"),
        lazy="dynamic",
    )
    directors = db.relationship(
        "Director",
        secondary=movie_directors,
        backref=db.backref("movies", lazy="dynamic"),
        lazy="dynamic",
    )

    def poster_path(self):
        return "https://image.tmdb.org/t/p/w342/{}".format(self.poster)

    def director_string(self):
        return ", ".join(d.name for d in self.directors)

    def __repr__(self):
        return "<Movie: {}>".format(self.letterboxd_link)


# -------------------------------------------------------------------------------


def make_movie(letterboxd_link):
    mid = find_tmdbid(letterboxd_link)
    movie = tmdb.Movies(mid)
    movie.info()
    movie.credits()
    title = movie.title
    original_title = movie.original_title
    year = int(movie.release_date.split("-")[0])
    poster = movie.poster_path
    synopsis = movie.overview

    genres = movie.genres
    directors = []
    for x in movie.crew:
        if x["job"] == "Director":
            directors.append({"id": x["id"], "name": x["name"]})

    ignore = False
    for g in genres:
        if g["name"] in ILLEGAL_GENRES:
            ignore = True

    movie_entry = Movie(
        letterboxd_link=letterboxd_link,
        tmdb=mid,
        title=title,
        original_title=original_title,
        year=year,
        poster=poster,
        synopsis=synopsis,
        ignore_me=ignore,
    )
    return movie_entry, genres, directors
