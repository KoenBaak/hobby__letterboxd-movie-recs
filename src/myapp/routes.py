from flask import render_template, flash, redirect, url_for, request, Response

from myapp import app, db, APP_NAME, MODEL
from myapp.forms import UsernameForm
from myapp.db_models import Movie, Genre, Director, make_movie

from letterboxd_scrape import get_ratings

import os


@app.context_processor
def give_name():
    # make all templates aware of the app name
    return dict(app_name=APP_NAME, enumerate=enumerate)


# -------------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    form = UsernameForm()
    if form.validate_on_submit():
        return redirect(url_for("recommendations", username=form.username.data.lower()))
    return render_template("index.html", form=form)


@app.route("/recommendations/<username>")
def recommendations(username):
    loaded = request.args.get("loaded", False)
    if not loaded:
        target = url_for("recommendations", username=username) + "?loaded=True"
        message = "Getting ratings of <b>{}</b> and making recommendations...<br>This can take long depending on how many films you have rated".format(
            username
        )
        return render_template("loading.html", target=target, message=message)
    data = get_ratings(username)
    if data == 404:
        flash("Not a valid Letterboxd username")
        return redirect(url_for("index"))
    N, b, U = MODEL.train(data)
    recs = MODEL.recommend(b, U, data)[:1000]
    result = []
    for movie, prediction in recs:
        result.append(
            (Movie.query.filter_by(letterboxd_link=movie).first(), f"{prediction:.2f}")
        )
    print("---------------", MODEL.cv(data, 5))
    return render_template("recs.html", username=username, movies=result)


@app.route("/film/<film_link>")
def film_page(film_link):
    m = Movie.query.filter_by(letterboxd_link=film_link).first()
    if not m:
        flash("Movie not found in database.")
        return redirect(url_for("index"))
    return render_template("film.html", movie=m)


def moviedb_update_gen():
    db.create_all()
    log_file = open("db_update_log.txt", "w")
    ignore_file = open("db_ignore.txt", "w")
    counter = 0
    for movie_link in MODEL._movie_indices:
        counter += 1
        m = Movie.query.filter_by(letterboxd_link=movie_link).first()
        if m:
            yield "{}: movie {} already in db, skipping...<br>".format(
                str(counter).ljust(5), movie_link
            )
            continue
        try:
            m, genres, directors = make_movie(movie_link)
            db.session.add(m)
            if m.ignore_me:
                ignore_file.write("{}\n".format(movie_link))
                yield "{}: ignored {}<br>".format(str(counter).ljust(5), movie_link)
                continue
            for g in genres:
                ge = Genre.query.filter_by(tmdb=g["id"]).first()
                if not ge:
                    ge = Genre(tmdb=g["id"], name=g["name"])
                    db.session.add(ge)
                    yield ".......{}: added genre {}<br>".format(
                        str(counter).ljust(5), ge.name
                    )
                m.genres.append(ge)
                db.session.commit()
            for d in directors:
                di = Director.query.filter_by(tmdb=d["id"]).first()
                if not di:
                    di = Director(tmdb=d["id"], name=d["name"])
                    db.session.add(di)
                    yield ".......{}: added director {}<br>".format(
                        str(counter).ljust(5), di.name
                    )
                m.directors.append(di)
            yield "{}: added {}<br>".format(str(counter).ljust(5), movie_link)
        except Exception as e:
            yield "{}: Error when tryin to add {}, see logfile...<br>".format(
                str(counter).ljust(5), movie_link
            )
            log_file.write("error with {} : {} \n\n".format(movie_link, e))
            ignore_file.write("{}\n".format(movie_link))
    log_file.close()
    ignore_file.close()
    db.session.commit()
    yield "done"


@app.route("/_db_update")
def moviedb_update():
    return Response(moviedb_update_gen())


def retrain_gen():
    data_folder = "/home/koen/Desktop/rec/pkl_data"
    ignore_file = "db_ignore.txt"
    if not os.path.exists(data_folder):
        yield "No data folder found"
        return
    from recsys import RatingMatrix, RecSys, TrainedModel

    yield "reading data<br>"
    rm = RatingMatrix.read_pickle_folder(data_folder)
    yield "data read<br>"
    rm.filter(minvotes=500, inplace=True)
    yield "filtered minvotes 500<br>"
    cor = rm._remove_multiple_mids()
    cor.__next__()
    with open(ignore_file, "r") as f:
        for movie in f.readlines():
            moviename = movie[:-1]
            if moviename in rm._mid:
                cor.send(rm._mid[moviename])
    cor.close()
    yield "filtered ignore movies<br>"
    model = RecSys(rm)
    model.train(verbose=False)
    yield "model trained<br>"
    trained = TrainedModel(model)
    trained.pickle("model.pkl")
    yield "done, new model will be active after reboot"


@app.route("/_retrain")
def retrain():
    return Response(retrain_gen())
