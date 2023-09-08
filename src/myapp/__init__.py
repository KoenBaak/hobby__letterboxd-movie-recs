from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap

from recsys import TrainedModel

# -------------globals-----------------------------------------------------------
APP_NAME = "dummy name"
ILLEGAL_GENRES = set(["Documentary"])
SQLALCHEMY_DATABASE_URI = "sqlite:///data.db"
MODEL = TrainedModel.read_json("model.json")
# -------------------------------------------------------------------------------

app = Flask(__name__)

# app configuration
app.config["SECRET_KEY"] = "you-will-never-guess"
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# end configuration

db = SQLAlchemy(app)
bootstrap = Bootstrap(app)

from myapp import db_models, routes
