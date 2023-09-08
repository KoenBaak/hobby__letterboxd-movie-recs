from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import ValidationError, DataRequired


class UsernameForm(FlaskForm):
    username = StringField("Letterboxd Username", validators=[DataRequired()])
    # submit = SubmitField()

    def validate_username(self, username):
        if not all(
            x in "abcdefghijklmnopqrstuvwxyz0123456789_" for x in username.data.lower()
        ):
            raise ValidationError(
                "Username may only contain upper of lower case letters, numbers and underscores."
            )
        if len(username.data) > 15:
            raise ValidationError("Username too long.")
