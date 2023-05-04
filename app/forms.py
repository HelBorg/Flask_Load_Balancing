from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectMultipleField, widgets, FloatField, IntegerField, FieldList, \
    FormField, Form
from wtforms.validators import DataRequired, optional

from app.errors import ParameterRequiredException


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class FieldLizt(Form):
    field = FieldList(
        FloatField('Matrix'),
        min_entries=2,
        max_entries=20)


class MatrixForm(FlaskForm):
    matrix = FieldList(
        FormField(FieldLizt),
        min_entries=2,
        max_entries=20)


class RequestForm(FlaskForm):
    ALGORITMS = ["LVP", "ALVP"]
    ALGO_PARAMETERS = {
        "LVP": ["h"],
        "ALVP": ["L", "h", "alpha_0", "gamma_0"]
    }
    num = IntegerField('Number of agents', validators=[DataRequired()])
    steps = IntegerField('Steps', validators=[DataRequired()])

    algs = MultiCheckboxField("Algorithms", choices=ALGORITMS)
    submit = SubmitField('Submit')

    def validate_required_parameters(self):
        for alg in self.algs.data:
            self.check_parameters(alg)

    def check_parameters(self, alg):
        for param in self.ALGO_PARAMETERS[alg]:
            if not self.data[alg + "_" + param]:
                raise ParameterRequiredException(alg + "_" + param)


for alg, items in RequestForm.ALGO_PARAMETERS.items():
    for param in items:
        setattr(RequestForm, f"{alg}_{param}", FloatField(param, validators=[optional()]))
