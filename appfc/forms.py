from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectMultipleField, widgets, FloatField, IntegerField, FieldList, \
    SelectField, FormField, Form, StringField, RadioField
from wtforms.validators import DataRequired, optional
from wtforms.widgets import TextArea

from appfc.errors import ParameterRequiredException


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class FieldLizt(Form):
    fiel = FieldList(
        FloatField('Matrix', default=0),
        min_entries=2,
        max_entries=20)


class RequestForm(FlaskForm):
    ALGORITMS = ["LVP", "ALVP"]
    ALGO_PARAMETERS = {
        "LVP": ["h"],
        "ALVP": ["L", "h", "alpha", "gamma", "mu", "eta"]
    }
    MATRIX_GENERATION = ["Default", "Custom"]
    NOISE_GENERATION = ["None", "St. normal distr.", "Custom"]

    num = IntegerField('Number of agents', validators=[DataRequired()])
    steps = IntegerField('Steps', validators=[DataRequired()])
    matr = SelectField("Matrix Generation", choices=MATRIX_GENERATION)
    matrix = FieldList(
        FormField(FieldLizt),
        min_entries=2,
        max_entries=20)
    noise = RadioField("Noise Generation", choices=NOISE_GENERATION)
    custom_noise = StringField("Custom noise function", widget=TextArea())
    algs = MultiCheckboxField("Algorithms", choices=ALGORITMS)
    submit = SubmitField('Submit')

    def validate_required_parameters(self):
        for alg in self.algs.data:
            self.check_parameters(alg)

    def check_parameters(self, alg):
        for param in self.ALGO_PARAMETERS[alg]:
            if not self.data[alg + "_" + param]:
                raise ParameterRequiredException(alg + "_" + param)

    def matrix_generation(self, num_agents, value):
        matrix = self.matrix
        if not matrix or not num_agents:
            return None
        matrix.entries = []
        for i in range(len(matrix), num_agents):
            matrix_form = FieldLizt()
            matrix_form.fiel = [value(i, j, num_agents) for j in range(num_agents)]
            matrix.append_entry(matrix_form)

    def change_field(self, field, value):
        self[field].data = value
        self[field].raw_data = None


for alg, items in RequestForm.ALGO_PARAMETERS.items():
    for param in items:
        setattr(RequestForm, f"{alg}_{param}", FloatField(param, validators=[optional()]))
