from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectMultipleField, widgets, FloatField
from wtforms.validators import DataRequired, optional

from app.errors import ParameterRequiredException


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class RequestForm(FlaskForm):
    algorithms = ["LVP", "ALVP"]
    algo_parameters = {
        "LVP": ["h"],
        "ALVP": ["L", "h", "alpha_0", "gamma_0"]
    }

    num = StringField('Number of agents', validators=[DataRequired()])
    algs = MultiCheckboxField("Algorithms", choices=algorithms)
    submit = SubmitField('Submit')

    def validate_required_parameters(self):
        for alg in self.algs.data:
            self.check_parameters(alg)

    def check_parameters(self, alg):
        for param in RequestForm.algo_parameters[alg]:
            if not self.data[alg + "_" + param]:
                raise ParameterRequiredException(alg + "_" + param)


for alg, items in RequestForm.algo_parameters.items():
    for param in items:
        setattr(RequestForm, f"{alg}_{param}", FloatField(param, validators=[optional()]))
