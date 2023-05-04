import logging
import os
import shutil

from flask import render_template, flash
from werkzeug.datastructures import MultiDict

from app import app
from app.errors import ParameterRequiredException
from app.forms import RequestForm, MatrixForm
from app.services import run_algorithms

PLOTS_PATH = 'app/templates/html_files/'
NAME_MAP = {
    "LVP_agent_dynamic.html": "LVP_Dynamic",
    "ALVP_agent_dynamic.html": "ALVP_Dynamic",
    "error_comparison.html": "Error_comparison"
}


@app.route("/", methods=["GET", "POST"])
def home():
    form = RequestForm()
    matrix = MatrixForm()
    plots = []
    submit = True

    if form.num.data and form.num.data != len(matrix.matrix.data):
        num_agents = form.num.data
        data = [[(f"matrix-{i}-field-{j}", 1) for j in range(num_agents)] for i in range(num_agents)]
        data = [item for sublist in data for item in sublist]
        matrix = MatrixForm(MultiDict(data))
        submit = False

    if submit and form.validate_on_submit():
        print("Building")
        plots = on_submit(form)

    return render_template(
        "algorithms.html",
        title='Simulation',
        form=form,
        matrix=matrix,
        plots=plots,
    )


def on_submit(form):
    try:
        form.validate_required_parameters()
    except ParameterRequiredException as ex:
        flash(ex.message)
        return "", "", ""

    if not os.listdir(PLOTS_PATH):
        logging.info("Deleting files")
        shutil.rmtree(PLOTS_PATH, ignore_errors=True)
        os.mkdir(PLOTS_PATH)

    data = form.data
    run_algorithms(data)

    plots = {}
    for file in os.listdir(PLOTS_PATH):
        with open(PLOTS_PATH + file, "r+", encoding="utf-8") as f:
            plot_label = NAME_MAP[file]
            plots[plot_label] = f.read()
    logging.info(plots.keys())
    return plots
