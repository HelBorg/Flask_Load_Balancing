import logging
import os
import pickle
import shutil

import numpy as np
from flask import render_template, flash

from appfc import app
from appfc.errors import ParameterRequiredException
from appfc.forms import RequestForm, FieldLizt
from appfc.services import run_algorithms

PLOTS_PATH = 'appfc/templates/html_files/'
NAME_MAP = {
    "LVP_agent_dynamic.html": "LVP_Dynamic",
    "ALVP_agent_dynamic.html": "ALVP_Dynamic",
    "error_comparison.html": "Error_comparison"
}


@app.route("/", methods=["GET", "POST"])
def home():
    form = RequestForm()
    weights = form.matrix
    refresh_matrix = compare_with_previous(form)
    plots = []
    submit = True

    num_agents = form.num.data
    if "Custom" == form.matr.data and refresh_matrix:
        form.matrix_generation(num_agents, lambda *args: 0)
        submit = False

    elif "Default" == form.matr.data and refresh_matrix:
        form.matrix_generation(num_agents, default_matrix_value)
        submit = False

    if submit and form.validate_on_submit():
        plots = on_submit(form, weights)

    with open("cache/form.pickle", "wb") as file:
        pickle.dump(form.data, file)

    return render_template(
        "algorithms.html",
        title='Simulation',
        form=form,
        plots=plots,
    )


def compare_with_previous(form):
    if not os.path.isfile('cache/form.pickle'):
        return
    with open('cache/form.pickle', 'rb+') as file:
        previous_form = pickle.load(file)
    data = form.data
    diff = [key for key, entry_1 in data.items() if key in previous_form and entry_1 != previous_form[key]]
    refresh_matrix = "num" in diff or "matr" in diff

    matr = form.matrix.data
    if "matrix" in diff:
        form.change_field("matr", "Custom")

    return refresh_matrix


def on_submit(form, weights):
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
    run_algorithms(data, weights.data)

    plots = {}
    for file in os.listdir(PLOTS_PATH):
        with open(PLOTS_PATH + file, "r+", encoding="utf-8") as f:
            plot_label = NAME_MAP[file]
            plots[plot_label] = f.read()
    logging.info(plots.keys())
    return plots


def default_matrix_value(i, j, num_agents):
    half_agents = num_agents // 2
    return 0.5 if j == (i + half_agents) % num_agents or j == (i + half_agents + 1) % num_agents else 0
