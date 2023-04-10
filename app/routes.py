import codecs
import sys
import time

import numpy as np
from flask import render_template, flash
from mpi4py import MPI

from app import app
from app.errors import ParameterRequiredException
from app.forms import RequestForm
import logging

# logging.basicConfig(filename=f'cache/loggs/91_loggs.log', filemode='w', level=logging.INFO)


ALG_PATH = "app\load_balancing\mpi_start_point.py"


@app.route("/", methods=["GET", "POST"])
def home():
    form = RequestForm()
    plot, plot_2, plot_3 = "", "", ""
    print("Here")
    if form.validate_on_submit():
        plot, plot_2, plot_3 = on_submit(form)

    return render_template(
        "algorithms.html",
        title='Register',
        form=form,
        plots=[plot, plot_2, plot_3]
    )


def on_submit(form):
    try:
        form.validate_required_parameters()
    except ParameterRequiredException as ex:
        flash(ex.message)
        return "", "", ""
    print("OMG")
    data = form.data
    run_algorithms(data)

    plot = codecs.open("./app/templates/html_files/trash2.html", 'r').read()
    plot_2 = codecs.open("./app/templates/html_files/trash1.html", 'r').read()
    plot_3 = codecs.open("./app/templates/html_files/trash3.html", 'r').read()
    print("kkk")
    return plot, plot_2, plot_3


def run_algorithms(data):
    pass


if __name__ == "__main__":
    app.run(debug=False)
