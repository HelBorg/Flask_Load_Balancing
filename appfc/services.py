import logging
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpi4py import MPI

ALG_PATH = "appfc/load_balancing/mpi_start_point.py"
PARAMETERS = "parameters"
ALGORITHM = "alg"


def run_algorithms(data, weights):
    num_agents = int(data["num"])
    logging.info("Starting the simulation")
    errors = {}
    parameters = {
        "steps": data["steps"],
        "noise": data["noise"],
        "noise_function": data.get("noise_function", None),
        "weights": extract_weights(weights)
    }
    for ind, algorithm in enumerate(data['algs']):
        parameters_alg = {
            key.split("_", 1)[-1]: value for key, value in data.items() if key.startswith(algorithm)
        }
        parameters_alg.update(parameters)
        parameters_alg["gen_queue"] = ind == 0

        set_up_subprocesses(algorithm, parameters_alg, num_agents)

        seq_data = create_sequence_plot(num_agents, algorithm)
        errors[algorithm] = compute_error(pd.DataFrame(seq_data), num_agents)
    logging.info("Creating error plot")
    create_error_plot(errors)


def extract_weights(weigths):
    result = []
    for dic in weigths:
        result.append(dic['fiel'])
    return result


def set_up_subprocesses(algorithm, parameters, num_agents):
    logging.info("Launching workers")
    ch_comm = MPI.COMM_SELF.Spawn(sys.executable, args=[ALG_PATH], maxprocs=num_agents)

    logging.info("Passing info")
    input_info = {ALGORITHM: algorithm, PARAMETERS: parameters}
    ch_comm.bcast(input_info, root=MPI.ROOT)

    logging.info("Closing subprocesses")
    ch_comm.Disconnect()
    time.sleep(0.25)


def compute_error(df, num):
    df["Mean"] = df.mean(axis=1)
    for i in range(num):
        df[f"{i}"] = np.sqrt(np.power(df[f"{i}"] - df[f"Mean"], 2))
    return sum([df[f"{i}"] for i in range(num)])


def create_sequence_plot(num_agents, algorithm):
    fig = go.Figure()
    seq_data = {}
    for k in range(num_agents):
        sequence = np.load(f"cache/agent_{k}_sequence_{algorithm}.npy")[:, 0]
        seq_data[f"{k}"] = sequence
        fig.add_trace(go.Scatter(y=sequence,
                                 x=list(range(len(sequence))),
                                 name=f"Agent {k}",
                                 mode="lines"))

        fig.update_layout(
            xaxis_title="t",
            yaxis_title="Queue length",
            font=dict(size=20)
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='white')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='white')

    fig.write_html(f"appfc/templates/html_files/{algorithm}_agent_dynamic.html")
    return seq_data


def create_error_plot(errors):
    logging.info(f"Get {errors}, now ploting")
    fig = go.Figure()
    for key, value in errors.items():
        fig.add_trace(go.Scatter(y=value,
                                 x=list(range(len(value))),
                                 name=key,
                                 mode="lines",
                                 line={'dash': 'solid'}))

    fig.update_layout(
        xaxis_title="t",
        yaxis_title="Error",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.9
        ),
        font=dict(size=20)
    )

    fig.write_html(f"appfc/templates/html_files/error_comparison.html")
