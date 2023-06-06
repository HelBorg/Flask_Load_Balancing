import logging

import numpy as np
import pandas as pd
from mpi4py import MPI

from appfc.load_balancing.protocol import create_agent, local_voting, acc_local_voting, generate_queue, admm
from appfc.services import PARAMETERS, ALGORITHM

NOISE_GENERATION = ["None", "St. normal distr.", "Custom"]

def start():
    comm = MPI.Comm.Get_parent()
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    if local_rank == 0:
        logging.basicConfig(filename=f'cache/loggs/_loggs_{local_rank}.log', filemode='w', level=logging.INFO)

    logging.info("MPI start point")
    input_info = comm.bcast(None, root=0)
    alg = input_info.get(ALGORITHM, None)
    parameters = input_info.get(PARAMETERS, None)
    steps = parameters["steps"]
    gen_queue = parameters["gen_queue"]
    weights = np.array(parameters["weights"])
    noise = parameters["noise"]

    noise_function = lambda t: 0
    if noise == "Custom":
        noise_function = lambda t: eval(parameters["custom_noise"])
    elif noise == "St. normal distr.":
        noise_function = lambda t: np.random.normal()

    # reset local seed
    np.random.seed()

    # Generate a common graph (everyone use the same seed)
    Adj = weights.copy()

    Adj[Adj != 0] = 1
    queue = generate_queue(local_rank, num_steps=steps, generate=gen_queue)
    agent = create_agent(Adj, weights, queue, is_logging=True)

    if local_rank == 0:
        logging.info(f"Running {alg} with parameters {parameters}")
        logging.info(f"Connectivity matrix: {Adj}")

    sequence = None
    if alg == "LVP":
        sequence, _ = local_voting(agent, parameters, noise_function, num_iterations=steps)
    elif alg == "ALVP":
        sequence, _ = acc_local_voting(agent, parameters, noise_function, num_iterations=steps)
    # elif alg == "ADMM":
    #     sequence, _ = admm(agent, parameters, noise_function, num_iterations=steps)

    np.save(f"cache/agent_{local_rank}_sequence_{alg}.npy", sequence)

    comm.Disconnect()

start()
