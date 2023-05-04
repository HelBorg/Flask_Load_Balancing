import logging

import numpy as np
from mpi4py import MPI

from app.load_balancing.protocol import create_agent, local_voting, acc_local_voting
from app.services import PARAMETERS, ALGORITHM


def start():
    logging.info("MPI start point")
    comm = MPI.Comm.Get_parent()
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    input_info = comm.bcast(None, root=0)
    alg = input_info.get(ALGORITHM, None)
    parameters = input_info.get(PARAMETERS, None)
    steps = parameters["steps"]

    # reset local seed
    np.random.seed()

    # Generate a common graph (everyone use the same seed)
    Adj = np.zeros((nproc, nproc))
    for i in range(nproc):
        Adj[i, [(i + nproc//2) % nproc, (i + nproc//2 + 1) % nproc]] = 1
    agents = create_agent(Adj, num_steps=steps)

    if local_rank == 0:
        logging.info(f"Running {alg} with parameters {parameters}")
        logging.info(f"Connectivity matrix: {Adj}")

    sequence = None
    if alg == "LVP":
        sequence, _ = local_voting(agents, parameters, num_iterations=steps)
    elif alg == "ALVP":
        sequence, _ = acc_local_voting(agents, parameters, num_iterations=steps)

    np.save(f"cache/agent_{local_rank}_sequence_{alg}.npy", sequence)

    comm.Disconnect()


start()
