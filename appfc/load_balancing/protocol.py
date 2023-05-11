import datetime
import logging

import numpy as np
import pandas as pd
from mpi4py import MPI

from lvp.local_voting import LocalVoting, AgentLB, AcceleratedLocalVoting


def create_agent(connections, W, queue, is_logging=False):
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()
    logging.info(f"Agent {local_rank} is being created")

    if is_logging:
        logging.basicConfig(filename=f'cache/loggs/_loggs_{local_rank}.log', filemode='w', level=logging.INFO)

    np.random.seed()

    logging.info(f"Needed {connections}")
    np.save("cache/agent_{}_sequence_lvp_conn.npy".format(local_rank), connections)
    # create local agent
    agent = AgentLB(queue=queue,
                    produc=5,
                    in_neighbors=np.nonzero(connections[local_rank, :])[0].tolist(),
                    out_neighbors=np.nonzero(connections[:, local_rank])[0].tolist(),
                    in_weights=W[local_rank, :].tolist())
    return agent


def local_voting(agent, parameters, noise_function, num_iterations=100):
    # instantiate the consensus algorithm
    algorithm = LocalVoting(
        gamma=parameters["h"],
        agent=agent,
        initial_condition=np.array([0]),
        noise_function=noise_function,
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=num_iterations)
    return sequence, algorithm


def acc_local_voting(agent, parameters, noise_function, num_iterations=100):
    # instantiate the consensus algorithm
    if "gamma" in parameters and isinstance(parameters["gamma"], float):
        parameters["gamma"] = [parameters["gamma"]]

    algorithm = AcceleratedLocalVoting(
        parameters=parameters,
        agent=agent,
        initial_condition=np.array([0]),
        noise_function=noise_function,
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=num_iterations, verbose=True)
    return sequence, algorithm

def acc_local_voting(agent, parameters, noise_function, num_iterations=100):
    # instantiate the consensus algorithm
    if "gamma" in parameters and isinstance(parameters["gamma"], float):
        parameters["gamma"] = [parameters["gamma"]]

    algorithm = AcceleratedLocalVoting(
        parameters=parameters,
        agent=agent,
        initial_condition=np.array([0]),
        noise_function=noise_function,
        enable_log=True)  # enable storing of the generated sequences

    # run the algorithm
    sequence = algorithm.run(iterations=num_iterations, verbose=True)
    return sequence, algorithm


def generate_queue(local_rank, num_steps, generate=True):
    if generate:
        size = np.random.poisson(lam=num_steps//2, size=1)[0] + 1
        if local_rank == 5:
            size = 100
        queue_raw = [[np.random.randint(num_steps), np.random.poisson(10)] for i in range(size)]
        add = [[0, np.random.poisson(lam=local_rank*10 + 1, size=1)[0]] for i in range(size)]
        queue_raw = np.append(queue_raw, add, axis=0)
        queue = pd.DataFrame(queue_raw, columns=["time", "complexity"])
        queue.to_csv(f"cache/agent_{local_rank}_queue.csv", index=False)
    else:
        queue = pd.read_csv(f"cache/agent_{local_rank}_queue.csv")
    return queue

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    if local_rank == 0:
        print(f"Time: {datetime.datetime.now()}")

    # Generate a common graph (everyone use the same seed)
    Adj = np.zeros((nproc, nproc))
    for i in range(nproc):
        Adj[i, [(i + 2) % nproc, (i + 3) % nproc]] = 1

    agents = create_agent(Adj, is_logging=True)
    sequence, algorithm = local_voting(agents, {"h": 0.2})

    parameters = {
        "L": 3,
        "mu": 1,
        "h": 0.3,
        "eta": 0.8,
        "gamma": [0.1],
        "alpha": 0.15
    }
    sequence, algorithm = acc_local_voting(agents, parameters)

    # print solution
    print("Agent {}: {}".format(local_rank, algorithm.get_result()))

    # save data
    np.save("cache/agents.npy", nproc)
    np.save("cache/agent_{}_sequence_lvp.npy".format(local_rank), sequence)
    logging.warning(sequence)
