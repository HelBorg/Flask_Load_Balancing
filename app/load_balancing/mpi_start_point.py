import numpy as np
from mpi4py import MPI

from app.load_balancing.protocol import create_agent, local_voting, acc_local_voting
from app.routes import PARAMETERS, ALGORITHM


def start():
    comm = MPI.Comm.Get_parent()
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    input_info = comm.bcast(None, root=0)
    alg = input_info.get(ALGORITHM, None)
    parameters = input_info.get(PARAMETERS, None)

    # Generate a common graph (everyone use the same seed)
    Adj = np.zeros((nproc, nproc))
    for i in range(nproc):
        Adj[i, [(i + 2) % nproc, (i + 3) % nproc]] = 1
    agents = create_agent(Adj)

    if alg == "LVP":
        sequence, _ = local_voting(agents, parameters)
        np.save("cache/agent_{}_sequence_lvp.npy".format(local_rank), sequence)

    elif alg == "ALVP":
        sequence, _ = acc_local_voting(agents, parameters)
        np.save("cache/agent_{}_sequence_alvp.npy".format(local_rank), sequence)

    np.save(f"cache/ok_{local_rank}.py", sequence)

    comm.Disconnect()


start()
