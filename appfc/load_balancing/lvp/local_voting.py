import logging

import numpy as np
import pandas as pd
from disropt.agents import Agent
from disropt.algorithms import Consensus, ADMM
from disropt.functions import Variable, QuadraticForm
from disropt.problems import Problem
from mpi4py import MPI
from pandas import DataFrame


class AgentLB(Agent):
    def __init__(self, produc, queue: DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.produc = produc
        self.queue = queue.sort_values("time").reset_index(drop=True)

    def neighbor_send(self, obj, out_neighbors):
        """
        Send data to concrete neighbors
        :param obj:
        :param out_neighbors:
        :return:
        """
        self.communicator.neighbors_send(obj, out_neighbors)

    def update_value(self, old_value, new_value, step):
        self.rearrange_tasks(np.floor(old_value - new_value), step)
        self.execute_tasks(step)

    def get_queue_length(self, step):
        """
        Get queue complexity
        :param step:
        :return:
        """
        return sum(self.get_queue(step).complexity)

    def get_queue(self, step):
        """
        Get queue aat current step
        :return:
        """
        # logging.warning(self.queue[self.queue.time <= step])
        return self.queue[self.queue.time <= step]

    def rearrange_tasks(self, x, step):
        """
        Change number of tasks by x by sending or recieving
        :param x: increase by
        :param step: number of current step
        :return:
        """
        if x == 0:
            self.neighbors_exchange(0)
            self.neighbors_exchange(0)
            return
        elif x < 0:
            self.receive_tasks(-x)
        else:
            self.send_tasks(x, step)

    def send_tasks(self, x, step):
        """
        Send x tasks
        :param x:
        :param step:
        :return:
        """
        # get neibors who vote to receive
        neib_info = self.neighbors_exchange(0)
        if len(neib_info) == 0:
            return

        queue = self.get_queue(step).sort_values("complexity")

        # send tasks
        to_send = {}
        for key, complex in neib_info.items():
            if complex == 0 or x < 0 or queue.shape[0] == 0:
                to_send[key] = pd.DataFrame()
                continue

            # extract tasks to send
            num_tasks = 0
            if abs(complex) > abs(x):
                complex = x

            while complex > 0 and queue.shape[0] > num_tasks:
                complex -= queue.iloc[num_tasks].complexity
                num_tasks += 1

            x -= sum(queue.iloc[:num_tasks].complexity)
            send = queue.iloc[:num_tasks]
            self.queue = self.queue[~self.queue.index.isin(send.index)]
            # send tasks
            to_send[key] = send

        try:
            self.neighbors_exchange(to_send if to_send else 0, dict_neigh=True)
        except BaseException:
            print("Oh mg exception")
            print(f"Agent {self.id} send {to_send}")
            print(self.agent.in_weights)

        self.queue = self.queue.reset_index(drop=True).sort_values("time")

    def receive_tasks(self, x):
        """
        Tell neibors that can get x tasks and receive tasks from them
        :param x: number of tasks to receive
        :return:
        """
        self.neighbors_exchange(x)
        res = self.neighbors_exchange(0)

        for key, val in res.items():
            if not isinstance(val, pd.DataFrame):
                continue

            self.queue = pd.concat([self.queue.iloc[:1], val, self.queue.iloc[1:]])
        self.queue = self.queue.reset_index(drop=True).sort_values("time")

    def execute_tasks(self, step):
        """
        Execute tasks: remove first tasks in the queue with respect to productivity
        :return:
        """
        execute = self.produc
        while execute != 0:
            if self.get_queue_length(step) == 0:
                break

            first_task = self.queue.iloc[0, 1]
            if first_task > execute:
                self.queue.iloc[0, 1] = first_task - execute
                break
            else:
                execute -= first_task
                self.queue = self.queue.iloc[1:]

    def update_problem(self, step: int = 0):
        comm = MPI.Comm.Get_parent()
        nproc = comm.Get_size()
        local_rank = comm.Get_rank()

        queue = self.agent.get_queue_length(step)
        neib_queue = self.neighbors_exchange(queue)
        num_neib_tasks = queue + sum(neib_queue.values())
        print(f"Agent {local_rank}: num_tasks min - {queue}, with neighbors - {num_neib_tasks}")

        n = 1
        # define the local objective function
        P = np.eye(n)
        x = Variable(n)
        fn = QuadraticForm(x, P)

        # define a (common) constraint set
        A = np.eye(n)
        f = A @ x - num_neib_tasks
        constr = f == 0

        # local problem
        problem = Problem(fn, constr)

        self.set_problem(problem)


class ConsensusLB(Consensus):
    def __init__(
            self,
            agent: AgentLB,
            initial_condition: np.ndarray,
            noise_function,
            enable_log: bool = False):
        super(ConsensusLB, self).__init__(agent=agent,
                                          initial_condition=initial_condition,
                                          enable_log=enable_log)
        self.noise_function = noise_function

    def iterate_run(self, step, **kwargs):
        """Run a single iterate of the algorithm
        :param step: current step
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh] + self.noise_function(step)
        x_avg = self.one_step()
        self.agent.update_value(self.x, x_avg, step)

    def one_step(self):
        """
        Implement one algorithm step calculations
        :param step: number of current step
        :return: average value of the queue
        """
        logging.info("I am here 3")
        pass

    def run(self, iterations: int = 100, verbose: bool = False, **kwargs):
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations + 1]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.x = self.agent.get_queue_length(k)

            if self.enable_log:
                self.sequence[k] = self.x

            if k % 25 == 0:
                print(f"Agent {self.agent.id}: x = {self.x} step = {k}")

            self.iterate_run(k, **kwargs)

        if self.enable_log:
            self.x = self.agent.get_queue_length(k + 1)
            self.sequence[k + 1] = self.x

        if self.enable_log:
            return self.sequence


class LocalVoting(ConsensusLB):

    def __init__(
            self,
            gamma,
            agent: AgentLB,
            initial_condition: np.ndarray,
            noise_function,
            enable_log: bool = False):
        super(LocalVoting, self).__init__(agent=agent,
                                          initial_condition=initial_condition,
                                          enable_log=enable_log,
                                          noise_function=noise_function)
        self.gamma = gamma

    def one_step(self):
        return self.x - self.gamma * sum([(self.x - self.x_neigh[i]) for i in self.agent.in_neighbors])


class AccelerateParameters:
    mu = float
    L: float
    gamma: []
    h: float
    eta: float
    a: float

    def from_dict(self, d):
        self.__dict__.update(d)
        return self


class AcceleratedLocalVoting(ConsensusLB):
    def __init__(self,
                 parameters: dict,
                 agent: AgentLB,
                 initial_condition: np.ndarray,
                 noise_function,
                 enable_log: bool = False):
        super(AcceleratedLocalVoting, self).__init__(agent=agent,
                                                     initial_condition=initial_condition,
                                                     enable_log=enable_log,
                                                     noise_function=noise_function)
        self.nesterov_step = 0

        self.L = parameters.get("L")
        self.mu = parameters.get("mu")
        self.h = parameters.get("h")
        self.eta = parameters.get("eta")
        self.gamma = parameters.get("gamma", [])
        self.alpha = parameters.get("alpha")

    def one_step(self):
        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))
        x_n = 1 / (self.gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * self.gamma[0] * self.nesterov_step + self.gamma[1] * self.x)

        y_n = sum([self.agent.in_weights[i] * (x_n - self.x_neigh[i]) for i in self.agent.in_neighbors])
        x_avg = x_n - self.h * y_n

        self.nesterov_step = 1 / self.gamma[0] * \
                             ((1 - self.alpha) * self.gamma[0] * self.nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * y_n)

        H = self.h - self.h * self.h * self.L / 2
        if H - self.alpha * self.alpha / (2 * self.gamma[1]) < 0:
            logging.warning(H)
            logging.exception(f"Oh no: {H - self.alpha * self.alpha / (2 * self.gamma[1])}")
            logging.info("Exception")
            raise BaseException()

        return x_avg


class ADMM_LB(ADMM):
    def __init__(
            self,
            agent: AgentLB,
            initial_lambda: dict,
            initial_z: np.ndarray,
            enable_log: bool = False
    ):
        super(ADMM_LB, self).__init__(agent=agent,
                                      initial_lambda=initial_lambda,
                                      initial_z=initial_z,
                                      enable_log=enable_log)
        # self.noise_function = noise_function

    def run(self, iterations: int = 100, penalty: float = 0.1, verbose: bool = False, **kwargs) -> np.ndarray:
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")

        if self.enable_log:
            # initialize sequence of x and z
            x_dims = [iterations]
            for dim in self.x_shape:
                x_dims.append(dim)
            self.x_sequence = np.zeros(x_dims)
            self.z_sequence = np.zeros(x_dims)

            # initialize sequence of lambda
            self.lambda_sequence = {}
            for j in self.augmented_neighbors:
                self.lambda_sequence[j] = np.zeros(x_dims)

        self.initialize_algorithm()

        for k in range(iterations):
            # store current lambda and z
            if self.enable_log:
                for j in self.augmented_neighbors:
                    self.lambda_sequence[j][k] = self.lambd[j]
                self.z_sequence[k] = self.z

            self.iterate_run(rho=penalty, step=k, **kwargs)

            # store primal solution
            if self.enable_log:
                self.x_sequence[k] = self.x

            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return (self.x_sequence, self.lambda_sequence, self.z_sequence)

    def _update_local_solution(self, x: np.ndarray, z: np.ndarray, z_neigh: dict, rho: float, step: int, **kwargs):
        """Update the local solution

        Args:
            x: current solution
            z: current auxiliary primal variable
            z_neigh: auxiliary primal variables of neighbors (dictionary)
            rho: penalty parameter
        """
        self.z_neigh = z_neigh

        # update dual variables
        self.lambd[self.agent.id] += rho * (x - z)
        for j, z_j in z_neigh.items():
            self.lambd[j] += rho * (x - z_j)

        # update primal variables
        self.x = x
        self.z = z
        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        if local_rank == 0:
            print(f"X: {x}, z: {z}")

    def initialize_algorithm(self):
        """Initializes the algorithm
        """
        # exchange z with neighbors
        self.z_neigh = self.agent.neighbors_exchange(self.z)

    def iterate_run(self, rho: float, step: int, **kwargs):
        """Run a single iterate of the algorithm
        """

        # build local problem
        x_i = Variable(self.x_shape[0])

        penalties = [(x_i - self.z_neigh[j]) @ (x_i - self.z_neigh[j]) for j in self.agent.in_neighbors]
        penalties.append((x_i - self.z) @ (x_i - self.z))
        sumlambda = sum(self.lambd.values())

        obj_function = self.agent.problem.objective_function + sumlambda @ x_i + (rho / 2) * sum(penalties)
        pb = Problem(obj_function, self.agent.problem.constraints)

        # solve problem and save data
        x = pb.solve()

        # exchange primal variables and dual variables with neighbors
        x_neigh = self.agent.neighbors_exchange(x)
        lambda_neigh = self.agent.neighbors_exchange(self.lambd, dict_neigh=True)

        # compute auxiliary variable
        z = (sum(x_neigh.values()) + x) / (self.degree + 1) + \
            (sum(lambda_neigh.values()) + self.lambd[self.agent.id]) / (rho * (self.degree + 1))

        # exchange auxiliary variables with neighbors
        z_neigh = self.agent.neighbors_exchange(z)

        # update local data
        self._update_local_solution(x, z, z_neigh, rho, step, **kwargs)

    def get_result(self):
        """Return the current value primal solution, dual variable and auxiliary primal variable

        Returns:
            tuple (primal, dual, auxiliary): value of primal solution (np.ndarray), dual variables (dictionary of np.ndarray), auxiliary primal variable (np.ndarray)
        """
        return (self.x, self.lambd, self.z)
