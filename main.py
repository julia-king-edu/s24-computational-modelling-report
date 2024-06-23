# import block

from mesa import Agent, Model
from mesa.space import SingleGrid
import numpy as np
from scipy import stats
import math
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import json

# define agents

class SSTAgent(Agent):
    """Agents for SST Model
    """
    
    # class variables
    
    # class methods
    def fit_beta(vector, max_sum: int) -> dict:
        """given a vector of values, finds the integer alpha and beta parameters resulting in the best fitting beta distribution where alpha + beta < max_sum

        Args:
            vector (array-like): vector of observed data
            max_sum (int): max of alpha + beta

        Raises:
            ValueError: if max_sum < 3

        Returns:
            dict: contains "alpha" and "beta"
        """
        if (max_sum < 3):
            raise ValueError("Max sum must be > 2")

        # initialize
        best_alpha = None
        best_beta = None
        best_log_likelihood = -np.inf
        # find best values
        for alpha in range(1, max_sum): # produces including max_sum - 1, excludes max_sum to ensure beta will never be 0
            for beta in range(1, max_sum - alpha + 1): # produces 1-12 (including) for alpha = 8 and max_sum = 20
                log_likelihood = stats.beta.logpdf(vector, alpha, beta).sum()
                if log_likelihood > best_log_likelihood:
                    best_alpha = alpha
                    best_beta = beta
                    best_log_likelihood = log_likelihood
        return {"alpha" : best_alpha, "beta" : best_beta}

    def min_disutility(private_alpha, private_beta, social_alpha, social_beta, gamma, w) -> dict:
        """
        Computes the expression that minimizes the total disutility.

        Args:
            private_alpha (int): Alpha parameter for distribution of private attitude.
            private_beta (int): Beta parameter for distribution of private attitude.
            social_alpha (int): Alpha parameter for perceived distribution of social norm.
            social_beta (int): Beta parameter for perceived distribution of social norm.
            gamma (float): Parameter for the disutility function.
            w (float): Weighting factor for social disutility.

        Returns:
            dict: Contains "expression" and "disutility".
        """
        possible_expressions = np.linspace(0.01, 0.99, 99)
        social_extremity = stats.beta.cdf(possible_expressions, social_alpha, social_beta)
        social_disutility = np.exp(gamma * (np.abs(0.5 - social_extremity) - 0.5))
        selfdistance = stats.beta.cdf(possible_expressions, private_alpha, private_beta)
        self_disutility = np.exp(gamma * (np.abs(0.5 - selfdistance) - 0.5))
        total_disutility = w * social_disutility + (1 - w) * self_disutility

        min_index = np.argmin(total_disutility)
        return {"expression": possible_expressions[min_index], "disutility": total_disutility[min_index]}
    
    # constructor
    def __init__(self, unique_id, model, pos, private_mean, w, gamma):
        super().__init__(unique_id, model)
        self.private = private_mean
        self.private_alpha = int(private_mean * 100)
        self.private_beta = 100 - self.private_alpha
        self.pos = pos
        self.w = w
        self.gamma = gamma
        
        # reduce private alpha and beta to lowest possible values
        private_gcd = math.gcd(self.private_alpha, self.private_beta)
        self.private_alpha = self.private_alpha // private_gcd
        self.private_beta = self.private_beta // private_gcd
        
        # set public attitude to mean of private attitude (default value at beginning of simulation)
        self.expressed = private_mean
    
    # object methods
    def perceive_neighbors(self):
        """first part of each round, agents perceive the expressed positions of their neighbors
        """
        neighbors_expressed = []
        for neighbor in self.model.grid.iter_neighbors(self.pos, moore = True):
            neighbors_expressed.append(neighbor.expressed)
            
        # find best alpha and beta that are integers and where alpha + beta <= 20
        best_dist = SSTAgent.fit_beta(neighbors_expressed, 20)
        self.social_alpha = best_dist["alpha"]
        self.social_beta = best_dist["beta"]
    
    def update_expressed(self):
        """second part of each round, agents compute their utility-maximizing attitude and update their expressed attitude to that.
        """
        result = SSTAgent.min_disutility(self.private_alpha, self.private_beta, self.social_alpha, self.social_beta, self.gamma, self.w)
        self.expressed = result["expression"]
        self.disutility = result["disutility"]
        

# define model

class SSTModel(Model):
    """
    Implements SST model.
    """
    
    # class variables
    # var
    
    # class methods
    # generate by typing def
    
    # constructor
    def __init__(self, width, height, w, gamma, percentile_steps = 0.05):
        super().__init__()
        self.grid = SingleGrid(width, height, torus = True)
        self.stepcount = -2 # starts at -2 because 2 rounds of calibration are necessary
        self.n_agents = height * width
        self.percentile_steps = int(percentile_steps * 100)
        
        # Create agents
        self.agent_list = []
        for y in range(height):
            for x in range(width):
                new_agent = SSTAgent(x + y * width, self, pos = (x, y), private_mean = round(np.random.beta(10,10), 2), w = w, gamma = gamma) # 100 possible private attitudes, matches their design
                # Add the agent to cell
                self.grid.place_agent(new_agent, (x, y))
                self.agent_list.append(new_agent)
        
        self.expressed_states = []
        
    
    # object methods
    def __expressed_list__(self) -> list:
        """Helper function that creates a list of the agents' currently expressed opinions. This is called before all usages of the list instead of once per step to ensure accuracy.

        Returns:
            list: a list of the agents' currently expressed opinions
        """
        return sorted([agent.expressed for agent in self.agent_list])
    
    def __variance__(self) -> float:
        """Returns the variance in currently expressed opinions. The higher the variance, the higher the segregation.

        Returns:
            float: variance in currently expressed opinions
        """
        return np.var(self.__expressed_list__())
    
    def __polarization_percentiles__(self) -> list:
        """Returns the expressed opinions in percentiles. Starts with 0 + percentile_steps and ends before 1. percentile_steps can be specified at init.

        Returns:
            list: percentile list.
        """
        expressed_list = self.__expressed_list__()
        percentiles = np.percentile(expressed_list, range(0 + self.percentile_steps, 100, self.percentile_steps))
        return percentiles
    
    def __polarization_total__(self) -> float:
        """Computes total polarization since initialization of the model using current_variance / initial_variance

        Returns:
            float: total polarization since initialization
        """
        return self.__variance__() / self.variance[0]
    
    def __swap_agents__(self, n_swaps: int = 1, max_attempts: int = 1000000) -> bool:
        """
        Attempts to swap two agents if it reduces disutility for both.

        Args:
            max_attempts (int): Maximum number of attempts to find a pair to swap.

        Returns:
            bool: True if a successful swap occurred, False otherwise.
        """
        swaps = 0
        for i in range(max_attempts):
            a1, a2 = random.sample(self.agent_list, 2)
            a1_disutility = SSTAgent.min_disutility(a1.private_alpha, a1.private_beta, a2.social_alpha, a2.social_beta, a1.gamma, a1.w)["disutility"]
            a2_disutility = SSTAgent.min_disutility(a2.private_alpha, a2.private_beta, a1.social_alpha, a1.social_beta, a2.gamma, a2.w)["disutility"]
            if (a1_disutility < a1.disutility) and (a2_disutility < a2.disutility):
                self.grid.swap_pos(a1, a2)
                a1.perceive_neighbors()
                a2.perceive_neighbors()
                a1.update_expressed()
                a2.update_expressed()
                swaps += 1
                print(f"Swap {swaps} done.")
            if (swaps >= n_swaps):
                return True
        return False
    
    def __update_summary__(self):
        # generate docstring by typing """
        self.variance.append(self.__variance__())
        self.pol_total.append(self.__polarization_total__())
        self.pol_percentiles.append(self.__polarization_percentiles__())
        self.disutility.append(sum([agent.disutility for agent in self.agent_list]))
    
    def summary(self) -> list:
        # generate docstring by typing """
        totals = pd.DataFrame({"step" : range(0, self.stepcount + 1), "variance" : self.variance, "polarization" : self.pol_total, "disutility" : self.disutility})
        pol_percentiles = pd.DataFrame(self.pol_percentiles, columns = [f"p_{i}" for i in range(0 + self.percentile_steps, 100, self.percentile_steps)])
        return {"totals" : totals, "pol_percentiles" : pol_percentiles}
    
    def __calibrate__(self, parallel : bool = False):
        """Executes steps of the model until calibrated.
        """
        while (self.stepcount < 0):
            self.stepcount += 1
            print(f"Starting calibration round {self.stepcount}.")
            # agents compute perceived social norm and adjust expressed attitude
            if (parallel):
                with ThreadPoolExecutor() as executor:
                    list(executor.map(lambda agent: agent.perceive_neighbors(), self.agent_list))
                    list(executor.map(lambda agent: agent.update_expressed(), self.agent_list))
            else:
                for agent in self.agent_list:
                    agent.perceive_neighbors()
                for agent in self.agent_list:
                    agent.update_expressed()
            print(f"Calibration round {self.stepcount} done.")
        # compute initial variance (i.e. segregation) and polarization
        self.variance = [self.__variance__()]
        self.pol_total = [1]
        self.pol_percentiles = [self.__polarization_percentiles__()]
        self.disutility = [sum([agent.disutility for agent in self.agent_list])]
        
    def save_state(self):
        """Saves the expressed attitudes and coordinates of each agent, as well as the summary statistics."""
        # current expressed attitudes
        state = [{"x": agent.pos[0], "y": agent.pos[1], "expressed": agent.expressed} for agent in self.agent_list]
        state = {"step": self.stepcount, "state": state}
        self.expressed_states.append(state)
        with open("paper/code/replication/out/expressed.json", 'w') as f:
            json.dump(self.expressed_states, f)
        # summary statistics
        summary = self.summary()
        summary["totals"].to_csv("paper/code/replication/out/totals.csv", index = False)
        summary["pol_percentiles"].to_csv("paper/code/replication/out/pol_percentiles.csv", index = True)
    
    def step(self, parallel : bool = False):
        """Executes one step of the model"""
        if self.stepcount < 0:
            self.__calibrate__(parallel = parallel)
        self.stepcount += 1

        start = time.time()
        # parallel execution of perceiving neighbors and updating expressed method
        if (parallel):
            with ThreadPoolExecutor() as executor:
                list(executor.map(lambda agent: agent.perceive_neighbors(), self.agent_list))
                list(executor.map(lambda agent: agent.update_expressed(), self.agent_list))
        else:
            for agent in self.agent_list:
                agent.perceive_neighbors()
            for agent in self.agent_list:
                agent.update_expressed()          
        print(f"Time for perceive_neighbors and update_expressed {time.time() - start}")

        start = time.time()
        # Agent swaps
        if not self.__swap_agents__(n_swaps=10, max_attempts=10000):
            print("Not all pairs were swapped.")
        print(f"Time for swaps {time.time() - start}")

        # Compute summary statistics
        self.__update_summary__()
        print(f"Step {self.stepcount} done.")




# run model
model = SSTModel(30, 30, w = 0.5, gamma = 20)
expressed = []
for i in range(5000):
    model.step(parallel = True)
    state = model.summary()
    print(state["totals"])
    print(state["pol_percentiles"])
    if (i % 20) == 0:
        # export
        model.save_state()