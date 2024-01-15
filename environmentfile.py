import itertools
import numpy as np
import csv
from copy import deepcopy
import random
import tensorflow as tf
from tqdm import tqdm

'''
Implements the environment for the agent to interact with.

1. Creates the environment with action and states space
2. Handles the transition from one state to another
3. Returns the reward based on the action taken and the state of the system after action is applied to it!
    This means the reward is based of the action the agent took and the cost of running the system in the NEXT STATE

'''

class environment:
    """Defines the environment"""

    def __init__(self,n_components=18,solution_cost_file='System/Alderson_2015_modified/states_solution_full_V2.csv', initial_failure_rate=0.005,n_repair_crews=2, attrition_rate=0.005,cost_of_replacement=1):
        self.n_components = n_components
        self.initial_failure_rate = initial_failure_rate
        self.n_repair_crews = n_repair_crews
        self.attrition_rate = attrition_rate
        self.cost_of_replacement = cost_of_replacement
        self.current_state = [0 for i in range(n_components*2)]
        self.action_space = generate_action_space(n_components,n_repair_crews)
        self.state_space = generate_state_space(n_components)
        self.state_cost = load_csv_to_dict(f'{solution_cost_file}')
        self.reward_range = find_max_and_min_reward(self.state_cost, n_repair_crews, cost_of_replacement)
    
    def state_transition(self, state: list[int], action: list[int]) -> tuple[list[int], float, float, bool]:
        """Takes a state and action and returns the next state, cost and reward. 0 is a working component and 1 is broken
        Args:
            state  (list): list of values for each component
            action (list): list that represents a vector of the action taken

        Returns: 
            next state (list) : the next state
            cost       (float): cost of the action and system state after actions are accounted for
            reward     (float): the reward of the agent scaled between 0 and 1
            done       (bool) : done flag, default FALSE
        """
        
        # Creating a copy of the state to avoid overwriting the originial variable
        state = deepcopy(state)

        cost = 0

        # Updating every single component status indicator
        for idx in range(self.n_components):
            
            # If component is UP and action is NOACTION
            if state[idx] == 0 and action[idx] == 0:
                if np.random.rand() < state[self.n_components + idx]:
                    state[idx] = 1
                else:
                    state[self.n_components + idx] += self.attrition_rate
                
            # If component is UP and action is REPLACE
            if state[idx] == 0 and action[idx] == 1:
                cost += self.cost_of_replacement
                state[self.n_components + idx] = self.initial_failure_rate
            
            # If component is DOWN and action is REPLACE
            if state[idx] == 1 and action[idx] == 1:
                cost += self.cost_of_replacement
                state[idx] = 0
                state[self.n_components + idx] = self.initial_failure_rate
            
            # If component is DOWN and action is NOACTION nothing changes

        # Calculates cost of the system after action is applied to state
        cost += self.state_cost[bit_val(state[:self.n_components])]

        assert cost > 0, 'costs can not be smaller or equal to 0... fatal math error'

        # Calculating reward and scaling it to be in range (0,1)
        max_reward = self.reward_range[0]
        min_reward = self.reward_range[1]
        reward = 100 * (1 / cost)
        reward = (reward - min_reward) / (max_reward - min_reward)
        
        # Setting done condition (NOT USED)
        done = False
        new_state = state

        return new_state, cost, reward, done
    
class environment_surprise_attack:
    """Defines the environment, passes done flag, when all components are fixed. Returns reward 1 when all components are fixed"""

    def __init__(self,n_components=18,solution_cost_file='System/Alderson_2015_modified/states_solution_full_V2.csv', initial_failure_rate=0,n_repair_crews=2, attrition_rate=0,cost_of_replacement=1):
        self.n_components = n_components
        self.initial_failure_rate = initial_failure_rate
        self.n_repair_crews = n_repair_crews
        self.attrition_rate = attrition_rate
        self.cost_of_replacement = cost_of_replacement
        self.current_state = [0 for i in range(n_components*2)]
        self.action_space = generate_action_space(n_components,n_repair_crews)
        self.state_space = generate_state_space(n_components)
        self.state_cost = load_csv_to_dict(f'{solution_cost_file}')
        self.reward_range = find_max_and_min_reward(self.state_cost, n_repair_crews, cost_of_replacement)
    
    def state_transition(self, state: list[int], action: list[int]) -> tuple[list[int], float, float, bool]:
        """Takes a state and action and returns the next state, cost and reward. 0 is a working component and 1 is broken
        Args:
            state  (list): list of values for each component
            action (list): list that represents a vector of the action taken

        Returns: 
            next state (list) : the next state
            cost       (float): cost of the action and system state after actions are accounted for
            reward     (float): the reward of the agent scaled between 0 and 1
            done       (bool) : done flag, default FALSE
        """
        
        # Creating a copy of the state to avoid overwriting the originial variable
        state = deepcopy(state)

        cost = 0

        # Updating every single component status indicator
        for idx in range(self.n_components):
            
            # If component is UP and action is NOACTION => Let component fail with failure probability of this component, if not failed raise failure probability by attrition rate
            if state[idx] == 0 and action[idx] == 0:
                if np.random.rand() < state[self.n_components + idx]:
                    state[idx] = 1
                else:
                    state[self.n_components + idx] += self.attrition_rate
                
            # If component is UP and action is REPLACE => Add the replace cost; reset failure rate of component
            if state[idx] == 0 and action[idx] == 1:
                cost += self.cost_of_replacement
                state[self.n_components + idx] = self.initial_failure_rate
            
            # If component is DOWN and action is REPLACE => Add replace cost; set component state to UP again; reset failure rate of component
            if state[idx] == 1 and action[idx] == 1:
                cost += self.cost_of_replacement
                state[idx] = 0
                state[self.n_components + idx] = self.initial_failure_rate
            
            # If component is DOWN and action is NOACTION nothing changes => No change in anything

        # Calculates cost of the system after action is applied to state
        cost += self.state_cost[bit_val(state[:self.n_components])]

        assert cost > 0, 'costs can not be smaller or equal to 0... fatal math error'

        # Calculating reward and scaling it to be in range (0,1)
        max_reward = self.reward_range[0]
        min_reward = self.reward_range[1]
        reward = 100 * (1 / cost)
        reward = (reward - min_reward) / (max_reward - min_reward)

        new_state = state
        
        # Setting done condition if all components are fixed
        done = True
        for indicator in new_state[:self.n_components]:
            if indicator == 1:
                done = False
                break
        
        # Returning high reward when Agent triggers done flag
        if done:
            reward += 1 

        return new_state, cost, reward, done

    


def find_max_and_min_reward(state_solution: dict[int, float], n_repair_crews: int, cost_of_replacement: float) -> tuple[float, float]:
    """Calculates the max and min possible reward of any given environment
    Args:
        state_solution      (csv)  : csv file that contains the base10 number of any state and the corresponding cost
        n_repair_crews      (int)  : max number of possible REPAIR actions taken at the same time
        cost_of_replacement (float): cost of the REPLACE action
    
    Returns: 
        max_reward (float): the max possible reward for the agent
        min_reward (float): the min possible reward for the agent

    """

    # finds min and max value of the costs from all possible states
    values = state_solution.values()
    numeric_list = []
    for value in values:
        numeric_list.append(float(value))
    min_cost = min(numeric_list)
    # adds the max action cost to the max state cost
    max_cost = max(numeric_list) + (n_repair_crews * cost_of_replacement)

    # scales the reward accordingly
    max_reward = 100 * (1 / min_cost)
    min_reward = 100 * (1 / max_cost)

    return max_reward, min_reward


def generate_action_space(n_components: int, n_repair_crews: int) -> np.ndarray:
    """Generates all possible action for a system
    Args:
        n_components (int): number of components of the system.
        n_repair_crews (int): number of repair crews that determines the max number of REPLACE action in any given timestep.
    
    Returns:
        np.array: matrix of all possible actions.
        """
    
    # Checks for impossible combinations
    if n_repair_crews < 0 or n_components < n_repair_crews:
        raise ValueError("Number of components must be greater than or equal to the number of repair crews.")
    
    # Generate all possible actions based on the number of max simultainious REPAIR actions
    combinations = list(itertools.combinations(range(n_components), n_repair_crews))
    while n_repair_crews >= 1:
        combinations += (itertools.combinations(range(n_components), n_repair_crews -1))
        n_repair_crews -= 1
    
    # Creating numpy array to store all actions
    matrix = np.zeros((len(combinations), n_components), dtype=int)
    for i, combination in enumerate(combinations):
        for idx in combination:
            matrix[i, idx] = 1
    
    return matrix


def generate_state_space(n_components: int) -> np.ndarray:
    """Generates all possible binary combinations of compoment status but neglects component attrition
    Args:
        n_components (int): number of components in the system
    
    Returns:
        np.ndarray: matrix of all possible component status
    """
    if n_components < 1:
        raise ValueError("Number of elements must be greater than 0.")
    
    # Generate all possible binary combinations of length n
    combinations = list(itertools.product([0, 1], repeat=n_components))
    # Convert the list of tuples to a numpy array
    matrix = np.array(combinations, dtype=int)
    
    return matrix

def load_csv_to_dict(file_path: str) -> dict[int, float]:
    """loads a csv file into a dictionary
    Args:
        file_path (str): path to the state solution file
    
    Returns: 
        dict[int, float]: returns a dictionary with all states(int) as keys and all costs(float) as values
        """
    result_dict = {}
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Skipping the first row manually with a flag in the file, since it contains the headers
        next(csv_reader)

        for row in csv_reader:
            if len(row) == 2:  # Ensure there are exactly two elements in the row
                key, value = row
                key = key.replace('.', ',')
                result_dict[int(key)] = float(value)
            else: 
                raise IndexError("There have to be exactly two columns with keys and values. Something is wrong with your csv file")

    return result_dict


def return_random_state_with_n_broken_components(env, state_n_broken_components: int, n_worst_states: int=5):
    """Takes a number of broken components to return the worsts states to
    Args:
        env                 (class): environment class
        n_broken_components (int)  : number of broken components in the state
        n_worst_states      (int)  : top x worst states for specified number of broken components. Only one gets randomly selected and returned
        
    Returns:
        initial_state (list): state"""
    
    # Finding the number of broken components in each state
    n_broken_components = {num: [] for num in range(env.n_components + 1)}

    for state in env.state_cost:
        bit_state = bits(state, env.n_components)
        counter = bit_state.count("1")
        n_broken_components[counter].append(state)
    
    # Choosing the worst states for each number of component broken, as specified in the initialization
    counter = 0
    for value in n_broken_components.values():

        state_cost_combination = {state: float for state in value}
        for state in value: 
            state_cost_combination[state] = env.state_cost[state]
        
        highest_values = sorted(state_cost_combination.values(), reverse=True)[:n_worst_states]
        keys_with_highest_values = [key for key, value in state_cost_combination.items() if value in highest_values]
        keys_with_highest_values = keys_with_highest_values[:n_worst_states]

        n_broken_components[counter] = keys_with_highest_values
        counter += 1

    possible_states = n_broken_components[state_n_broken_components]
    initial_state = bits(random.choice(possible_states), env.n_components)
    initial_state = list(initial_state)
    counter = 0
    for element in initial_state:
        initial_state[counter] = int(element)
        counter += 1

    return initial_state


def evaluate(self, episodes: int=100) -> tuple[list, list]:
    """Evaluates the agent for the specified amount of episodes
    
    Args: 
        self             (object): the agent object
        evaluation_episodes (int): number of episodes the agent is evaluated on
        
    Returns:
        running_cost   (list): average cost per episode of the system
        running_reward (list): average reward per episode of the system
    """

    # Setting storage variables
    running_cost = []
    running_reward = []

    for episode in range(episodes):
        state = [0 for i in range(self.env.n_components)]
        state = state +  [self.env.initial_failure_rate for x in range(self.env.n_components)]

        episode_reward = []
        episode_cost = []

        for step in range(self.max_steps):

            # Choose the action, based on the agent best guess
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            actionidx = tf.argmax(action_probs[0]).numpy()
            action = self.env.action_space[actionidx]

            # Apply the sampled action to our environment and storing cost and reward
            next_state, cost, reward, done = self.env.state_transition(state, action)
            episode_cost.append(cost)
            episode_reward.append(reward)
            state = next_state

        running_cost.append(np.mean(episode_cost))
        running_reward.append(np.mean(episode_reward))
        

    return running_cost, running_reward


def evaluate_surprise_attrition(self, increase_attrition_for_steps: tuple, increase_attrition_to: float, episodes: int=100) -> tuple[list, list]:
    """Evaluates the agent for the specified amount of episodes
    
    Args: 
        self                        (object): the agent object
        episodes                       (int): number of episodes the agent is evaluated on
        increase_attrition_for_steps (tuple): specifies the amount of steps for which the attrition rate will be increased in the environment 
                                                e.g. (100, 200) to increase rate between step 100 and 200
        increase_attrition_to        (float): new attrition rate for the duration of the increased rate
        
    Returns:
        running_cost   (list): average cost per episode of the system
        running_reward (list): average reward per episode of the system
    """

    # Setting storage variables
    running_cost = []
    running_reward = []

    for episode in tqdm(range(episodes), desc='Evaluation', unit='episode'):
        state = [0 for i in range(self.env.n_components)]
        state = state +  [self.env.initial_failure_rate for x in range(self.env.n_components)]

        episode_reward = []
        episode_cost = []

        for step in range(self.max_steps):

            if increase_attrition_for_steps[1] >= step >= increase_attrition_for_steps[0]:
                self.env.attrition_rate=increase_attrition_to

            # Choose the action, based on the agent best guess
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            actionidx = tf.argmax(action_probs[0]).numpy()
            action = self.env.action_space[actionidx]

            # Apply the sampled action to our environment and storing cost and reward
            next_state, cost, reward, done = self.env.state_transition(state, action)
            episode_cost.append(cost)
            episode_reward.append(reward)
            state = next_state

        running_cost.append(np.mean(episode_cost))
        running_reward.append(np.mean(episode_reward))
        

    return running_cost, running_reward



# The following code is from Dr. ALDERSON and therefore not my own

def bits(a,b):
  '''Given an integer (a) and a number of bits (b), return the bitstring'''
  res=bin(a+2**b)
  if a<0:
    return res[2:]
  else:
    return res[3:]

def bit_val(s):
    '''Given a string of 0-1 chars ('bits'), returns decimal value.'''
    val = 0
    e = 0
    while e < len(s):
        val += int(s[-1-e]) * 2**e
        e += 1
    return val
