import numpy as np
from copy import deepcopy
import csv
from tqdm import tqdm
import multiprocessing
from functools import partial


'''
Description:
    - the One-Step-Optimal-Search slgorithm optimized for parallel computing.
    - All components that aren't broken are assumed to be new, this might need to be adjusted later down the line.

    - Request the right resources on your supercomputer cluster! 
    - It is advisable to write a simple bash script that starts multiple instances of this program in order to brute force the entire state
      space much quicker. 

    - Start with the following command line args: (file_name_for_results) (start_state) (end_state)
      example: python3 OSAS_Policy.py osas_data 0 100
      Note that this will do states 0 to 99 and exclude 100! This is meant for parallalization so you can easily start multiple
      instances of this program with different state ranges to calculate. Use different file names every time and merge them at the end 
     (e.g. osas100 (for states 0 to 99) then osas200 (for states 100 to 199) etc.)
'''

# Fetching environment object from parent directory
import sys
from Environment_package.environment_generator import environment, bit_val, bits

# Defining the hyperparameters for the run
env = environment(n_components=18, n_repair_crews=2, solution_cost_file='System/states_solution_full.csv')
gamma = 0.7 # How should future costs be taken into account (1 is full addition of future cost, 0 is only the immediate cost will be considered)
max_steps = 3 # Define the max steps the agent can take in the environment here
sample_size = 1 # The sample size determines how many times each action will be evaluated. The final action will then be chosen based on the average cost each action incurred. 


# Fetching command line arguments
run_name = str(sys.argv[1])
start_state = int(sys.argv[2])
end_state = int(sys.argv[3])

# Creating a csv file for the run
try: 
    with open(f'data/osas/{run_name}.csv', 'x', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['State', 'Best Action Cost', 'Best Action Reward']
        csv_writer.writerow(header)
except FileExistsError:
    pass

def calculate_action_value(action: list[int], state: list[int]):
    '''Takes an action and samples it to find the average action value
    Args:
        state  (list): current state in ints
        action (list): action with ints 
        gamma (float): float value of gamma used
        
    Returns: 
        action_values (tuple): ((mean_cost), (mean_reward))
        '''

    # Tracking reward and cost to make sure they create the same policy as they should...
    sample_vzero = []
    sample_vzero_reward = [] #

    # Creating a sample for each action, because environment is non-deterministic
    for sample in range(sample_size):

        # First step in the environment with our predetermined action
        new_state,cost,reward,done = env.state_transition(state, action)
        vzero = cost
        vzero_reward = reward #

        # Playing the episode through with no action taken at each step and discounted reward and cost logged 
        for exp in range(1,max_steps):

            new_state, cost, reward, done = env.state_transition(new_state, [0 for i in range(env.n_components)])

            # Defining cut-off point where no further steps are taking since change to policy is minor. This helps to save resources
            if (gamma ** exp) < 0.00001:
                break
            else:
                vzero += (gamma ** exp) * cost
                vzero_reward += (gamma ** exp) * reward
            # vzero += (gamma ** exp) * cost
            # vzero_reward += (gamma ** exp) * reward
        sample_vzero.append(vzero)
        sample_vzero_reward.append(vzero_reward)

    return (np.mean(sample_vzero), np.mean(sample_vzero_reward))


def state_range_iteration(start_state: int, end_state: int) -> None:
    """Takes a state range and saves the the best action for each state in a file. [start_state, end_state]
    Args:
        state (list): a list containing the current state of the system
    
        """
    for state in tqdm(range(start_state, end_state), unit='State'):
        state = bits(state, env.n_components)
        state = list(state)
        for component in range(len(state)):
            state[component] = int(state[component])

        # Adding initial failure rate to state, this might need adjusting, since all components that arent broken are assumed to be new here.
        state = np.append(state, [env.initial_failure_rate for x in range(env.n_components)])
        state = deepcopy(state)

        bit_val_state = bit_val(state[:env.n_components])

        if __name__ == '__main__':
            num_cores = multiprocessing.cpu_count()
            # pool = multiprocessing.Pool(processes=num_cores)
            pool = multiprocessing.Pool(processes=1)

            calculate_action_value_fixed_state = partial(calculate_action_value, state=state)
            results = pool.map(calculate_action_value_fixed_state, env.action_space)
            print(results)

            pool.close()
            pool.join()
            
            # Unpack results
            action_costs = [result[0] for result in results]
            action_reward = [result[1] for result in results]

            # plt.hist(sample_vzero, bins=20)
            # plt.savefig('test.png')
            #print(np.argmin(action_costs))
            #print(action_costs)

            # Looking for actions with the same cost or reward value
            min_cost_index = np.argmin(action_costs)
            min_cost = action_costs[min_cost_index]
            best_action_cost = []
            action = 0
            for cost in action_costs:
                if min_cost == cost:
                    best_action_cost.append(action)
                action +=1
            
            max_reward_index = np.argmax(action_reward)
            max_reward = action_reward[max_reward_index]
            best_actions_reward = []
            action = 0
            for reward in action_reward:
                if max_reward == reward:
                    best_actions_reward.append(action)
                action += 1

            with open(f'data/osas/{run_name}.csv', 'a+', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([bit_val_state, best_action_cost, best_actions_reward])


if __name__ == '__main__':
    state_range_iteration(start_state, end_state)

