import numpy as np
import json
# Fetching environment object from parent directory
import sys
import os

sys.path.append('../masterthesisgitcode')
from Environment_package.environment_generator import environment, environment_surprise_attack, bits

# Description:
#   - Policy is to fix everything that is broken
#   - when more components are broken that repair crews are available it will choose random components 



def normal_attrition_eval(episodes=100, max_steps=360):

    env = environment(18, 'System/Alderson_2015_modified/states_solution_full_V2.csv', cost_of_replacement=1, attrition_rate=0.005, initial_failure_rate=0.005)

    initial_state = [0 for i in range(env.n_components)]
    initial_state += [env.initial_failure_rate for i in range(env.n_components)]

    # Setting up variables to keep track of data
    running_reward = []
    running_cost = []

    running_repaired_arcs = []
    running_failed_arcs = []
    running_preemptive_repair_data = []

    for episode in range(episodes):

        # Initializing storage varibles to keep track of what arcs are repaired and failed during an episode
        n_repaired_arcs = [int(0) for x in range(env.n_components)]
        n_failed_arcs = [int(0) for x in range(env.n_components)]
        preemptive_repair_data = [[0,0] for x in range(env.n_components)] # each embedded list is for one component, the 0th element is the number of 
        # repairs, when the component was broken, the 1th element is the number of preemptive repairs when the component was not broken

        episode_reward = []
        episode_cost = []

        # Define initial state
        state = initial_state
        for step in range(max_steps):

            # compile action according to broken elements in state
            # First check how many components are broken
            num_broken = 0
            index_of_broken_components = []
            for indexcomp, indicator in enumerate(state[:env.n_components]):
                if indicator == 1:
                    num_broken += 1
                    index_of_broken_components.append(indexcomp)

            action = []
            action_counter = 0 # Making sure that action doesnt contain more fixes than allowed
            if num_broken <= env.n_repair_crews:
                for indicator in state[:env.n_components]:
                    
                        if indicator == 1:
                            if action_counter < env.n_repair_crews:
                                action.append(1)
                                action_counter += 1
                            else:
                                action.append(0)  
                        else:
                            action.append(0)
            else:
                choicea = np.random.choice(index_of_broken_components, env.n_repair_crews, replace=False)
                action = [0 for i in range(env.n_components)]
                for indexl in choicea:
                    action[indexl] = 1
            
            # Check for every component if it was preemptively repaired or not
            for i in range(env.n_components):
                if state[i] == 1 and action[i] == 1:
                    preemptive_repair_data[i][0] += 1
                if state[i] == 0 and action[i] == 1:
                    preemptive_repair_data[i][1] += 1

            # Apply the chosen action to our environment
            next_state, cost, reward, done = env.state_transition(state, action)
            episode_reward.append(reward)
            episode_cost.append(cost)
            state = next_state

            # Save the data on what arcs got repaired and what arcs are broken
            for i in range(env.n_components):
                n_repaired_arcs[i] += action[i]
                n_failed_arcs[i] += state[i]

        # Save the episode data in a temporary variable to later calculate the average over all episodes
        running_failed_arcs.append(n_failed_arcs)
        running_repaired_arcs.append(n_repaired_arcs)
        running_preemptive_repair_data.append(list(preemptive_repair_data))      

        mean_failed_arcs = []
        mean_repaired_arcs = []
        mean_preemptive_repair_data = []

        # Save the average for all collected data in the variables 
        for i in range(env.n_components):
                arc_failures = []
                arc_repairs = []
                preemptive_repairs_per_arc = []
                for element in running_failed_arcs:
                    arc_failures.append(element[i])
                for element in running_repaired_arcs:
                    arc_repairs.append(element[i])

                # getting the mean for each arc for the preemptive repair vs failure repair data
                for element in running_preemptive_repair_data:
                    # creates a list that contains all tuples from each episode for this arc
                    preemptive_repairs_per_arc.append(element[i])
                preemptive_repair = []
                failure_repair = []

                # creating two lists, one contains all the preemptive repair values over all episode for this arc and the other contains the failure_repair values over all episodes
                for element in preemptive_repairs_per_arc:
                    preemptive_repair.append(element[0])
                    failure_repair.append(element[1])

                mean_failed_arcs.append(np.mean(arc_failures))
                mean_repaired_arcs.append(np.mean(arc_repairs))
                # creating the tuple for this arc and saving it from the data 
                mean_preemptive_repair_data.append((np.mean(preemptive_repair), np.mean(failure_repair)))

        
        running_reward.append(np.mean(episode_reward))
        running_cost.append(np.mean(episode_cost))

    # Saving evaluation results straight from function
    folder_path = f'data/custom_data/policy1'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    with open(f'data/custom_data/policy1/policy1_evaluation_cost.json', 'w') as f:
        json.dump(running_cost, f)
    with open(f'data/custom_data/policy1/policy1_evaluation_reward.json', 'w') as f:
        json.dump(running_reward, f)
    with open(f'data/custom_data/policy1/policy1_component_data.json', 'w') as f:
        json.dump(mean_failed_arcs, f)
        f.write('\n')
        json.dump(mean_repaired_arcs, f)
        f.write('\n')
        json.dump(mean_preemptive_repair_data, f)
    
    print(f'Done\nAverage Cost Per Step: {np.mean(running_cost)}')
    print(f'results saved at: /data/custom_data/policy1')

def evaluate_surprise_attrition(increase_attrition_for_steps: tuple, increase_attrition_to: float, episodes: int=100) -> tuple[list, list]:

    # Defining Hyperparameters
    env_surpriseattrition = environment(18, 'System/Alderson_2015_modified/states_solution_full_V2.csv', initial_failure_rate=0.005, attrition_rate=0.005, cost_of_replacement=0.1)
    episodes = 100
    max_steps = 360

    # Defining initial state
    initial_state = [0 for i in range(env_surpriseattrition.n_components)]
    initial_state += [env_surpriseattrition.initial_failure_rate for i in range(env_surpriseattrition.n_components)]

    # Setting up variables to keep track of data
    running_reward = []
    running_cost = []

    running_repaired_arcs = []
    running_failed_arcs = []
    running_preemptive_repair_data = []

    for episode in range(episodes):
        
        # Initializing storage varibles to keep track of what arcs are repaired and failed during an episode
        n_repaired_arcs = [int(0) for x in range(env_surpriseattrition.n_components)]
        n_failed_arcs = [int(0) for x in range(env_surpriseattrition.n_components)]
        preemptive_repair_data = [[0,0] for x in range(env_surpriseattrition.n_components)] # each embedded list is for one component, the 0th element is the number of 
        # repairs, when the component was broken, the 1th element is the number of preemptive repairs when the component was not broken
        episode_reward = []
        episode_cost = []

        # Define initial state
        state = initial_state
        for step in range(max_steps):

            if increase_attrition_for_steps[1] >= step >= increase_attrition_for_steps[0]:
                env_surpriseattrition.attrition_rate=increase_attrition_to

            # compile action according to broken elements in state
            # First check how many components are broken
            num_broken = 0
            index_of_broken_components = []
            for indexcomp, indicator in enumerate(state[:env_surpriseattrition.n_components]):
                if indicator == 1:
                    num_broken += 1
                    index_of_broken_components.append(indexcomp)

            action = []
            action_counter = 0 # Making sure that action doesnt contain more fixes than allowed
            if num_broken <= env_surpriseattrition.n_repair_crews:
                for indicator in state[:env_surpriseattrition.n_components]:
                    
                        if indicator == 1:
                            if action_counter < env_surpriseattrition.n_repair_crews:
                                action.append(1)
                                action_counter += 1
                            else:
                                action.append(0)  
                        else:
                            action.append(0)
            else:
                choicea = np.random.choice(index_of_broken_components, env_surpriseattrition.n_repair_crews, replace=False)
                action = [0 for i in range(env_surpriseattrition.n_components)]
                for indexl in choicea:
                    action[indexl] = 1



            # Check for every component if it was preemptively repaired or not
            for i in range(env_surpriseattrition.n_components):
                if state[i] == 1 and action[i] == 1:
                    preemptive_repair_data[i][0] += 1
                if state[i] == 0 and action[i] == 1:
                    preemptive_repair_data[i][1] += 1

            # Apply the chosen action to our environment
            next_state, cost, reward, done = env_surpriseattrition.state_transition(state, action)
            episode_reward.append(reward)
            episode_cost.append(cost)
            state = next_state

            # Save the data on what arcs got repaired and what arcs are broken
            for i in range(env_surpriseattrition.n_components):
                n_repaired_arcs[i] += action[i]
                n_failed_arcs[i] += state[i] 

        # Save the episode data in a temporary variable to later calculate the average over all episodes
        running_failed_arcs.append(n_failed_arcs)
        running_repaired_arcs.append(n_repaired_arcs)
        running_preemptive_repair_data.append(list(preemptive_repair_data))      

        mean_failed_arcs = []
        mean_repaired_arcs = []
        mean_preemptive_repair_data = []

        # Save the average for all collected data in the variables 
        for i in range(env_surpriseattrition.n_components):
                arc_failures = []
                arc_repairs = []
                preemptive_repairs_per_arc = []
                for element in running_failed_arcs:
                    arc_failures.append(element[i])
                for element in running_repaired_arcs:
                    arc_repairs.append(element[i])

                # getting the mean for each arc for the preemptive repair vs failure repair data
                for element in running_preemptive_repair_data:
                    # creates a list that contains all tuples from each episode for this arc
                    preemptive_repairs_per_arc.append(element[i])
                preemptive_repair = []
                failure_repair = []

                # creating two lists, one contains all the preemptive repair values over all episode for this arc and the other contains the failure_repair values over all episodes
                for element in preemptive_repairs_per_arc:
                    preemptive_repair.append(element[0])
                    failure_repair.append(element[1])

                mean_failed_arcs.append(np.mean(arc_failures))
                mean_repaired_arcs.append(np.mean(arc_repairs))
                # creating the tuple for this arc and saving it from the data 
                mean_preemptive_repair_data.append((np.mean(preemptive_repair), np.mean(failure_repair)))

        
        running_reward.append(np.mean(episode_reward))
        running_cost.append(np.mean(episode_cost))

    # Saving evaluation results straight from function
    folder_path = f'data/custom_data/policy1'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Saving evaluation results straight from function
    with open(f'data/custom_data/policy1/policy1_surpriseattrition_evaluation_cost.json', 'w') as f:
        json.dump(running_cost, f)
    with open(f'data/custom_data/policy1/policy1_surpriseattrition_evaluation_reward.json', 'w') as f:
        json.dump(running_reward, f)
    with open(f'data/custom_data/policy1/policy1_surpriseattrition_component_data.json', 'w') as f:
        json.dump(mean_failed_arcs, f)
        f.write('\n')
        json.dump(mean_repaired_arcs, f)
        f.write('\n')
        json.dump(mean_preemptive_repair_data, f)
    
    print(f'Done\nAverage Cost Per Step: {np.mean(running_cost)}')
    print(f'results saved at: /data/custom_data/policy1')


def surprise_attack_eval(states):
    '''Evaluates the policy on a broken system and stops the episode, when system is fully fixed'''

    env = environment_surprise_attack(18, 'System/Alderson_2015_modified/states_solution_full_V2.csv', initial_failure_rate=0, attrition_rate=0, cost_of_replacement=1)
    episodes = 100
    max_steps = 360
    running_reward = []
    running_cost = []

    for x in range(len(states)):
        episode_reward = []
        episode_cost = []

        # Initialize the starting state by picking a state from the given state list
        state = list(bits(states[x], env.n_components))
        counter = 0
        for element in state:
            state[counter] = int(element)
            counter += 1
        state += [env.initial_failure_rate for i in range(env.n_components)]

        for step in range(max_steps):

            # compile action according to broken elements in state
            action = []
            action_counter = 0 # Making sure that action doesnt contain more fixes than allowed
            for indicator in state[:env.n_components]:
                
                    if indicator == 1:
                        if action_counter < env.n_repair_crews:
                            action.append(1)
                            action_counter += 1
                        else:
                            action.append(0)  
                    else:
                        action.append(0)
            
            # Apply the chosen action to our environment
            next_state, cost, reward, done = env.state_transition(state, action)
            episode_reward.append(reward)
            episode_cost.append(cost)
            state = next_state

            # Break loop if done
            if done:
                break

        running_reward.append(np.mean(episode_reward))
        running_cost.append(np.sum(episode_cost))

    # Saving evaluation results straight from function
    with open(f'data/custom_data/policy1/policy1_surpriseattack_evaluation_cost.json', 'w') as f:
        json.dump(running_cost, f)
    with open(f'data/custom_data/policy1/policy1_surpriseattack_evaluation_reward.json', 'w') as f:
        json.dump(running_reward, f)
    
    print(f'Done\nAverage Cost Per Step: {np.mean(running_cost)}')
    print(f'results saved at: /data/custom_data/policy1')

# List with all starting states with 1 to 18 broken components and the 5 worst states each
starting_states = [4, 16, 8192, 16384, 32768, 2064, 16388, 16512, 20480, 81920,
                   16424, 17424, 18448, 67600, 132112,
                   17428, 18449, 18450, 18452, 18456,
                   18453, 18454, 18460, 18484, 18516,
                   18196, 19220, 26645, 26646, 26652,
                   19221, 19222, 19228, 19252, 19284,
                   27413, 27414, 27420, 27444, 27476,
                   27415, 27421, 27422, 27445, 27446,
                   27423, 27447, 27453, 27454, 27479,
                   27455, 27487, 27511, 27517, 27518,
                   27519, 27583, 27615, 27639, 27645,
                   27647, 28543, 28607, 28639, 28663,
                   28671, 31743, 32639, 32703, 32735,
                   32767, 61439, 64511, 65407, 65471,
                   65535, 98303, 126975, 130047, 130943,
                   131071, 196607, 229375, 258047, 261119,262143]
        
surprise_attack_eval(starting_states)
# evaluate_surprise_attrition((0, 360), 0.01)
# normal_attrition_eval(episodes=100)



