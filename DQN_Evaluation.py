from DQN_Versions.DQNV1 import tensoragentv1
# Add other DQN Versions here

import os
import pickle
import json
import numpy as np
from Environment_package.environment_generator import environment, environment_surprise_attack

"""Descriptions:
- This file allows for the evaluation of trained DQN policies. 
- Start the evaluation by editing the agent_name and eval_mode variable. 
- Determine how many episodes you want to evaluate on. 
=> Results will be automatically stored in data/{Policy_Name}/{agent_name} in json and pkl files
=> For the graphical analysis of the results functions are provided in the support.py file
"""

# Specify the name of the DQN Agent that is to be evaluated
agent_name='V1012'
# Specify the scenario (available scenarios: normal, surprise, attack)
eval_mode = 'surprise'
# Specify number of episodes to evaluate the agent on 
episodes = 100

storage_folder = f'data/{agent_name[:2]}/{agent_name}'
if not os.path.exists(storage_folder):
    os.mkdir(storage_folder)


env = environment(n_components=18, n_repair_crews=2,solution_cost_file='System/states_solution_full.csv', cost_of_replacement=1, attrition_rate=0.005, initial_failure_rate=0.005)
# env = environment(n_components=18, n_repair_crews=2,solution_cost_file='System/states_solution_full.csv', cost_of_replacement=1)

agent = tensoragentv1(agent_name, env, storage_folder=storage_folder)

print(agent.model.summary())
if eval_mode == 'normal':
    running_cost, running_reward, n_failed_arc, n_repaired_arc, preemptive_repair_data, states_visited, broken_pair_matrix = agent.evaluate(episodes=episodes)
    save_name_ind = ''
elif eval_mode == 'surprise':
    running_cost, running_reward, n_failed_arc, n_repaired_arc, preemptive_repair_data, states_visited, broken_pair_matrix = agent.evaluate_surprise_attrition((0, 360), 0.01, episodes=episodes)
    save_name_ind = '_surpriseattrition'
elif eval_mode == 'attack':
    env = environment_surprise_attack(n_components=18, n_repair_crews=2, solution_cost_file='System/Alderson_2015_modified/states_solution_full_V2.csv')
    # List with all starting states with 1 to 18 broken components and the 10 worst states each
    starting_states = [2, 4, 8, 32, 64, 128, 256, 512, 1024, 4096, 
        40, 272, 768, 1536, 8196, 8224, 16416, 16512, 20480, 81920,
        352, 1664, 2688, 5632, 16417, 16418, 16424, 16516, 24704, 131112,
        1569, 1570, 1576, 2600, 16428, 16496, 16736, 24616, 65888, 131424,
        1580, 1648, 1888, 2912, 9768, 16737, 16738, 16740, 16744, 16752,
        1889, 1890, 1892, 1896, 1904, 2016, 3936, 5984, 10080, 11104,
        10081, 10082, 10084, 10088, 10096, 10208, 12128, 14176, 26464, 42848,
        10083, 10085, 10086, 10089, 10090, 10092, 10097, 10098, 10100, 10104,
        10087, 10091, 10093, 10094, 10099, 10101, 10102, 10105, 10106, 10108,
        10095, 10103, 10107, 10109, 10110, 10215, 10219, 10221, 10222, 10227,
        10111, 10223, 10231, 10235, 10237, 10238, 12143, 12151, 12155, 12157,
        10239, 12159, 12271, 12279, 12283, 12285, 12286, 14207, 14319, 14327,
        12287, 14335, 16255, 16367, 16375, 16379, 16381, 16382, 26623, 28543,
        16383, 28671, 30719, 32639, 32751, 32759, 32763, 32765, 32766, 45055,
        32767, 49151, 61439, 63487, 65407, 65519, 65527, 65531, 65533, 65534,
        65535, 98303, 114687, 126975, 129023, 130943, 131055, 131063, 131067, 131069,
        131071, 196607, 229375, 245759, 258047, 260095, 262015, 262127, 262135, 262139, 
        262143]
    running_cost, running_reward = agent.evaluate_surprise_attack(starting_states)
    save_name_ind = '_surpriseattack'
else:
    raise ValueError('Select a valid evaluation mode')

# Saving evaluation results straight from function
with open(storage_folder, 'w') as f:
    json.dump(running_cost, f)
with open(f'{storage_folder}/{agent_name}_surpriseattrition_evaluation_reward.json', 'w') as f:
    json.dump(running_reward, f)
if eval_mode in ['normal', 'surprise']:
    with open(f'{storage_folder}/{agent_name}{save_name_ind}_component_data.json', 'w') as f:
        json.dump(n_failed_arc, f)
        f.write('\n')
        json.dump(n_repaired_arc, f)
        f.write('\n')
        json.dump(preemptive_repair_data, f)
    # storing the dict data as a pickle
    with open(f'{storage_folder}/{agent_name}{save_name_ind}_states_visited_data.pkl', 'wb') as f:
        pickle.dump(states_visited, f)
    # saving the broken arc matrix as a np file
    np.save(f'{storage_folder}/{agent_name}{save_name_ind}_broken_pair_matrix.npy', broken_pair_matrix)


print(f'Done\nCost per Step: {np.mean(running_cost)}')
print(f'results saved at: {storage_folder}')
