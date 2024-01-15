from DQN_Versions.DQNV1 import tensoragentv1
from Environment_package.environment_generator import environment
from support import create_figure
import sys
import json

# Fetching command line arguments
agent_name = str(sys.argv[1])
start_episode = int(sys.argv[2])
end_episode = int(sys.argv[3])
max_episodes = int(sys.argv[4])


# Define the storage folder of the data from the agent
storage_folder=f'data/V1/{agent_name}'
import os
# Make new dir if none currently exist 
os.makedirs(storage_folder, exist_ok=True)

# Define the environment with all hyperparameters
env = environment(18, n_repair_crews=2, solution_cost_file='System/states_solution_full.csv')
# Define the agent 
agent = tensoragentv1(agent_name=agent_name, environment=env, start_epsilon=0.9, max_episodes=max_episodes,storage_folder=storage_folder)

# Train the agent
agent.train(start_episode, end_episode)

# Evaluation of agent
evaluation_cost, evaluation_reward = agent.evaluate(episodes=100)

# Saving evaluation results straight from function
try:
    with open(f'{storage_folder}/{agent_name}_evaluation_reward.json', 'r') as f:
        existing_data_reward = json.load(f)
        existing_data_reward.extend(evaluation_reward)
    with open(f'{storage_folder}/{agent_name}_evaluation_reward.json', 'w') as f:
        json.dump(existing_data_reward, f)
except FileNotFoundError:
    with open(f'{storage_folder}/{agent_name}_evaluation_reward.json', 'w') as f:
        json.dump(evaluation_reward, f)

try:
    with open(f'{storage_folder}/{agent_name}_evaluation_cost.json', 'r') as f:
        existing_data_cost = json.load(f)
        existing_data_cost.extend(evaluation_cost)
    with open(f'{storage_folder}/{agent_name}_evaluation_cost.json', 'w') as f:
        json.dump(existing_data_cost, f)
except FileNotFoundError:
    # If the file doesn't exist, just dump the first results
    with open(f'{storage_folder}/{agent_name}_evaluation_cost.json', 'w') as f:
        json.dump(evaluation_cost, f) # Does this clear the variable??

# Opening and loading training rewards straight from json file
with open(f'{storage_folder}/{agent_name}_evaluation_cost.json', 'r') as f:
    evaluation_cost = json.load(f)
with open(f'{storage_folder}/{agent_name}_evaluation_reward.json', 'r') as f:
    evaluation_reward = json.load(f)

# Opening and loading training rewards straight from json file
with open(f'{storage_folder}/{agent_name}_training_cost.json', 'r') as f:
    training_cost = json.load(f)
with open(f'{storage_folder}/{agent_name}_training_reward.json', 'r') as f:
    training_reward = json.load(f)

# Creating Figure
create_figure(training_reward, training_cost, evaluation_cost, evaluation_reward, 0, 0, agent_name, storage_folder=storage_folder)

print(f'episode: {start_episode} - {end_episode} successfully ended')