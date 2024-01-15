import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import multiprocessing

from tqdm import tqdm

from keras.utils import *
import tensorflow as tf
from copy import deepcopy
import json

# # Fetching environment object from parent directory
from Environment_package.environment_generator import bit_val, bits


# DESCRIPTION: 
# - this is the cli tuned version of V1. It is able to split the training process up into multiple training runs, that are executed in sequence. 
#   This conserves memory which seems to be an issue. 
#   This file should be started from the command line with the following parameters (agent_name) (start episode) (end episode) (total episodes)
#   example -> $python3 DQLV1tensorflow_cli.py v1001 0 1000 1000
#   This sets the the epislon correctly for each run. This allows multiple runs to be executed in a bash script. See sbashV1.sh for an example
#
# - this Agent does not see the failure rate of each component. It only sees if each component is functional.


def check_existing_model(action_size: int, state_size: int, agent_name: str, storage_folder: str):
    '''takes the agent name and loads if an existing model is found
    Args: 
        action_size    (int): total number of possible actions
        state_size     (int): total number of values from an input state
        agent_name     (str): name of the agent
        storage_folder (str): location of the data for the agent
        
    Returns:
        model        (keras obj): a keras model 
        target_model (keras obj): the target model as keras obj'''

    # Making sure that there is no other agent with the same name, if so use the existing agent
    try:
        model = keras.models.load_model(f'{storage_folder}/{agent_name}.keras')
        target_model = keras.models.load_model(f'{storage_folder}/{agent_name}_target.keras')
        print(f'Agent with name: {agent_name} found. Using these existing models')
    except:
        print('No existing agent found. Creating new model')
        model = create_model(action_size, state_size)
        target_model = create_model(action_size, state_size)

    return model, target_model

def create_model(action_size, state_size):
    
    inputs = layers.Input(shape=(state_size,))
    layer1 = layers.Dense(128, activation='relu')(inputs)
    layer2 = layers.Dense(256, activation='relu')(layer1)
    action = layers.Dense(action_size, activation='linear')(layer2)

    return keras.Model(inputs=inputs, outputs=action)

class ReplayMemoryV2:
    """The memory buffer, that stores replay experience for the V2 Agent"""

    def __init__(self, max_memory_length=100000):
        self.action_history = []
        self.state_history = []
        self.next_state_history = []
        self.rewards_history = []
        self.done_history = []
        # Deepmind paper suggest 1mio, but this might cause memory issues
        self.max_memory_length = max_memory_length
    
    def push(self, action: list, state: list, next_state: list, reward: float, done: bool):
        '''pushes a transition tuple onto the memory stack'''
        self.action_history.append(action)
        self.state_history.append(state)
        self.next_state_history.append(next_state)
        self.rewards_history.append(reward)
        self.done_history.append(done)

    	# Pop first element if buffer starts to get to big (FIFO)
        if len(self.done_history) > self.max_memory_length:
            self.action_history.pop(0)
            self.state_history.pop(0)
            self.next_state_history.pop(0)
            self.rewards_history.pop(0)
            self.done_history.pop(0)
    
    def sample(self, batch_size: int):
        """Creates a random sample from the replay memory in length of the batch size
        Args:
            batch_size (int): the batch size the model is tuned to

        Returns:
            action_sample (list)
            state_sample  (array)
            next_state_sample (array)
            rewards_sample (list)
            done_sample (tensor)
        """
        indices = np.random.choice(range(len(self.done_history)), size=batch_size)
        action_sample = deepcopy([self.action_history[i] for i in indices])
        state_sample = deepcopy(np.array([self.state_history[i] for i in indices]))
        next_state_sample = deepcopy(np.array([self.next_state_history[i] for i in indices]))
        rewards_sample = deepcopy([self.rewards_history[i] for i in indices])
        done_sample = deepcopy([self.done_history[i] for i in indices])

        return action_sample,  state_sample, next_state_sample, rewards_sample, done_sample

        

class tensoragentv1:
    """This agent gets to see the failure rates

    Attributes:
        agent_name          (str)  : name of the agent
        max_episodes        (int)  : max number of episodes in order to get constant epsilon decay value 
        max_steps           (int)  : steps per episode for agent
        environment         (class): environment the agent should learn on
        gamma               (float): discount factor
        start_epsilon       (float): original starting value of epsilon in order to get constant epsilon decay value
        epsilon             (float): starting value for epsilon
        epsilon_min         (float): min value for epsilon
        max_steps           (int)  : steps per episode for agent
        optimizer           (class): keras optimizer for the model
        loss_function       (class): keras loss function for the model
        update_after_action (int)  : after how many actions the model is updated
        update_target_model (int)  : after how many episodes the target model is updated
        storage_folder      (str)  : path to storage folder for results
    """

    def __init__(self, agent_name: str, environment, max_episodes: int=5000, gamma: float =0.90, start_epsilon: float=0.9, epsilon: float=1.0, epsilon_min: float=0.1, max_steps: int=360, optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0), loss_function=keras.losses.Huber(), update_after_actions: int=4, update_target_model: int=32, storage_folder: str='data') -> None:
        """Initializes agent"""
        # Hyperparameters
        self.agent_name = agent_name
        self.env = environment
        self.action_size = environment.action_space.shape[0]
        self.gamma = gamma
        self.max_steps = max_steps
        self.storage_folder = storage_folder
        self.episodes = max_episodes

        # Epsilon parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.start_epsilon = start_epsilon
        self.epsilon_decay_factor = (self.epsilon_min/self.start_epsilon) ** (1/self.episodes)

        # Creating the model 
        self.model = check_existing_model(environment.action_space.shape[0], environment.n_components, agent_name, storage_folder)[0]  # n_components needs to be *2 to add nodes for the failure chance for each component
        self.target_model = check_existing_model(environment.action_space.shape[0], environment.n_components, agent_name, storage_folder)[1]
        self.optimizer = optimizer
        self.loss_function = loss_function

        # How often to update the network
        self.update_after_actions = update_after_actions
        # How often to update the target network
        self.update_target_model = update_target_model

        self.ReplayMemory = ReplayMemoryV2()

    def load_optimizer_config(self):
        '''loads config from d'''
        try:
            with open(f'{self.storage_folder}/{self.agent_name}_optimizer.pkl', 'rb') as file:
                optimizer_config = pickle.load(file)
                self.optimizer = keras.optimizers.Adam.from_config(optimizer_config)
            print('Loaded existing optmizer config')
        except:
            print('No existing optimizer config found. Creating new one')

    def save_optimizer_config(self):
        '''saves the current optimizer config'''
        optimizer_config = self.optimizer.get_config()
        with open(f'{self.storage_folder}/{self.agent_name}_optimizer.pkl', 'wb') as file:
            pickle.dump(optimizer_config, file)


    def train(self, start_episode: int, end_episode: int, batch_size: int=64) -> None:
        """Training the agent for the specified amount of episodes, using batches

        Args:
            start_episode   (int): starting episode (this determines the epsilon value)
            end_episode     (int): last episode the agent trains on
            batch_size (int): the size of batches for updating the model
        
        Returns:
            None: saves training data every 10 episodes in json
            """
        # Set the number of CPU threads
        num_cpus = multiprocessing.cpu_count()
        print("Num CPUs Available: ", num_cpus)
        # tf.config.threading.set_intra_op_parallelism_threads(num_cpus)

        # Loading the optimizer config, if none exist a new optimizer will be initalized
        self.load_optimizer_config()

        # Setting the correct epsilon based on the current episode
        self.epsilon = self.start_epsilon
        for i in range(start_episode):
            self.epsilon = self.epsilon * self.epsilon_decay_factor
        print('first epsilon : ', self.epsilon)
        
        # Initializing storage variables for training
        running_reward = []
        running_cost = []  

        # Training the agent for specified amount of episodes
        for episode in tqdm(range(start_episode, end_episode)):

            # Creating a clean new state with all components set to working and initial failure rate
            state = [0 for i in range(self.env.n_components)]
            state = state +  [self.env.initial_failure_rate for x in range(self.env.n_components)]

            # Initializing storage variables for episode
            episode_reward = []
            episode_cost = []

            # Saving the model as backup in every 100 intervalls
            if episode % 100 == 0:
                self.model.save(f'{self.storage_folder}/{self.agent_name}.keras')

            # Going x steps for every episode
            for step in range(self.max_steps):

                # Use epsilon-greedy to select action
                if self.epsilon > np.random.uniform(0,1):
                    # Take a random action if epsilon value is bigger than random float
                    actionidx = np.random.choice(self.action_size)      
                else: 
                    # Predict action based on best calculated q-value
                    state_tensor = tf.convert_to_tensor(state[:self.env.n_components])
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)
                    actionidx = tf.argmax(action_probs[0]).numpy()

                action = self.env.action_space[actionidx]

                # Apply the chosen action to our environment and push it on the replay memory
                next_state, cost, reward, done = self.env.state_transition(state, action)
                episode_reward.append(reward)
                episode_cost.append(cost)
                self.ReplayMemory.push(actionidx, state[:self.env.n_components], next_state[:self.env.n_components], reward, done)

                # Updating the state to the next state
                state = next_state

                # Update the model after the specified amount of steps
                if step % self.update_after_actions == 0 and len(self.ReplayMemory.rewards_history) > batch_size:

                    # Sampling experiences from the replay memory
                    action_sample, state_sample, next_state_sample, rewards_sample, done_sample = self.ReplayMemory.sample(batch_size)

                    # Calculate the target q-values from the target model
                    target_model_q_values = self.target_model.predict(next_state_sample, verbose=0)
                    target_q_values = rewards_sample + self.gamma * tf.reduce_max(target_model_q_values, axis=1)

                    # Finals steps are not yet implemented!!

                    # Create a mask so we only calculate loss on the updated Q-values, the mask contains the chosen action index on a tensor from there the targeted q-value is identified in the output layer
                    masks = tf.one_hot(action_sample, self.action_size)
                    #print(masks)
                    
                    # Training the model on the states and target Q-values
                    with tf.GradientTape() as tape:
                        q_values = self.model(state_sample)
                        # Apply mask to the Q-values to identifiy the q-value that corresponds to the chosen action
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between target Q-value and predicted Q-value
                        loss = self.loss_function(target_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # Update the target network after specified amount of episodes
                if step % self.update_target_model == 0:
                    self.target_model.set_weights(self.model.get_weights())
                
                if done:
                    break
        
            running_reward.append(np.mean(episode_reward))
            running_cost.append(np.mean(episode_cost))

            # Updating the epsilon value
            self.epsilon = self.epsilon * self.epsilon_decay_factor

            # Save a copy of current reward and cost history as backup, incase training fails after 36 hours for any reason
            if episode % 50 == 0 or episode == end_episode - 1:
                # Load existing data from the JSON file (if it exists)
                try:
                    with open(f'{self.storage_folder}/{self.agent_name}_training_reward.json', 'r') as f:
                        existing_data_reward = json.load(f)
                        existing_data_reward.extend(running_reward)
                    with open(f'{self.storage_folder}/{self.agent_name}_training_reward.json', 'w') as f:
                        json.dump(existing_data_reward, f)
                        running_reward = []
                except FileNotFoundError:
                    with open(f'{self.storage_folder}/{self.agent_name}_training_reward.json', 'w') as f:
                        json.dump(running_reward, f)
                        running_reward = []

                try:
                    with open(f'{self.storage_folder}/{self.agent_name}_training_cost.json', 'r') as f:
                        existing_data_cost = json.load(f)
                        existing_data_cost.extend(running_cost)
                    with open(f'{self.storage_folder}/{self.agent_name}_training_cost.json', 'w') as f:
                        json.dump(existing_data_cost, f)
                        running_cost = []
                except FileNotFoundError:
                    # If the file doesn't exist, just dump the first results
                    with open(f'{self.storage_folder}/{self.agent_name}_training_cost.json', 'w') as f:
                        json.dump(running_cost, f)
                        running_cost = []

        # Creating final save of model
        self.model.save(f'{self.storage_folder}/{self.agent_name}.keras')
        self.target_model.save(f'{self.storage_folder}/{self.agent_name}_target.keras')

        # Saving the optimizer config 
        self.save_optimizer_config()

        print('last epsilon: ', self.epsilon)

    

    def evaluate(self, episodes: int=100) -> tuple[list, list]:
        """Evaluates the agent for the specified amount of episodes
        
        Args: 
        evaluation_episodes (int): number of episodes the agent is evaluated on
        
        Returns:
        running_cost   (list): average cost per episode of the system
        running_reward (list): average reward per episode of the system
        """

        # Setting storage variables
        running_cost = []
        running_reward = []

        running_repaired_arcs = []
        running_failed_arcs = []
        running_preemptive_repair_data = []

        states_visited = {key: [] for key in range(len(self.env.state_space))}

        # Stores all pairs of broken arcs during eval, structure is a 18x18 matrix with one layer per episode
        # in the end we will average across the dimension 0
        broken_pair_matrix = np.zeros((episodes, self.env.n_components, self.env.n_components))


        for episode in tqdm(range(episodes), desc='Evaluation', unit='episode'):

           # Initializing storage varibles to keep track of what arcs are repaired and failed during an episode
            n_repaired_arcs = [int(0) for x in range(self.env.n_components)]
            n_failed_arcs = [int(0) for x in range(self.env.n_components)]
            preemptive_repair_data = [[0,0] for x in range(self.env.n_components)] # each embedded list is for one component, the 0th element is the number of 
            # repairs, when the component was broken, the 1th element is the number of preemptive repairs when the component was not broken
            episode_reward = []
            episode_cost = []

            # Initializing state
            state = [0 for i in range(self.env.n_components)]
            state = state +  [self.env.initial_failure_rate for x in range(self.env.n_components)]

            # Initialize the states visited counter for each state to 0 for the episode
            for i in range(len(self.env.state_space)):
                states_visited[i].append(0)

            for step in range(self.max_steps):

                # Append current state to states_visited dict in order to store that data
                states_visited[bit_val(state[:self.env.n_components])][episode] += 1
                
                # Store data on what arcs broke in relation to others in the episode layer of the matrix
                index_of_broken_components = []
                for i, indicator in enumerate(state[:self.env.n_components]):
                    if indicator == 1:
                        index_of_broken_components.append(i)
                
                # create a list with unique pairs to write the data into the matrix
                list_of_pairs = []
                for io in index_of_broken_components:
                    for ii in index_of_broken_components:
                        list_of_pairs.append((io, ii))
                unique_pairs = set()
                for pair in list_of_pairs:
                    unique_pairs.add(tuple(sorted(pair)))
                for pair in unique_pairs:
                    broken_pair_matrix[episode][pair[0]][pair[1]] += 1
                    broken_pair_matrix[episode][pair[1]][pair[0]] += 1 # Adding on both columns to create symetrical matrix

                # Append current state to states_visited dict in order to store that data
                states_visited[bit_val(state[:self.env.n_components])][episode] += 1

                # Choose the action, based on the agent best guess
                state_tensor = tf.convert_to_tensor(state[:self.env.n_components])
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                actionidx = tf.argmax(action_probs[0]).numpy()
                action = self.env.action_space[actionidx]

                # Check for every component if it was preemptively repaired or not
                for i in range(self.env.n_components):
                    if state[i] == 1 and action[i] == 1:
                        preemptive_repair_data[i][0] += 1
                    if state[i] == 0 and action[i] == 1:
                        preemptive_repair_data[i][1] += 1

                # Apply the sampled action to our environment and storing cost and reward
                next_state, cost, reward, done = self.env.state_transition(state, action)
                episode_cost.append(cost)
                episode_reward.append(reward)
                state = next_state

                # Save the data on what arcs got repaired
                for i in range(self.env.n_components):
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
            for i in range(self.env.n_components):
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


            running_cost.append(np.mean(episode_cost))
            running_reward.append(np.mean(episode_reward))
        
        # Average across all layers   
        broken_pair_matrix = np.mean(broken_pair_matrix, axis=0)

        return running_cost, running_reward, mean_failed_arcs, mean_repaired_arcs, mean_preemptive_repair_data, states_visited, broken_pair_matrix

    
    def evaluate_surprise_attrition(self, increase_attrition_for_steps: tuple, increase_attrition_to: float, episodes: int=100) -> tuple[list, list]:
        """Evaluates the agent for the specified amount of episodes
        
        Args: 
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

        running_repaired_arcs = []
        running_failed_arcs = []
        running_preemptive_repair_data = []

        states_visited = {key: [] for key in range(len(self.env.state_space))}

        # Stores all pairs of broken arcs during eval, structure is a 18x18 matrix with one layer per episode
        # in the end we will average across the dimension 0
        broken_pair_matrix = np.zeros((episodes, self.env.n_components, self.env.n_components))

        initial_attrition_rate = self.env.attrition_rate

        for episode in tqdm(range(episodes), desc='Evaluation', unit='episode'):

            # Initializing storage varibles to keep track of what arcs are repaired and failed during an episode
            n_repaired_arcs = [int(0) for x in range(self.env.n_components)]
            n_failed_arcs = [int(0) for x in range(self.env.n_components)]
            preemptive_repair_data = [[0,0] for x in range(self.env.n_components)] # each embedded list is for one component, the 0th element is the number of 
            # repairs, when the component was broken, the 1th element is the number of preemptive repairs when the component was not broken
            episode_reward = []
            episode_cost = []

            # Initialize state
            state = [0 for i in range(self.env.n_components)]
            state = state +  [self.env.initial_failure_rate for x in range(self.env.n_components)]

            # Initialize the states visited counter for each state to 0 for the episode
            for i in range(len(self.env.state_space)):
                states_visited[i].append(0)


            for step in range(self.max_steps):

                # Store data on what arcs broke in relation to others in the episode layer of the matrix
                index_of_broken_components = []
                for i, indicator in enumerate(state[:self.env.n_components]):
                    if indicator == 1:
                        index_of_broken_components.append(i)
                
                # create a list with unique pairs to write the data into the matrix
                list_of_pairs = []
                for io in index_of_broken_components:
                    for ii in index_of_broken_components:
                        list_of_pairs.append((io, ii))
                unique_pairs = set()
                for pair in list_of_pairs:
                    unique_pairs.add(tuple(sorted(pair)))
                for pair in unique_pairs:
                    broken_pair_matrix[episode][pair[0]][pair[1]] += 1
                    broken_pair_matrix[episode][pair[1]][pair[0]] += 1

                # Append current state to states_visited dict in order to store that data
                states_visited[bit_val(state[:self.env.n_components])][episode] += 1

                # Increasing attrition rate during specified steps
                if increase_attrition_for_steps[1] >= step >= increase_attrition_for_steps[0]:
                    self.env.attrition_rate=increase_attrition_to
                # Resetting attrition rate to default after surprise is over
                elif step > increase_attrition_for_steps[1]:
                    self.env.attrition_rate=initial_attrition_rate

                # Choose the action, based on the agent best guess
                state_tensor = tf.convert_to_tensor(state[:self.env.n_components])
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                actionidx = tf.argmax(action_probs[0]).numpy()
                action = self.env.action_space[actionidx]

                # Check for every component if it was preemptively repaired or not
                for i in range(self.env.n_components):
                    if state[i] == 1 and action[i] == 1:
                        preemptive_repair_data[i][0] += 1
                    if state[i] == 0 and action[i] == 1:
                        preemptive_repair_data[i][1] += 1

                # Apply the sampled action to our environment and storing cost and reward
                next_state, cost, reward, done = self.env.state_transition(state, action)
                episode_cost.append(cost)
                episode_reward.append(reward)
                state = next_state

                # Save the data on what arcs got repaired
                for i in range(self.env.n_components):
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
            for i in range(self.env.n_components):
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


            running_cost.append(np.mean(episode_cost))
            running_reward.append(np.mean(episode_reward))
        
        # Average across all layers
        broken_pair_matrix = np.mean(broken_pair_matrix, axis=0)

        return running_cost, running_reward, mean_failed_arcs, mean_repaired_arcs, mean_preemptive_repair_data, states_visited, broken_pair_matrix

    
    def evaluate_surprise_attack(self, states) -> tuple[list, list]:
        """Evaluates the agent on the specified starting states
        
        Args: 
        states  (list): list with all int values of starting states to go through
        
        Returns:
        running_cost   (list): average cost per episode of the system
        running_reward (list): average reward per episode of the system
        """

        # Setting storage variables
        running_cost = []
        running_reward = []

        for x in tqdm(range(len(states))):
            # Initialize the starting state by picking a state from the given state list
            state = list(bits(states[x], self.env.n_components))
            counter = 0
            for element in state:
                state[counter] = int(element)
                counter += 1
            state += [self.env.initial_failure_rate for i in range(self.env.n_components)]

            episode_reward = []
            episode_cost = []

            for step in range(self.max_steps):


                # Choose the action, based on the agent best guess
                state_tensor = tf.convert_to_tensor(state[:self.env.n_components])
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                actionidx = tf.argmax(action_probs[0]).numpy()
                action = self.env.action_space[actionidx]

                # Apply the sampled action to our environment and storing cost and reward
                next_state, cost, reward, done = self.env.state_transition(state, action)
                episode_cost.append(cost)
                episode_reward.append(reward)
                state = next_state

                # Break loop if done
                if done:
                    break

            running_cost.append(np.sum(episode_cost))
            running_reward.append(np.sum(episode_reward))
            

        return running_cost, running_reward




