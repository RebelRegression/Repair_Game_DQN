This is a project to demonstrate the use of the Deep-Q-Network (DQN) Algorithm by Mnih et al. on a maintenance repair problem for a generic fuel infrastructure.

## 1. Introduction
The goal is to demonstrate that we can use the DQN algorithm to find a maintenance policy that determines the repair schedule of components on a generic fuel infrastructure model. 
This repository fulills this purpose by providing a fully integrated environment simulation that a policy can interact with. It is therefore possible to implement other policies to benchmark against it. The policies can be evaluated on their performance in multiple different scenarios. The policies are concerened with making a decision at each step on whether to replace any component (in this model components are pipes connecting the nodes) or not under the given constraints. 
The repository is therefore structured in these key components:
- Min-Cost Flow Model
- Environment Simulation
- Policy Implementation (DQN besides heuristic policies)
- Data visualization and collection functions

Note that it is possible to fully change each component with a different implementation. It is easy to change the hyperparameters for the underlying Min-Cost model as well as implementing different policies. It is also possible to interact with environment hyperparameters, all of which greatly impact the resulting policies. 

## 2. Min-Cost Flow Model 
This section and code is mostly inspired and taken from the paper "Operational Models if Infrastructure Resilience" by Alderson et al. (2015). In essence it implements the generic fuel infrastructure presented in it in python. It uses the pyomo library to solve the model.
This code can be found under the /System folder. 
- arc_data.csv: provides the hyperparameters for the arcs in the model.
- node_data.csv: provides the hyperparameters for the nodes in the model. Here values like demand and supply as well as penalities for shortages at each node can be defined. It is also possible to change the structure of the entire system with these files by adding or removing nodes.
- fuelnet_model_2022.py: builds the model with all constraints and the objective function. Here individual functions and constraints can be introducted.
- enumerate_metagraph_states.py: enumerates all possible states for the model and returns the value of the objective function. The states are represented as int numbers that are derived from a bitstring. The bitstring represents the state for each component with a 0 for a working condition and a 1 for a failed condition. E.g. '000000000000000000' would translate to int 0 which would represent the state where all components are working. '000000000000000001' would translate to int 1 and represent the state where the first arc/component (n01, n02) is non-operational. Precomputing all state - obj value pairs greatly increases the training performance of the DQN policies as it is a simple tabular lookup for the reward. However it is possible to implement an online version, where the states are only evaluated once the agent visits it during training or evaluation.
- states_solution_full.csv: contains all state - obj function value pairs. In this case the objective function returns the cost of operating the network in this specific state. E.g. when all components are broken the shortfall is maximal returning the cost of 150.

## 3. Environment Simulation
In order to have the policy interact with the flow model an environment file is created. Its purpose is to evaluate each action given by the policy and return the cost of the chosen action as well as the new state. This is done in the environment.py file. The environment is defined as a python class and is initialized with the hyperparameters for each run. In order to truly compare different policies these have to be kept the same for obvious reasons. 
The logic behind the state transition is regulated within the environment class by the function state_transition:
- It first receives a the current state and the selected action by the policy. The action has to be encoded in the same way like the state and represented as a bit vector, with 0 for no action on the corresponding component and 1 for the replace action.
- It then updates the state of every single component based on the action provided. If no action is provided the component will fail with the associated failure probability for that component, if its not already broken. If it does not fail the component failure probability is increased by the specified amount for the next turn.
- If a component is already broken and the action is to repair it the component becomes operational again and the component failure rate is reset to its initial value.
- If the component is operational and the action is still repair the components failure rate will just be reset to simulate a brandnew component.
- In the end the total cost of that action and the system state will be calculated. Note that each action costs a defined amount. This is then added to the cost of operating in this new state, that is derived from the state_solution_full.csv file.
- Additionaly the reward is passed. This is a scaled version of the cost in order to train the DQN agents as they are sensitive to huge values. The cost is passed for human interpretation. In addition a done flag is passed, that allows for the completion of an episode if a certain condition is met. This is only used in the surprise attack scenario. 

How do we keep track of the individual failure rate for each component?
The state representation is extended by n elements that hold the current failure rate for each component associated with that index. So the operational state of component 1 (n01, n02) will be at state[0] and the corresponding failure rate at state[18]. This is done to allow the agent to observe failure rates if that is desired.

In essence two different scenarios are coded here. The normal operation scenario allows for a continious operation of the infrastructure for a specified amount of steps. This is defined in the policy files themselves. The surprise attack scenario intializes an episode with a partially destroyed system (the surprise attack). It then sets all failure rates to zero so that operational components remain operational. It then passes the done flag once all components have been fixed. 

Multiple helper functions in this file facilitate the state transition and environment class. 

## 4. Policies

Multiple policies are implemented in this repository. This project comes with multiple DQN versions as well as one heuristic version and a One-Step-Optimal-Search (OSAS) policy. One can add different policies here. Each policy is coded in its specific file, but draws and stores data during training and evaluation to a specified storage folder. The folder can be found under /data/{policy_name}. The data stored during training are mostly 

#### DQN-Policies: 
we implemented 3 different versions in this initial repo. 
- V1: The first one learns only from the available failure indicators of the state. It therefore only sees the first 18 elements of the state vector since those contain the operational status of each component.
- V2: Learns from the failure indicators as well as the failure probability of each component. It therefore comes up with an action based on the entire state vector.
- V3: Is identical to V2 but it starts off each episode in a broken state. This is done to expand the search of the state-action space.

Each DQN policy is defined in the files and follows a similar pattern. When initializing an agent under a certain name it looks up if there are any existing models under that name. If so those models will be preloaded. This allows to create a checkpoint for that agent that it can continue training from. The implementation here is not ideal, since it is theoretical possible to compile a network and load it from a checkpoint. However this is currently not possible due to the dynamic nature of the learning algorithm. This has to be addressed in future updates. In order to accomodate for that fact an implementation has been chosen, where the user needs to specify the name. The starting episode of that run, the end episode of that run and then the total number of episodes this agent is to be trained. 

Starting the agent from the cli like so:
$python3 DQLV1.py {agent_name} 0 100 500 
This would correlate to the first one hundred episodes of the final 500 episodes training run for that agent. This is also necessary to calculate the current epsilon value in case the training process is to be split up in multiple seperate runs. This is necessary when training some very large number of episodes like 15k or above. 

Each agent is defined as a class with the hyperparameters for the run specified. More details about the specific implementation are given in the respective files.

#### Heuristic Policies
In addition to the DQN-Policies we implemented two heuristic policies. The planned heuristic policy $H_P$ and the random heuristic policy $H_R$. 

- $H_P$ provides a planned approach to repairing components by prioritizing broken components based on dictionary defined in the file. The arcs are ordered based off the flow they facilitate in the fully functional state int 0. This is ment to simulate the idea of fixing the "biggest" or supposedly most vital components first.
- $H_R$ repairs components at random once they are broken and makes no distinction between individual arcs.

Note that both policies are incapable of conducting preemptive repairs. This is in different from the OSAS and DQN policies. 

#### One-Step-Action-Search Policy
This policy follows a simplistic algorithm where in each state all possible actions are evaluated against each other. We then select the one with the highest return in this state. We then simulate the environment for an additional x steps without taking any action and take these costs into account. This allows for the future implied cost to be included in the action evaluation. 
This policy is calculated in parrallel as this greatly speeds up the computation process. For each state a number of child processes are spawned that evaluate each action independently. At the end all the data is combined and the parent process evaluates what action to take. It is therefore advisable to create a bash script to start multiple instances of the OSAS_Policy file with different start and end states to brute force the entire state space much faster. This is highly dependant on your server and available ressources. 

## Data Collection and Analysis
In order to compare different policies and generate insights on the learned strategies data has to be collected during training and evaluation. 
The data collected follows a specific pattern that then can be accessed by the support functions in the *support.py* file that allow for data visualization.

#### Data Collection
There are multiple data types and files created during training. This data can be accessed in the data/{policy_name}/{agent_name} folder. E.g. data/V1/V1001. 

The following data is collected for all policies:
- evaluation_cost: the average cost per episode during evaluation
- evaluation reward: the average reward per episode during evaluation
- component data: Number of repairs, preemptive repairs and downtimes on all components during the evaluation run

For the DQN Policies this data is also collected during the training time. 












