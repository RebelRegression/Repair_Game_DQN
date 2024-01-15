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
  
