This is a project to demonstrate the use of the Deep-Q-Network (DQN) Algorithm by Mnih et al. on a maintenance repair problem for a generic fuel infrastructure.

## 1. Introduction
The goal is to demonstrate that we can use the DQN algorithm to find a maintenance policy that determines the repair schedule of components on a generic fuel infrastructure model. 
This repository fulills this purpose by providing a fully integrated environment simulation that a policy can interact with. It is therefore possible to implement other policies to benchmark against it.
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
- 
