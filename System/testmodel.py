import pandas as pd
import os
import sys

print("Current working directory:", os.getcwd())

print("\n\nLoading data about network model...\n")

# # get node data
# node_data_df = pd.read_csv('node_data.csv', index_col=['node'])
# nodes = list(node_data_df.index.unique())
# print('There are %d nodes:' % len(nodes))
# print(nodes)

# # get arc data
# edges_df = pd.read_csv('arc_data.csv')
# # set the index to be ['i','j']
# edges_df = edges_df.set_index(['i','j'])
# edges = list(edges_df.index.values)
# print('There are %d edges: ' % len(edges))
# print(edges)

print("\nReading file of past solves...", end='')
states_df = pd.read_csv('states.csv', index_col=['state'])
soln_dict = states_df['performance'].to_dict()
print("done.  Read %d entries." % len(states_df))
maxindex = int(max(states_df.index.unique()))
print(maxindex)