import networkx as nx
import matplotlib.pyplot as plt
from pyomo.environ import *
from fuelnet_model_2022 import *
import pyomo.visualizer as visualize

print("\n\nLoading data about network model...\n")

# get node data
node_data_df = pd.read_csv('node_data.csv', index_col=['node'])
nodes = list(node_data_df.index.unique())
print('There are %d nodes:' % len(nodes))
print(nodes)

# get arc data
edges_df = pd.read_csv('arc_data.csv')
# set the index to be ['i','j']
edges_df = edges_df.set_index(['i','j'])
edges = list(edges_df.index.values)
print('There are %d edges: ' % len(edges))
print(edges)

num_components = len(edges)

antiedges = [ (j,i) for (i,j) in edges]
arcs = edges + antiedges

print('There are %d arcs:' % len(arcs))
print(arcs)

model = build_model(nodes, edges, arcs, node_data_df, edges_df)


# Create a NetworkX graph
G = nx.Graph()

# Add nodes and edges based on your model
for var in model.component_objects(Var, active=True):
    for key in var:
        # print(key)
        G.add_node(f"{var.name}[{key}]")

# Add edges between constraints and variables
for constr in model.component_data_objects(Constraint, active=True):
    for key in constr.keys():
        for var in model.component_data_objects(Var, active=True):
            for var_key in var:
                if var in constr[key]:
                    G.add_edge(f"{constr.name}[{key}]", f"{var.name}[{var_key}]")


# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=800, font_size=10, font_color='black')
plt.show()
