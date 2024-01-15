import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import networkx as nx
from environmentfile import bits, bit_val
import pandas as pd

def create_figure(training_running_reward: list, training_running_cost: list, evaluation_running_cost: list, evaluation_running_reward: list, training_window: int, evaluation_window: int, agent_name:str, storage_folder:str):
    """creates a 4x4 figure that shows data on the trained agent.
    
    Args:
        training_running_reward   (list): a list with float data, that represents all the training reward
        training_running_cost     (list): a list with float data, that represents the cost at each episode
        evaluation_running_cost   (list): a list with float data, that represents the cost at each evaluation episode
        evaluation_running_reward (list): a list with float data, that represents the reward at each evaluation episode
        training_window           (int) : number of episodes that the graph will use as average for each point
        evaluation_window         (int) : number of episodes that the graph will usa as average for each point
        agent_name                (str) : name that the png file will be saved as
        storage_folder            (str) : path of the folder, the png will be saved to
        
    Returns:
        agent_name.png (png): a picture of the data as graph representation"""

    trainingcost_values = []
    trainingreward_values = []
    for i in range(training_window,len(training_running_reward)):
        trainingcost_values.append(np.mean(training_running_cost[i-training_window:i]))
        trainingreward_values.append(np.mean(training_running_reward[i-training_window:i]))
        
    evalcost_values = []
    evalreward_values = []
    for i in range(training_window,len(evaluation_running_reward)):
        evalcost_values.append(np.mean(evaluation_running_cost[i-evaluation_window:i]))
        evalreward_values.append(np.mean(evaluation_running_reward[i-evaluation_window:i]))


    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Plot the first subplot
    axes[0, 0].set_xlabel('episode')
    axes[0, 0].set_ylabel('average cost per step')
    axes[0, 0].plot(trainingcost_values)
    axes[0, 0].set_title('Training: Cost')

    # Plot the second subplot
    axes[0, 1].set_xlabel('episode')
    axes[0, 1].set_ylabel('average reward per step')
    axes[0, 1].plot(trainingreward_values)
    axes[0, 1].set_title('Training: Reward')

    # Plot the second subplot
    axes[1, 0].set_xlabel('episode')
    axes[1, 0].set_ylabel('average cost per step')
    axes[1, 0].plot(evalcost_values)
    axes[1, 0].set_title('Evaluation: Cost')

    # Plot the second subplot
    axes[1, 1].set_xlabel('episode')
    axes[1, 1].set_ylabel('average reward per step')
    axes[1, 1].plot(evalreward_values)
    axes[1, 1].set_title('Evaluation: Reward')


    # Adjust layout for better spacing
    plt.tight_layout()
    try:
        plt.savefig(f'{storage_folder}/{agent_name}.png')
    except:
        plt.savefig(f'{agent_name}.png')
        print(f'Could not find given storage folder path: "{storage_folder}". \n Saving in this directory instead.')


def scale_data(agent_name: str, storage_folder: str, training_window: int, eval_window: int):
    """Takes the agent name and returns the scaled data for the plot
    Args:
        agent_name (str): the name of the agent, this will typically be V2001 it has to be the identifying part of the jsons files with the data
        storage_folder (str): path to the folder, where the jsons are stored
        training_wimdow (int): number of datapoint to take the average from for a smoother curve in training data (this does not change the actual file!)
        eval_window (int): number of datapoint to take the average from for a smoother curve in eval data (this does not change the actual file!)
    
    Returns:
        scaled_training_cost_data   (list): data ready for plotting
        scaled_training_reward_data (list): data ready for plotting
        scaled_eval_cost_data       (list): data ready for plotting
        scaled_eval_reward_data     (list): data ready for plotting
    
    """

    # Load data from JSON files
    try: 
        with open(f'{storage_folder}/{agent_name}_training_cost.json', 'r') as f:
            training_running_cost = json.load(f)
        with open(f'{storage_folder}/{agent_name}_training_reward.json', 'r') as f:
            training_running_reward = json.load(f)
        with open(f'{storage_folder}/{agent_name}_evaluation_cost.json', 'r') as f:
            evaluation_running_cost = json.load(f)
        with open(f'{storage_folder}/{agent_name}_evaluation_reward.json', 'r') as f:
            evaluation_running_reward = json.load(f) 
    except: 
        raise FileNotFoundError('Data jsons could not be found. Are you sure you got the right file path?')
    

    # Scaling data
    scaled_training_cost_data = []
    scaled_training_reward_data = []
    for i in range(training_window,len(training_running_cost)):
        scaled_training_cost_data.append(np.mean(training_running_cost[i-training_window:i]))
        scaled_training_reward_data.append(np.mean(training_running_reward[i-training_window:i]))

    scaled_eval_cost_data = []
    scaled_eval_reward_data = []
    for i in range(eval_window,len(evaluation_running_cost)):
        scaled_eval_cost_data.append(np.mean(evaluation_running_cost[i-eval_window:i]))
        scaled_eval_reward_data.append(np.mean(evaluation_running_reward[i-eval_window:i]))
    
    return scaled_training_cost_data, scaled_training_reward_data, scaled_eval_cost_data, scaled_eval_reward_data


def create_custom_figure(*agent_names, storage_folder:str, training_window: int, eval_window: int, name_of_plot_file: str):
    """Creates a custom plot with multiple agents for comparison, returns a save file
    Args:
        agent_names (str): names of agents to include in plot, has to be the identifyer on the beginning of each json file
        storage_folder (str): path to the folder where the data is stored
        training_window (int): averages over x steps of training data to smooth out the curve
        eval_window (int): averages over x steps of eval data to smooth out the curve
        
    Returns:
        file (png): plot saved as png in folder

    Example use: create_custom_figure('V3002', 'V3031', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='results')
    """

    if len(agent_names) > 7:
        raise ValueError('Please assign no more than 7 agents to this plot, we run out of nice colors otherwise')

    # Appending all scaled data to a dict
    scaled_data_dict = {}

    for name in agent_names:
        if name[:2] == 'V1':
            storage_folder = 'data/V1/state_solution_V2'
        elif name[:2] == 'V2':
            storage_folder = 'data/V2/state_solution_V2'
        elif name[:2] == 'V3':
            storage_folder = 'data/V3/state_solution_V2'

        scaled_training_cost_data, scaled_training_reward_data, scaled_eval_cost_data, scaled_eval_reward_data = scale_data(name, storage_folder, training_window, eval_window)
        scaled_data_dict[f'{name}_training_cost'] = scaled_training_cost_data
        scaled_data_dict[f'{name}_training_reward'] = scaled_training_reward_data
        scaled_data_dict[f'{name}_eval_cost'] = scaled_eval_cost_data
        scaled_data_dict[f'{name}_eval_reward'] = scaled_eval_reward_data

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']

    # Plot the first subplot
    axes[0, 0].set_xlabel('episode')
    axes[0, 0].set_ylabel('average cost per step')
    color_idx = 0
    for name in agent_names:
        data = scaled_data_dict[f'{name}_training_cost']
        axes[0, 0].plot(data, label=f'{name} (Mean: {np.mean(data):.2f})', color=colors[color_idx])
        color_idx += 1
    axes[0, 0].legend()
    axes[0, 0].set_title('Training: Cost')

    # Plot the second subplot
    axes[0, 1].set_xlabel('episode')
    axes[0, 1].set_ylabel('average reward per step')
    color_idx = 0
    for name in agent_names:
        data = scaled_data_dict[f'{name}_training_reward']
        axes[0, 1].plot(data, label=f'{name} (Mean: {np.mean(data):.2f})', color=colors[color_idx])
        color_idx += 1
    axes[0, 1].legend()
    axes[0, 1].set_title('Training: Reward')

    # Plot the second subplot
    axes[1, 0].set_xlabel('episode')
    axes[1, 0].set_ylabel('average cost per step')
    color_idx = 0
    for name in agent_names:
        data = scaled_data_dict[f'{name}_eval_cost']
        axes[1, 0].plot(data, label=f'{name} (Mean: {np.mean(data[-100:]):.2f})', color=colors[color_idx])
        color_idx += 1
    axes[1, 0].legend()
    axes[1, 0].set_title('Evaluation: Cost')

    # Plot the second subplot
    axes[1, 1].set_xlabel('episode')
    axes[1, 1].set_ylabel('average reward per step')
    color_idx = 0
    for name in agent_names:
        data = scaled_data_dict[f'{name}_eval_reward']
        axes[1, 1].plot(data, label=f'{name} (Mean: {np.mean(data[-100:]):.2f})', color=colors[color_idx])
        color_idx += 1
    axes[1, 1].legend()
    axes[1, 1].set_title('Evaluation: Reward')


    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f'{name_of_plot_file}.png')

def create_cost_training_figure(*agent_names, storage_folder:str, training_window: int, eval_window: int, name_of_plot_file: str):
    """Creates a custom plot with multiple agents for comparison, returns a save file
    Args:
        agent_names (str): names of agents to include in plot, has to be the identifyer on the beginning of each json file
        storage_folder (str): path to the folder where the data is stored
        training_window (int): averages over x steps of training data to smooth out the curve
        eval_window (int): averages over x steps of eval data to smooth out the curve
        
    Returns:
        file (png): plot saved as png in folder

    Example use: create_custom_figure('V3002', 'V3031', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='results')
    """

    if len(agent_names) > 7:
        raise ValueError('Please assign no more than 7 agents to this plot, we run out of nice colors otherwise')

    # Appending all scaled data to a dict
    scaled_data_dict = {}

    for name in agent_names:
        if name[:2] == 'V1':
            storage_folder = 'data/V1/state_solution_V2'
        elif name[:2] == 'V2':
            storage_folder = 'data/V2/state_solution_V2'
        elif name[:2] == 'V3':
            storage_folder = 'data/V3/state_solution_V2'

        scaled_training_cost_data, scaled_training_reward_data, scaled_eval_cost_data, scaled_eval_reward_data = scale_data(name, storage_folder, training_window, eval_window)
        scaled_data_dict[f'{name}_training_cost'] = scaled_training_cost_data
        # scaled_data_dict[f'{name}_training_reward'] = scaled_training_reward_data
        scaled_data_dict[f'{name}_eval_cost'] = scaled_eval_cost_data
        # scaled_data_dict[f'{name}_eval_reward'] = scaled_eval_reward_data

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']

    # Plot the first subplot
    ax.set_xlabel('episode')
    ax.set_ylabel('average cost per step')
    color_idx = 0
    for name in agent_names:
        data = scaled_data_dict[f'{name}_training_cost'][:14000]
        if name[:2] == 'V1':
            labelname = f'$PIP$'
        if name[:2] == 'V2':
            labelname = f'$FIP_F$'
        elif name[:2] == 'V3':
            labelname = f'$FIP_V$'
        ax.plot(data, label=f'{labelname}', color=colors[color_idx])
        color_idx += 1
    ax.legend()
    # ax.set_title('Training: Cost')


    # # Plot the second subplot
    # ax.set_xlabel('episode')
    # ax.set_ylabel('average cost per step')
    # color_idx = 0
    # for name in agent_names:
    #     data = scaled_data_dict[f'{name}_eval_cost']
    #     num_data_points = len(data) // 100

    #     # Reshape the data for candlestick plot
    #     reshaped_data = [data[i*100:(i+1)*100] for i in range(num_data_points)]


    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f'{name_of_plot_file}.png')
    
# create_cost_training_figure('V1021', 'V2021','V30223', storage_folder='data/V3', training_window=100, eval_window=1, name_of_plot_file='15k')


def create_custom_figure_eval(*agent_names, storage_folder:str, eval_window: int, name_of_plot_file: str):
    """Creates a custom plot with multiple agents for comparison with the eval data only, returns a save file
    Args:
        agent_names (str): names of agents to include in plot, has to be the identifyer on the beginning of each json file
        storage_folder (str): path to the folder where the data is stored
        eval_window (int): averages over x steps of eval data to smooth out the curve
        
    Returns:
        file (png): plot saved as png in folder

    Example use: create_custom_figure('V3002', 'V3031', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='results')
    """

    if len(agent_names) > 7:
        raise ValueError('Please assign no more than 7 agents to this plot, we run out of nice colors otherwise')

    # Appending all scaled data to a dict
    scaled_data_dict = {}

    for name in agent_names:

        # Splitting the name to get the identfier part
        partname = name.split('_')
        agent_indicator = partname[0]

        with open(f'{storage_folder}/{agent_indicator}/{name}_evaluation_cost.json', 'r') as f:
            evaluation_running_cost = json.load(f)
        with open(f'{storage_folder}/{agent_indicator}/{name}_evaluation_reward.json', 'r') as f:
            evaluation_running_reward = json.load(f) 
        scaled_eval_cost_data = []
        scaled_eval_reward_data = []
        for i in range(eval_window,len(evaluation_running_cost)):
            scaled_eval_cost_data.append(np.mean(evaluation_running_cost[i-eval_window:i]))
            scaled_eval_reward_data.append(np.mean(evaluation_running_reward[i-eval_window:i]))
        
        scaled_eval_cost_data = sorted(scaled_eval_cost_data)
        scaled_eval_reward_data =  sorted(scaled_eval_reward_data)
        
        try:
            # Extracting total number of repairs data
            with open(f'data/custom_data/{agent_indicator}/{name}_component_data.json') as f:
                # Reading file line by line
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines]
            n_failed_arc, n_repaired_arc, preemptive_repair_data = data_list
            total_repairs = sum(n_repaired_arc)

            print(f'Agent {name}\n       cost std: {np.std(evaluation_running_cost)}\n       rewa std: {np.std(evaluation_running_reward)}')
            print(f'       cost mea: {np.mean(evaluation_running_cost)}\n       rewa mea: {np.mean(evaluation_running_reward)}\n       repa tot: {round(sum(n_repaired_arc),2)}')
        except:
            print(f'no component data found for {name}')
            print(f'Agent {name}\n       cost std: {np.std(evaluation_running_cost)}\n       rewa std: {np.std(evaluation_running_reward)}')
            print(f'       cost mea: {np.mean(evaluation_running_cost)}\n       rewa mea: {np.mean(evaluation_running_reward)}\n')
            pass
        scaled_data_dict[f'{name}_eval_cost'] = scaled_eval_cost_data
        scaled_data_dict[f'{name}_eval_reward'] = scaled_eval_reward_data

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'black', 'yellow']


        # Plot the second subplot
    plt.xlabel('Episode')
    plt.ylabel('Average Cost per Step')
    color_idx = 0
    legend_labels = []  # To store legend labels
    markers = ['o','s','D','p','*','X','h', 6, 7, '3', '4']
    for name in agent_names:

        # Splitting the name to get the identfier part
        partname = name.split('_')
        agent_indicator = partname[0]
            
        data = scaled_data_dict[f'{name}_eval_cost']
        # axes[0].plot(data, label=f'{name} (Mean: {np.mean(data):.2f})', color=colors[color_idx])
        # axes[0].scatter( [y for y in range(len(data))],data,s=2,label=f'{name} (Mean: {np.mean(data):.2f})', color=colors[color_idx])
        plt.scatter( [y for y in range(len(data))],data,s=15,color=colors[color_idx], marker=markers[color_idx])
        if agent_indicator == 'policy1':
            # legend_labels.append(r'$\Lambda_1$' + f' (Mean: {np.mean(data):.2f})')  # Add label to the legend_labels list
            legend_labels.append(r'$H_R$')
        elif agent_indicator == 'policy2':
            # legend_labels.append(r'$\Lambda_2$' + f' (Mean: {np.mean(data):.2f})')
            legend_labels.append(r'$H_P$')
        elif agent_indicator == 'via':
            # legend_labels.append(r'OSAS' + f' (Mean: {np.mean(data):.2f})')
            legend_labels.append(r'$OSAS$')
        elif agent_indicator[:2] == 'V1':
            legend_labels.append(r'$PIP$')
        elif agent_indicator[:2] == 'V2':
            legend_labels.append(r'$FIP_F$')
        else:
            # legend_labels.append(f'${name[0]}_{name[1]}$ (Mean: {np.mean(data):.2f})') 
            legend_labels.append(f'$FIP_V$') 
        color_idx += 1
    # axes[0].legend()
    plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.02, 0.65))  # Adjust the 'loc' and 'bbox_to_anchor' parameters as needed
    # plt.title('Evaluation: Cost')


    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f'{name_of_plot_file}.png')
    print('\nCreated figure and saved in current directory\n')

# create_custom_figure_eval('policy1','policy2','V1021','V2021','V30221', 'V30222', 'V30223',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='different_network_architecture')
# create_custom_figure_eval('policy1','policy2','via','V1022','V2021','V30224',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='normal_eval_all_policies')


def create_heatmap_system(agent_name:str , data_type: str, max_steps: int=360):
    """Takes in ratios for each component and returns a heatmap, for each arc
    Args:
        agent_name  (str): name of the agents data files(only identifier part needed)
        data_type   (str): defines what data is shown on the heatmap
                           failed_ratio             : heatmap for failed arcs(steps that arc was in failed state/ max_steps)
                           repaired_ratio           : heatmap for repair ratio of each arc (number of repairs/total repairs)
                           preemptive_repair_ratio  : heatmap for each arc if it was preemptively repaired or not (preemptive_repairs/total_repairs)
        max_steps   (int): max steps in the environment to calculate the ratios
                        """
    
    data_type_options = ['failed_ratio', 'repaired_ratio', 'preemptive_repair_ratio']

    # Get the data from the file
    nameparts = agent_name.split('_')
    agent_identifier = nameparts[0]
    with open(f'data/custom_data/{agent_identifier}/{agent_name}_component_data.json') as f:
        # Reading file line by line
        lines = f.readlines()
    data_list = [json.loads(line) for line in lines]
    n_failed_arc, n_repaired_arc, preemptive_repair_data = data_list

    # Define node information using a dictionary (scaled positions,)value is (x, y, sd) where sd is the demand/supply at that node and x,y are the coordinates relative to the graph
    node_stats = {
        "n01": (0, 0, -1),
        "n02": (1, 0, -1),
        "n03": (2, 0, -1),
        "n04": (3, 0, -1),
        "n05": (0, -1, -1),
        "n06": (1, -1, -1),
        "n07": (2, -1, 10),
        "n08": (3, -1, -1),
        "n09": (0, -2, -1),
        "n10": (1, -2, -1),
        "n11": (2, -2, -1),
        "n12": (3, -2, 10),
        "n13": (0, -3, -1),
        "n14": (1, -3, -1),
        "n15": (2, -3, -1),
        "n16": (3, -3, -1),
    }
    
    # Create a list of arcs (scaled positions)
    arc_stats = [
        ("arc0", "n01", "n02"),
        ("arc1", "n01", "n05"),
        ("arc2", "n02", "n03"),
        ("arc3", "n02", "n07"),
        ("arc4", "n04", "n08"),
        ("arc5", "n05", "n09"),
        ("arc6", "n06", "n07"),
        ("arc7", "n06", "n10"),
        ("arc8", "n07", "n08"),
        ("arc9", "n08", "n12"),
        ("arc10", "n09", "n13"),
        ("arc11", "n10", "n11"),
        ("arc12", "n10", "n13"),
        ("arc13", "n11", "n12"),
        ("arc14", "n11", "n15"),
        ("arc15", "n12", "n16"),
        ("arc16", "n13", "n14"),
        ("arc17", "n14", "n15"),
    ]

    # print(preemptive_repair_data)

    class Arc:
        def __init__(self, name, i, j):
            self.name = name
            self.connection = (i, j)
            self.n_failed = 0
            self.failed_ratio = 0
            self.n_repaired = 0
            self.repaired_ratio = 0
            self.preemptive_repaired = (0, 0)
            self.preemptive_repair_ratio = 0
            self.never_repaired = False

    arcs = [Arc(name, i, j) for (name, i, j) in arc_stats]

    # Assign the data to each arc to colorcode it in the heatmap
    total_repairs = sum(n_repaired_arc)

    # print(n_failed_arc)
    for i, arc in enumerate(arcs):
        arc.n_failed = n_failed_arc[i]
        arc.failed_ratio = arc.n_failed / max_steps
        arc.n_repaired = n_repaired_arc[i]
        arc.repaired_ratio = arc.n_repaired / total_repairs

        arc.preemptive_repaired = preemptive_repair_data[i]
        if sum(arc.preemptive_repaired) == 0:
            arc.preemptive_repair_ratio = 0 # catch 0 divide error; color this arc differently to show it has never been repaired
            arc.never_repaired = True
        else: 
            arc.preemptive_repair_ratio = arc.preemptive_repaired[1] / (sum(arc.preemptive_repaired))

    # Create graph object
    G = nx.Graph()

    # adding nodes to the graph
    for node, (x, y, sd) in node_stats.items():
        G.add_node(node, pos=(x,y), sd=sd)


    # Adding arcs TODO based on data type
        for arc in arcs:
            if data_type == 'failed_ratio':
                G.add_edge(arc.connection[0], arc.connection[1], weight=arc.failed_ratio)
                cmap = plt.get_cmap('RdYlGn')
                cmap = cmap.reversed()
            elif data_type == 'repaired_ratio':
                G.add_edge(arc.connection[0], arc.connection[1], weight=arc.repaired_ratio)
                cmap = plt.get_cmap('RdYlGn')
            elif data_type == 'preemptive_repair_ratio':
                G.add_edge(arc.connection[0], arc.connection[1], weight=arc.preemptive_repair_ratio)
                cmap = plt.get_cmap('RdYlGn')
            else:
                raise KeyError(f'Not a valid data type option. Valid options are {data_type_options}')

    # Extract arc weights to be used for coloring
    arc_weights = [G[edge[0]][edge[1]]['weight'] for edge in G.edges()]
    # print(arc_weights)

    # Create a colormap for the arcs based on heatmap values
    # cmap = plt.get_cmap('RdYlGn')

    # Draw the graph with nodes and colored arcs
    pos = nx.get_node_attributes(G, 'pos')
    for node in node_stats:
        i,j,sd = node_stats[node]
        if sd > 0:
            node_label ={}
            node_label[node] = node
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color='black')
            nx.draw_networkx_labels(G, pos, node_label, font_size=12, font_color="whitesmoke")
        else:
            node_label ={}
            node_label[node] = node
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color='white', node_shape='o', edgecolors='black')
            nx.draw_networkx_labels(G, pos, node_label, font_size=12, font_color="black")
    


    if data_type == 'repaired_ratio':
        edge_labels = {(arc.connection[0], arc.connection[1]): f'{(arc.repaired_ratio*100):.2f}%' for arc in arcs}
    elif data_type == 'preemptive_repair_ratio':
        edge_labels = {(arc.connection[0], arc.connection[1]): f'{(arc.preemptive_repair_ratio*100):.2f}%' for arc in arcs}
    elif data_type == 'failed_ratio':
        edge_labels = {(arc.connection[0], arc.connection[1]): f'{(arc.failed_ratio*100):.2f}%' for arc in arcs}
    # else:
    #     edge_labels = {(edge[0], edge[1]): f"{edge[2]['weight']:.2f}" for edge in G.edges(data=True)}

    for i, arc in enumerate(arcs):
        if arc.never_repaired == True:
            edge_labels[i] == 'xx'


    plt.axis('off')

    # determine title and draw edges with the appropiate color range
    if agent_identifier == 'policy1':
        titlename = f'$H_R$'
    elif agent_identifier == 'policy2':
        titlename = f'$H_P$'
    elif agent_identifier[:2] == 'V1':
        titlename = f'$PIP$'
    elif agent_identifier[:2] == 'V2':
        titlename = f'$FIP_F$'
    elif agent_identifier[:2] == 'V3':
        titlename = f'$FIP_V$'
    elif agent_identifier == 'via':
        titlename = '$OSAS$'

    if data_type == 'failed_ratio':
        nx.draw_networkx_edges(G, pos, width=5, edge_color=arc_weights, edge_cmap=cmap, edge_vmin=0.0, edge_vmax=(max(n_failed_arc)/max_steps))
        plt.title(f'{titlename}', pad=10, fontsize=17, fontweight='bold')
    elif data_type =='repaired_ratio':
        nx.draw_networkx_edges(G, pos, width=5, edge_color=arc_weights, edge_cmap=cmap, edge_vmin=0.0, edge_vmax=(max(n_repaired_arc)/max_steps))
        plt.title(f'{titlename}', pad=10, fontsize=17, fontweight='bold')
    elif data_type == 'preemptive_repair_ratio':
        nx.draw_networkx_edges(G, pos, width=5, edge_color=arc_weights ,edge_cmap=cmap, edge_vmin=0.0, edge_vmax=1.0)
        
        '''# Add a color bar to show the color range of the cmap
        if cmap is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
            sm.set_array([])
            cbar = plt.colorbar(sm, orientation='vertical')
            cbar.set_ticks([0.2, 0.8])
            cbar.ax.set_yticklabels(['Repair after Failure', 'Preemptive Repair'])'''
        
        plt.title(f'{titlename}', pad=10, fontsize=17, fontweight='bold')


    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,font_weight='bold', font_color='black')

    plt.gcf().set_size_inches(10,10)
    if data_type == 'failed_ratio':
        plt.savefig(f'{agent_name}_failure_ratio', dpi=1000, bbox_inches='tight') 
    elif data_type =='repaired_ratio':
        plt.savefig(f'{agent_name}_repair_ratio', dpi=1000, bbox_inches='tight') 
    elif data_type == 'preemptive_repair_ratio':
        plt.savefig(f'{agent_name}_preemptive_repair_ratio', dpi=1000, bbox_inches='tight') 
    
    print(f'Created graph for agent: {agent_name} for data type: {data_type}')
    
    # plt.show()

def create_state_graph(state:int):
    """ Takes a int or bit state and returns a png with the state of the system, where red arcs are broken and blue arcs are functional
    Args:
        state (int/list): state that is to be displayed
    
    Returns: 
        png (file): saves the png file to the current directory  
        """
    
    # Define node information using a dictionary (scaled positions); value is (x, y, sd) where sd is the demand/supply at that node and x,y are the coordinates relative to the graph
    node_stats = {
        "n01": (0, 0, -1),
        "n02": (1, 0, -1),
        "n03": (2, 0, -1),
        "n04": (3, 0, -1),
        "n05": (0, -1, -1),
        "n06": (1, -1, -1),
        "n07": (2, -1, 10),
        "n08": (3, -1, -1),
        "n09": (0, -2, -1),
        "n10": (1, -2, -1),
        "n11": (2, -2, -1),
        "n12": (3, -2, 10),
        "n13": (0, -3, -1),
        "n14": (1, -3, -1),
        "n15": (2, -3, -1),
        "n16": (3, -3, -1),
    }
    # Create a list of arcs (scaled positions)
    arc_stats = [
        ("arc0", "n01", "n02"),
        ("arc1", "n01", "n05"),
        ("arc2", "n02", "n03"),
        ("arc3", "n02", "n07"),
        ("arc4", "n04", "n08"),
        ("arc5", "n05", "n09"),
        ("arc6", "n06", "n07"),
        ("arc7", "n06", "n10"),
        ("arc8", "n07", "n08"),
        ("arc9", "n08", "n12"),
        ("arc10", "n09", "n13"),
        ("arc11", "n10", "n11"),
        ("arc12", "n10", "n13"),
        ("arc13", "n11", "n12"),
        ("arc14", "n11", "n15"),
        ("arc15", "n12", "n16"),
        ("arc16", "n13", "n14"),
        ("arc17", "n14", "n15"),
    ]

    # Handling the given state apply values to all graph arcs
    if type(state) is int: 
        intstate = state
        display_state = list(bits(state, 18))
        for i, element in enumerate(display_state):
            display_state[i] = int(element)
    elif type(state) is list:
        display_state = state[:18]
        charbitstate = state[:18]
        for i, element in enumerate(state):
            charbitstate[i] = str(element)
        intstate = bit_val(charbitstate)
    else: 
        raise TypeError('State has to be int or list in the format [0, 1, ...]')

    # display_state.reverse()

    # Define arc class
    class Arc:
        def __init__(self, name, i, j):
            self.name = name
            self.connection = (i, j)
            self.failed = 0

    # Add all arcs according to the arc list
    arcs = [Arc(name, i, j) for (name, i, j) in arc_stats]

    # Set state of each arc based on the given state
    for i, arc in enumerate(arcs):
        arc.failed = display_state[i]

    # Create graph object
    G = nx.Graph()

    # adding nodes to the graph
    for node, (x, y, sd) in node_stats.items():
        G.add_node(node, pos=(x,y), sd=sd)


    # Adding arcs
    for arc in arcs:
        G.add_edge(arc.connection[0], arc.connection[1], weight=arc.failed)

    # Extract arc weights to be used for coloring
    arc_weights = [G[edge[0]][edge[1]]['weight'] for edge in G.edges()]
    # Define a color map
    color_map = {0: 'black', 1: 'red'}
    edge_colors = [color_map[G[u][v]['weight']] for u, v in G.edges()]

    # Draw the graph with nodes and colored arcs
    pos = nx.get_node_attributes(G, 'pos')
    for node in node_stats:
        i,j,sd = node_stats[node]
        if sd > 0:
            node_label ={}
            node_label[node] = node
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color='black')
            nx.draw_networkx_labels(G, pos, node_label, font_size=12, font_color="whitesmoke")
        else:
            node_label ={}
            node_label[node] = node
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color='white', node_shape='o', edgecolors='black')
            nx.draw_networkx_labels(G, pos, node_label, font_size=12, font_color="black")
    

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=3)
    plt.title(f'State: {intstate}', pad=10, fontsize=14, fontweight=10)
    plt.gcf().set_size_inches(10,10)
    plt.savefig(f'state_{intstate}_graph', dpi=1000, bbox_inches='tight')

    print(f'Created Graph for state: {state}')


def solve_for_state(intstate:int):
    """solves the model for a specific state and returns the optimal flow on each arc
    Args:
        int_state  (int): an int state for the Alderson Model
        """
    import pyomo.environ as pyo
    import time
    from System.Alderson_2015_full.fuelnet_model_2022 import build_model, print_results
    
    node_file = 'System/Alderson_2015_modified/node_data.csv'
    arc_file = 'System/Alderson_2015_modified/arc_data.csv'

    # get node data
    node_data_df = pd.read_csv(node_file, index_col=['node'])
    nodes = list(node_data_df.index.unique())

    # get arc data
    edges_df = pd.read_csv(arc_file)
    # set the index to be ['i','j']
    edges_df = edges_df.set_index(['i','j'])
    edges = list(edges_df.index.values)
    num_components = len(edges)

    antiedges = [ (j,i) for (i,j) in edges]
    arcs = edges + antiedges
    
    # Building model:
    model = build_model(nodes, edges, arcs, node_data_df, edges_df)
    opt = pyo.SolverFactory("cbc")

    # set the state of the system to that specified
    bitstate = bits(intstate,num_components)
    model.set_state(bitstate)

    start_time = time.perf_counter()
    results = opt.solve(model, tee=True)
    end_time = time.perf_counter()
    print('Took', end_time - start_time, 'seconds to solve the model.')
    print ("The solver returned a status of:"+str(results.solver.status))
    print_results(model)

# solve_for_state(0)

# create_custom_figure_eval('policy1','policy2','via','V1022','V2021','V30223',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='norm_eval_all_policies')
# create_custom_figure_eval('policy1_surpriseattrition','policy2_surpriseattrition','via_surpriseattrition','V1022_surpriseattrition','V2021_surpriseattrition','V30223_surpriseattrition',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='surp_eval_all_policies')
# create_custom_figure_eval('V1021_surpriseattrition','V1022_surpriseattrition','V1023_surpriseattrition','V1024_surpriseattrition',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='surp_eval_v1_policies')
# create_custom_figure_eval('V2021_surpriseattrition','V2022_surpriseattrition',storage_folder='data/custom_data', eval_window=1, name_of_plot_file='surp_eval_v2_policies')
# create_custom_figure_eval('V30221','V30222','V30223', storage_folder='data/custom_data', eval_window=1, name_of_plot_file='norm_eval_v3_policies')
# create_custom_figure_eval('policy1_surpriseattack','policy2_surpriseattack','via_surpriseattack', 'V4024', storage_folder='data/custom_data', eval_window=1, name_of_plot_file='surp-ack-all-policies')

# create_heatmap_system('policy1', 'failed_ratio',360)
# agentnames = ['policy1','policy2','via','V1021','V1022','V2021','V30223']
# agentnames = ['V2021', 'V30223']
# for name in agentnames: 
#     create_heatmap_system(f'{name}_surpriseattrition', 'failed_ratio',360)
#     create_heatmap_system(f'{name}', 'failed_ratio',360)
#     create_heatmap_system(f'{name}_surpriseattrition', 'repaired_ratio',360)
#     create_heatmap_system(f'{name}', 'repaired_ratio',360)
#     create_heatmap_system(f'{name}_surpriseattrition', 'preemptive_repair_ratio',360)
#     create_heatmap_system(f'{name}', 'preemptive_repair_ratio',360)
# create_heatmap_system('V1021_surpriseattrition', 'preemptive_repair_ratio',360)
    
# create_custom_figure('V1021', 'V2021','V3021', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='15k')
# create_custom_figure('V1031', 'V2031','V3031', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='3repaircrews')
# create_custom_figure('V3027', 'V3028','V3029','V30210', 'V30211', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='V3-diff_starting_states')
# create_custom_figure('V4021', 'V4022','V4023','V4024', storage_folder='data/V4', training_window=100, eval_window=10, name_of_plot_file='V4-diff_starting_states')
# create_custom_figure('V30212', 'V30215', 'V30214','V30213', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='V3-diff_batch_size')
# create_custom_figure('V30212', 'V30216','V3021', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='V3-diff_replacement_cost')
# create_custom_figure('V30218', 'V30215', storage_folder='data/V3', training_window=100, eval_window=10, name_of_plot_file='V3-diff_network_structure')

# create_custom_figure_eval('via_surpriseattack', 'V4024','V4025', 'policy2_surpriseattack', 'policy1_surpriseattack',storage_folder='data/custom_data', training_window=1, eval_window=1, name_of_plot_file='surpriseattackeval')
# create_custom_figure_eval('V3021', 'V1022', storage_folder='data/custom_data', training_window=1, eval_window=1, name_of_plot_file='surpriseeval')
# create_custom_figure_eval('V30217','V30218','policy2', 'via', storage_folder='data/custom_data', training_window=5, eval_window=10, name_of_plot_file='Normalattrition_eval')
# create_custom_figure_eval('V30212','policy1_2', 'policy2_2', 'via', storage_folder='data/custom_data', training_window=1, eval_window=1, name_of_plot_file='normal_eval_1replacement_cost')
# create_custom_figure_eval('V30212','V30212_1', 'V30212_5', 'V30216', storage_folder='data/custom_data', training_window=1, eval_window=1, name_of_plot_file='normal_eval_diff_replacement_cost_2')
# create_custom_figure_eval('V1026','policy1', 'policy2', 'via', 'V30218','V2025',storage_folder='data/custom_data', training_window=10, eval_window=10, name_of_plot_file='normal_eval_low_attrition_rate')
# create_custom_figure_eval('V1026_surpriseattrition', 'policy1_surpriseattrition', 'policy2_surpriseattrition', 'via_surpriseattrition', 'V30218_surpriseattrition', 'V2025_surpriseattrition',storage_folder='data/custom_data', training_window=10, eval_window=10, name_of_plot_file='Surpriseattrition_eval_low_attrition')
# create_custom_figure_eval('policy2', 'V30218',storage_folder='data/custom_data', training_window=10, eval_window=10, name_of_plot_file='normal_eval_pol2vV30218')
# create_custom_figure_eval('policy2','V1021','V30221', 'V30222', 'V30223',storage_folder='data/custom_data', training_window=10, eval_window=10, name_of_plot_file='different_network_architecture')
# create_custom_figure_eval('policy2_surpriseattrition','V1021_surpriseattrition','V30221_surpriseattrition', 'V30222_surpriseattrition', 'V30223_surpriseattrition',storage_folder='data/custom_data', training_window=10, eval_window=10, name_of_plot_file='different_network_architecture_surpriseattrition')

def combine_via_csv_files(outputfilename):

    try:
        with open(f'data/via/{outputfilename}.csv'):
            input = input(f'filename: {output_file} already exists. Press y to continue')
    except:
        input = 'y'

    if input == 'y' or 'Y':
        # max int states 
        filecounter = 0
        for i in range(262143+1):
            try:
                with open(f'data/via/Run2/Splitfiles/via{i}.csv', 'r', newline='') as readfile:
                    with open(f'data/via/Run2/{outputfilename}.csv', 'a', newline='') as output_file:
                        csv_reader = csv.reader(readfile)
                        csv_writer = csv.writer(output_file)

                        # Skip header in input file, unless its first file
                        if filecounter > 0: 
                            next(csv_reader, None)
                        
                        # Iterate over the rows in the input file and write them to the output file
                        for row in csv_reader:
                            csv_writer.writerow(row)

                filecounter += 1
            except:
                pass

# combine_via_csv_files('Run2_via_full')

def get_bit_val_from_str(connections: list) -> int:
    """Takes two node str numbers and returns the idx in the state list"""

    arc_stats = [
        ("arc0", "1", "2"),
        ("arc1", "1", "5"),
        ("arc2", "2", "3"),
        ("arc3", "2", "7"),
        ("arc4", "4", "8"),
        ("arc5", "5", "9"),
        ("arc6", "6", "7"),
        ("arc7", "6", "10"),
        ("arc8", "7", "8"),
        ("arc9", "8", "12"),
        ("arc10", "9", "13"),
        ("arc11", "10", "11"),
        ("arc12", "10", "13"),
        ("arc13", "11", "12"),
        ("arc14", "11", "15"),
        ("arc15", "12", "16"),
        ("arc16", "13", "14"),
        ("arc17", "14", "15"),
    ]

    bitstate = []

    for arc in arc_stats:
        for element in connections:
            i = element[0]
            j = element[1]

            if i == arc[1] and j == arc[2]:
                bitstate.append(1)
                break
                
        
            bitstate.append(0)

    return bitstate

# print(bit_val('000001010000000000'))



def component_visual_data(initial_failure_rate: int, attrition_rate: int, episodes: int, surprise_attrition_rate: int=0.01):
    """takes in an initial failure probability and attrition rate and returns a graph that depicts the failure rate over
    the specified episodes of a component
    
    Args: 
        initial_failure_rate   (int): the initial failure rate of each component in the environment
        attrition_rate         (int): attrition rate that the component suffers at each timestep
        episodes               (int): episodes over which to visualize the current failure rate"""
    
    x = [i for i in range(0,episodes,10)]
    l = [i for i in range(episodes)]
    print(x)
    y = []
    z = []
    
    i = initial_failure_rate
    probability_of_not_breaking = [i]
    probability_of_not_breaking_surprise = [i]
    surp = initial_failure_rate

    for ep in range(episodes):
        if ep % 10 == 0:
            if i >= 0:
                y.append(i)
            else:
                y.append(0)
        if ep < 50:
            if ep % 10 == 0:
                if surp >= 0:
                    z.append(surp)
                else:
                    z.append(0)
            surp -= attrition_rate
        else:
            surp -= surprise_attrition_rate
            if ep % 10 == 0:
                if surp >= 0:
                    z.append(surp)
                else:
                    z.append(0)

        
        i -= attrition_rate
        probability_of_not_breaking.append(probability_of_not_breaking[ep]*i)

    surp = initial_failure_rate
    for ep in range(episodes):
        probability_of_not_breaking_surprise.append(probability_of_not_breaking_surprise[ep]*surp)
        surp -= surprise_attrition_rate
    
    probability_of_not_breaking.pop()
    probability_of_not_breaking_surprise.pop()

    # Filter the data for y and z that are >= 0
    x_filtered = []
    y_filtered = []
    z_filtered = []

    for xi, yi, zi in zip(x, y, z):
        if yi >= 0 and zi >= 0:
            x_filtered.append(xi)
            y_filtered.append(yi)
            z_filtered.append(zi)
        

    # plt.scatter(x_filtered, y_filtered, s=3, c='green', alpha=1, label='normal')
    # plt.scatter(x_filtered, z_filtered, s=3, c='red', alpha=1, label='surprise')
    # plt.scatter(l, probability_of_not_breaking, s=3, c='green', alpha=1, label='no surprise')
    plt.bar(l, probability_of_not_breaking)
    plt.axhline(y=0.5, color='red')
    # plt.scatter(l, probability_of_not_breaking_surprise, s=3, c='red', alpha=1, label='surprise')
    plt.xlabel('Step')
    plt.ylabel('P')
    # plt.title('Component Status')
    # plt.legend()
    plt.show()

# component_visual_data(0.995, 0.005, 50)


def show_states_visited_data(agent_name:str, show_as:str='heatmap'):
    """takes the data from the data folder and creates a heatmap with all states colored by how often they have been visited on average
    Args:
        agent_name (str): str name of the agent to identify the correct pkl file
        show_as    (str): how to show the data, valid options are (heatmap, barchart)
    
    Returns:
        heatmap    (file): heatmap image of all states
        """
    
    import pickle
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    from sklearn.preprocessing import QuantileTransformer
    
    # Load the data
    with open(f'data/custom_data/{agent_name}/{agent_name}_states_visited_data.pkl', 'rb') as f:
        states_visited = pickle.load(f)
    print(f'Loaded data from: data/custom_data/{agent_name}/{agent_name}_states_visited_data.pkl')
    
    # Get the number of states (assuming contiguous indices from 0 to 200)
    num_states = max(states_visited.keys()) + 1

    if show_as == 'heatmap':

        # Calculate the average visits for each state
        print('Calculating Matrix')
        average_visits = [np.mean(states_visited[state]) for state in range(num_states)]
        average_visits = [round(x,3) for x in average_visits]
        array = np.array(average_visits)


        # Quantile Transformation
        print('Normal')
        transformer = QuantileTransformer(output_distribution='uniform')
        quantile_transformed = transformer.fit_transform(array.reshape(-1, 1)).flatten()
        print('Normalized Matrix')

        # Create a 1D array and reshape it into a 2D square matrix
        result_matrix = quantile_transformed.reshape(int(np.sqrt(num_states)), -1)

        # Normalize the matrix using Z-score
        # result_matrix_clipped = np.clip(result_matrix, a_min=None, a_max=1)
        # normalized_matrix = (result_matrix - np.mean(result_matrix)) / np.std(result_matrix)

        # # Define the colormap
        # colors = ['white', 'yellow','orange', 'orangered', 'maroon','black']
        # n_bins = [-5,0,1,2,3,5,10000]
        # cmap_name = 'custom_colormap'
        # custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(n_bins))
        # # Create a BoundaryNorm to define the edges of the bins
        # norm = BoundaryNorm(n_bins, len(colors)-1, clip=False)

        # cmap = plt.get_cmap('YlOrRd')
        # cmap.set_under('white')
        print('Showing Matrix')
        plt.imshow(result_matrix, cmap='viridis')
        plt.colorbar(label='Average Visits')
        plt.title('Heatmap of Average Visits for States')
        plt.show()

    elif show_as == 'barchart':
        # Average all values in the dict
        averaged_dict = {key: sum(values) / len(values) for key, values in states_visited.items()}
        # Filter out all states that have never been visisted
        # filtered_dict = {key: value for key, value in averaged_dict.items() if value != 0}

        average_visits = [np.mean(states_visited[state]) for state in range(num_states)]
        average_visits = [int(x) for x in average_visits]

        # plt.bar(range(len(average_visits)), average_visits)
        plt.bar(list(averaged_dict.keys())[:1000], average_visits[:1000])
        plt.xlabel('State')
        plt.ylabel('Average Visits')
        plt.title(f'Average Visits Per State: {agent_name}')
        plt.show()

    else:
        valid_options = ['heatmap', 'barchart']
        raise ValueError(f"{show_as} option not valid. Available options are: {valid_options}")


# show_states_visited_data('policy2', 'barchart')
# create_state_graph(33)
# create_state_graph(3078)
# x = 131072
# for i in range (20):
#     create_state_graph(x)
#     # x += 512
#     x +=1
# create_state_graph(16512)
# create_state_graph(18448)
# create_state_graph(18452)
# create_state_graph(26644)
# create_state_graph(131857)
# create_state_graph(131905)
# create_state_graph(131907)
# create_state_graph(131969)
# create_state_graph(131745)
# create_state_graph(131713)
# create_state_graph(131714)
# create_state_graph(131715)
# create_state_graph(412)
# create_state_graph(262043)


def show_broken_pair_matrix(agent_name:str):
    """Shows a heatmap of the matrix with the frequency of all broken component pairs"""

    # Get the data from the file
    nameparts = agent_name.split('_')
    agent_id = nameparts[0]

    broken_component_matrix = np.load(f'data/custom_data/{agent_id}/{agent_name}_broken_pair_matrix.npy')

    # set the diagonal values to zero in order to not mess with the distribution
    for i in range(18):
        broken_component_matrix[i][i] = 0

    row_labels = ['(n01,n02)',
                  '(n01,n05)',
                  '(n02,n03)',
                  '(n02,n07)',
                  '(n04,n08)',
                  '(n05,n09)',
                  '(n06,n07)',
                  '(n06,n10)',
                  '(n07,n08)',
                  '(n08,n12)',
                  '(n09,n13)',
                  '(n10,n11)',
                  '(n10,n13)',
                  '(n11,n12)',
                  '(n11,n15)',
                  '(n12,n16)',
                  '(n13,n14)',
                  '(n14,n15)']

    # Sort the matrix to add the highest failure components to the top of y axis and most left side of x axis
    # The failure rate is the variable they are sorted by

    # Lets first import the failure rate for all components
    with open(f'data/custom_data/{agent_id}/{agent_name}_component_data.json') as f:
        # Reading file line by line
        lines = f.readlines()
    data_list = [json.loads(line) for line in lines]
    n_failed_arc, n_repaired_arc, preemptive_repair_data = data_list

    # Next we sort this data by value but instead of listing the raw values we sort the indexes
    # Enumerate the original list to keep track of the indexes
    enumerated_list = list(enumerate(n_failed_arc))

    # Use sorted to get the sorted list of indexes based on the values
    sorted_indexes = [index for index, value in sorted(enumerated_list, key=lambda x: x[1], reverse=True)]

    # Now with the sorted indexes we will fill a new matrix from top to bottom with the labels and indexes from our sorted list
    sorted_matrix = []
    sorted_labels = []
    for i in sorted_indexes:
        sorted_matrix.append(broken_component_matrix[i])
        sorted_labels.append(row_labels[i])
    
    # We now have to also sort the columns.... This is a bit more tricky
    final_sorted_matrix = [[] for x in range(len(sorted_indexes))]
    for index in sorted_indexes:
        for i,row in enumerate(sorted_matrix):
            final_sorted_matrix[i].append(sorted_matrix[i][index])
        
    # Defining the image
    plt.imshow(final_sorted_matrix, cmap='viridis_r', interpolation='nearest')

    # Define the row to be highlighted
    highlighted_rows = [sorted_labels.index('(n09,n13)'), sorted_labels.index('(n07,n08)')]

    # Highlight the specified row by adding a red border
    for i in highlighted_rows:
        plt.gca().add_patch(plt.Rectangle((-0.5, i - 0.5), len(row_labels), 1, fill=False, edgecolor='red', linewidth=4))

    # Add custom row labels
    plt.yticks(np.arange(len(sorted_labels)), sorted_labels)

    # Add custom column labels
    plt.xticks(np.arange(len(sorted_labels)), sorted_labels, rotation='vertical')

    # Add a colorbar for reference
    plt.colorbar()

    # Show the plot
    plt.show()

# show_broken_pair_matrix('V2021_surpriseattrition')

def violin_distribution_plots(*agent_names:str, storage_folder:str='data/custom_data'): 
    import seaborn as sns
    import matplotlib.pyplot as plt

    distribution_names = []
    names_map = {
        'policy1': '$H_R$',
        'policy2': '$H_P$',
        'via': '$OSAS$',
        'V1': '$PIP$',
        'V2': '$FIP_F$',
        'V3': '$FIP_V$'
    }
    data = []
    for name in agent_names:

        # Splitting the name to get the identfier part
        partname = name.split('_')
        agent_indicator = partname[0]

        if agent_indicator in ['policy1', 'policy2', 'via']:
            distribution_names.append(names_map[agent_indicator])
        else: 
            distribution_names.append(names_map[agent_indicator[:2]])

        with open(f'{storage_folder}/{agent_indicator}/{name}_evaluation_cost.json', 'r') as f:
            evaluation_running_cost = json.load(f)
        
        data.append(np.array(evaluation_running_cost))
        print(evaluation_running_cost)

    # Example data (replace this with your own data)
    # distribution_names = ['Distribution 1', 'Distribution 2', 'Distribution 3']
    # means = [10, 15, 20]
    # std_devs = [2, 3, 4]

    # Generate random data for the example
    # data = [np.random.normal(mean, std_dev, 100) for mean, std_dev in zip(means, std_devs)]
    print(data)

    # Reshape the data for seaborn
    reshaped_data = np.concatenate([np.vstack(data[i]).T for i in range(len(data))])
    distribution_labels = np.repeat(distribution_names, 100)
    print(data)

    # Create a DataFrame for seaborn
    df = pd.DataFrame({'Policy': distribution_labels, 'Cost per Step': reshaped_data.flatten()})

    # Create a violin plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Policy', y='Cost per Step', data=df, inner='quartile')
    plt.title('Surprise Condition Performance')
    plt.show()

# violin_distribution_plots('policy1','policy2','via','V1022','V2021','V30223')
# violin_distribution_plots('policy1_surpriseattrition','policy2_surpriseattrition','via_surpriseattrition','V1022_surpriseattrition','V2021_surpriseattrition','V30224_surpriseattrition')
# violin_distribution_plots('policy1')