import sys
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import time
from fuelnet_model_2022 import build_model, print_results

"""
Descriptions:
- Contains code from Alderson et al. (2015)
- Evaluates the objective function value for the underlying flow model for each possible state in the state space
- Creates a csv file where the cost for each state is associated with the int state representation
"""


# def get_state(model):
#     '''returns a bitstring corresponding to network state'''
#     state = ''
#     for (i,j) in sorted(edges):
#         state += ('%1s' % model.xhat[i,j].value)
#     return state

# def set_state(model,bitstate):
#     '''given a bitstring, set state of individual components'''
#     index = 0
#     for (i,j) in sorted(edges):
#         model.xhat[i,j] = model.xhat[j,i] = int(bitstate[index])
#         index += 1

#     newstate = get_state(model)
#     if newstate != bitstate:
#         print("Epic fail!")

def bits(a,b):
  '''Given an integer (a) and a number of bits (b), return the bitstring'''
  res=bin(a+2**b)
  if a<0:
    return res[2:]
  else:
    return res[3:]

def bit_val(s):
    '''Given a string of 0-1 chars ('bits'), returns decimal value.'''
    val = 0
    e = 0
    while e < len(s):
        val += int(s[-1-e]) * 2**e
        e += 1
    return val

def solve_it(model, soln_dict, intstate, isVerbose=False):
    '''solve the given model for the given state (intstate),
       but only execute if solution does not already exist in soln_dict'''
    bitstate = bits(intstate,num_components)
    if isVerbose:
        print("\nAttempting to Solve for state %s ..." % bitstate)
    #intstate = bit_val(bitstate)

    # if the solution exists, return it, maybe with a warning
    if intstate in soln_dict:
        print("Solution for %s already exists.  P(%s)=%.1f.  Why did you ask?" \
            % (bitstate, intstate, soln_dict[intstate]))
        return model, soln_dict[intstate]

    # set the state of the system to that specified
    model.set_state(bitstate)

    if isVerbose:
        solve_time = time.clock()
        results = opt.solve(model, tee=True)
        print('Took', time.clock() - solve_time, 'seconds to solve the model.')
        print ("The solver returned a status of:"+str(results.solver.status))
    else:
        results = opt.solve(model, tee=False)


    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        # this is feasible and optimal
        #model.solutions.load_from(results)
        solution = model.OBJ()
        soln_dict[intstate] = solution

        if isVerbose:
            print("Optimal Solution Found!!")
            print("Objective function value: %.1f" % solution)

        return model, solution

    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # do something about it? or exit?
        print("Problem is INFEASIBLE!!")

    else:
        # something else is wrong
        print("Something else is wrong...!")
        print (results.solver)




if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nUSAGE: Command line arguments required.")
        print("    python %s [nodefile.csv] [arcfile.csv]\n" % sys.argv[0])
        sys.exit()


    print("\n\nLoading data about network model...\n")
    #nodefile = 'node_data.csv'
    nodefile = sys.argv[1]
    #arcfile = 'arc_data.csv'
    arcfile = sys.argv[2]

    # get node data
    node_data_df = pd.read_csv(nodefile, index_col=['node'])
    nodes = list(node_data_df.index.unique())
    print('There are %d nodes:' % len(nodes))
    print(nodes)

    # get arc data
    edges_df = pd.read_csv(arcfile)
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

    opt = pyo.SolverFactory("cbc")

    isVerbose = False

    print("\nReading file of past solves...", end='')
    states_df = pd.read_csv('states.csv', index_col=['state'])
    soln_dict = states_df['performance'].to_dict()
    print("done.  Read %d entries." % len(states_df))

    # get the largest index value already solved
    maxindex = int(max(states_df.index.unique()))
    print('Largest index found: %d' % maxindex)

    bitstate = bits(maxindex,num_components)
    print("The state of system: %s (%d), performance = %.1f" % (bitstate,maxindex,soln_dict[maxindex]))

    stepsize = 100000
    largest_state = 2**num_components
    print("Largest state is %d" % largest_state)

    for state in range(maxindex,min(maxindex+stepsize,largest_state)):

        model, perf = solve_it(model, soln_dict, state, isVerbose)

        bitstate = model.get_state()
        intstate = bit_val(bitstate)
        #print("The state of system: %s (%d), performance = %.1f" % (bitstate,intstate,perf))

        # periodically, report the progress and save it to the csv file
        if state % 1000 == 0:
            print("\nSolved for state = %d" % state)
            print("\nDictionary now has %d entries." % len(soln_dict))
            #print(soln_dict)
            new_df = pd.DataFrame.from_dict(soln_dict,orient='index')
            new_df.columns = ['performance']
            new_df.index.names = ['state']
            new_df.to_csv('states.csv')

    print("\n\nLoop Complete!! Writing final results...")
    print("\nDictionary now has %d entries." % len(soln_dict))
    #print(soln_dict)
    new_df = pd.DataFrame.from_dict(soln_dict,orient='index')
    new_df.columns = ['performance']
    new_df.index.names = ['state']
    new_df.to_csv('states.csv')
