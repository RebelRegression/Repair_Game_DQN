import sys
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import time


def build_model(nodes, edges, arcs, node_data_df, edge_data_df):
    #Define model type
    model = pyo.ConcreteModel()

    # hold on to the arguments
    model.nodes = nodes
    model.edges = edges
    model.arcs = arcs 

    # index sets
    model.N = pyo.Set(initialize=nodes)
    model.E = pyo.Set(initialize=edges)
    model.A = pyo.Set(within=model.N*model.N,initialize=arcs)
    print(model.N)
    
    #Create model for costs
    c = edge_data_df['cost'].to_dict()
    u = edge_data_df['capacity'].to_dict()
    q = {}
    xhat = edge_data_df['xhat'].to_dict()

    # because the model uses directed arcs, not undirected edges,
    # we need to create data for the anti-parallel arcs
    for (i,j) in arcs: #list(u.keys()):
        c[j,i] = c[i,j]
        u[j,i] = u[i,j]
        q[i,j] = q[j,i] = 10
        xhat[j,i] = xhat[i,j]
    model.c = pyo.Param(model.N,model.N, initialize=c)
    model.u = pyo.Param(model.N,model.N, initialize=u)
    model.q = pyo.Param(model.N,model.N, initialize=q)
    model.xhat = pyo.Param(model.N,model.N, initialize=xhat,mutable=True)

    # node parameters
    d = node_data_df['supply'].to_dict()
    model.d = pyo.Param(model.N, initialize=d)

    # update 7/10/22: make node penalty data
    #p = {}
    #for i in nodes:
    #    p[i] = 10
    p = node_data_df['p'].to_dict()
    model.p = pyo.Param(model.N, initialize=p)

        
    #Flow variable
    model.Y = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    
    #Define fuel shortfall at Node
    model.S = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    
    #objective function
    def obj_expression(model):
        unmet_penalty = sum(model.p[n]*model.S[n] for n in model.N)
        flow_cost = sum((model.c[i,j] + model.q[i,j]*model.xhat[i,j])*model.Y[i,j] for (i,j) in model.A)
        return unmet_penalty + flow_cost 

    model.OBJ = pyo.Objective(rule=obj_expression, sense = pyo.minimize)

    # subject to:
    # flow balance (goes-inna, goes-outta) constraint
    def flow_constraint_rule(model,n):
        flow_out = sum([model.Y[n,j] for j in model.N if (n,j) in model.A])
        flow_in =  sum([model.Y[i,n] for i in model.N if (i,n) in model.A])
        
        return flow_out - flow_in - model.S[n]  <= model.d[n]

    print("Building constraint: flow_constraint...",end='')
    model.flow_constraint = pyo.Constraint([(n) for n in nodes], rule=flow_constraint_rule)
    print('done.')


    #total edge capacity contraint
    def total_arc_capacity_rule(model,i,j):
        return model.Y[i,j] + model.Y[j,i] <= model.u[i,j] * (1 - model.xhat[i,j])

    print("Building constraint: total_arc_capacity_constraint...",end='')
    model.total_arc_capacity_constraint = pyo.Constraint([(i,j) for (i,j) in arcs], rule=total_arc_capacity_rule)
    print('done.')


    def get_state():
        '''returns a bitstring corresponding to network state'''
        state = ''
        for (i,j) in sorted(model.edges):
            state += ('%1s' % model.xhat[i,j].value)
        return state
    # end of get_state
    model.get_state = get_state

    def set_state(bitstate):
        '''given a bitstring, set state of individual components'''
        index = 0
        for (i,j) in sorted(model.edges):
            model.xhat[i,j] = model.xhat[j,i] = int(bitstate[index])
            index += 1

        newstate = get_state()
        if newstate != bitstate:
            print("Epic fail!")
    # end of set_state
    model.set_state = set_state


    return model

def print_results(model):

    for n in sorted(model.N):
        #print n, G.node[n]
        #if 'supply' in G.node[n]:
        #  print '\tsupply is:', G.node[n]['supply']
        print("Node %s: supply = %.1f" % (n,model.d[n]), end='')
        if model.d[n] < 0:
            print(", shortfall = %.1f" % (model.S[n].value))
        else:
            print()

        # FIX THIS!!!  How to get a value from a sum object?
        flow_out = sum([model.Y[n,j] for j in model.N if (n,j) in model.A])
        flow_in =  sum([model.Y[i,n] for i in model.N if (i,n) in model.A])

        print("\tFlow Out: %s" % pyo.value(flow_out))
        for (i,j) in sorted(model.A):
            if n == i:
                print("\tArc(%s,%s): flow = %.1f, capacity= %.1f, cost= %.1f, xhat= %s" % \
                    (i,j,model.Y[i,j].value,model.u[i,j],model.c[i,j],model.xhat[i,j].value) ) 
        print("\tFlow In: %s" % pyo.value(flow_in))
        for (i,j) in sorted(model.A):
            if n == j:
                print("\tArc(%s,%s): flow = %.1f, capacity= %.1f, cost= %.1f, xhat= %s" % \
                    (i,j,model.Y[i,j].value,model.u[i,j],model.c[i,j],model.xhat[i,j].value) ) 


    print("Objective function value: %.1f" % model.OBJ())

if __name__ == "__main__":

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

    isVerbose = True

    if isVerbose:
        solve_time = time.clock()
        results = opt.solve(model, tee=True)
        print('Took', time.clock() - solve_time, 'seconds to solve the model.')
        print ("The solver returned a status of:"+str(results.solver.status))
    else:
        results = opt.solve(model, tee=False)


    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        # this is feasible and optimal
        solution = model.OBJ()

        if isVerbose:
            print("Optimal Solution Found!!")
            print("Objective function value: %.1f" % solution)
            print_results(model)


    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # do something about it? or exit?
        print("Problem is INFEASIBLE!!")

    else:
        # something else is wrong
        print (results.solver)
