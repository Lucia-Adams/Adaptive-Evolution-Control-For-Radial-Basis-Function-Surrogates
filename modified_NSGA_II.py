import numpy as np
from problem_wrapper import pymooProblem
from reproblem import *

# pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.dominator import Dominator

from pymoo.problems import get_problem
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from scipy.interpolate import RBFInterpolator

import matplotlib.pyplot as plt
import random

def new_binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)



problem = pymooProblem(RE21())
static_problem = problem.get_basic_problem()
n_objectives = problem.get_n_objectives()
n_variables = problem.get_n_variables()

# Uses Latin Hyper Cube Sampling as in most papers
# SBX crossover (can modify probability of crossover etc) and polynomial mutation as in Yu M
algorithm = NSGA2(pop_size=100, 
                sampling=LHS(), 
                selection=TournamentSelection(func_comp=new_binary_tournament),
                crossover=SBX(eta=15, prob=0.9), 
                mutation=PolynomialMutation(),
                survival=RankAndCrowding())


# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
algorithm.setup(problem, termination=('n_gen', 20), seed=1, verbose=False)

# To store points/individuals that have used the actual fitness evaluation
evaluated_archive = []
rbf_models=[]

plot = Scatter()

# until the algorithm has no terminated - we have set to when gets to certain number of generations in algorithm setup tuple
while algorithm.has_next():

    # ask the algorithm for the next solution to be evaluated
    # Returns the pymoo.core.population.Population class
    pop = algorithm.ask()
    decision_space = pop.get("X") # numpy array of points dimension n_variables

    if algorithm.n_gen ==1:
        # Every point has its evaluation as an attribute so can store all info in one list
        evaluated_archive.extend(pop)

        # Evaluate all individuals using the algorithm's evaluator (contains count of evaluations for termination)
        algorithm.evaluator.eval(problem, pop, skip_already_evaluated=False)
        objective_space = pop.get("F") # numpy array of points dimension n_objectives

        # Train initial RBF model
        for i in range(n_objectives):
            target_values = objective_space[:, i] # all rows, objective i column

            # Can pass in number of neighbors...also consider epsilon especially per objective
            obj_model = RBFInterpolator(decision_space, target_values, kernel='multiquadric', epsilon=0.5)
            rbf_models.append(obj_model)
        
        print(objective_space[0:3])

        # print(f"{n_objectives}\n{n_variables}\n{decision_space[0:3]}\n{objective_space[0:3]}")

    else:
        # We wish to aproximate all the points, maybe in two ways, then depending
        # on the error re-evaltuate with surrogate
        
        # List of all values in decision space of the population
        # For all values in decision space but not in evalauted archive
        # evaluate on rbf model
        model_evaluations = []
        for i in range(n_objectives):
            obj_model = rbf_models[i]
            # Could remove values in the archive but leave for now (gets all predicted values)
            evaluation = obj_model(decision_space) # returns ndarray of predicted objective values
            model_evaluations.append(evaluation)
        model_F = np.column_stack(tuple(model_evaluations))

        # static = StaticProblem(problem, F=F)
        # Evaluator().eval(static, pop)


        # Get say the top 3 points (maybe given a certain distance apart)
        # also choose depending on accuracy of predictions maybe - could look at neighbour influence
        sample_num_to_evaluate = 3

        nds = NonDominatedSorting()
        # fronts gives a lists of fronts with each entry being the point index in model_F
        fronts = nds.do(model_F)

        try:
            new_archive_points_ind = random.sample(list(fronts[0]), sample_num_to_evaluate)
        except ValueError:
            print("Not enough front points")

        # These points have been actually evaluated
        new_archive_points = Population([pop[i] for i in new_archive_points_ind])
        # When get "F" here then empty as points not been evaluated yet
        old_vals = new_archive_points.get("F")
        algorithm.evaluator.eval(problem, new_archive_points, skip_already_evaluated=False)
        new_vals = new_archive_points.get("F")

        for ind in new_archive_points: 
            if ind in pop: print(ind)

        print(f"{old_vals} {new_vals}")


        # Retrain model


        # take those points, evaluate with function and then retrain model

        # plot2 = Scatter()
        # for i, front in enumerate(fronts):
        #     plt.scatter(model_F[front, 0], model_F[front, 1])
        # plt.show()

        # ie go through model evaluations and find which dominates

        algorithm.evaluator.eval(problem, pop, skip_already_evaluated=False)

        # plot.add(pop.get("F")) 


        # I want the indivuals and I want an attribute of whether they have been
        # evlauted with the real fitness or not
        # Population is literally a list of indiduals

        # print(model_F[0:3])
        


        # Take all values and get estimates from RBF model - have RBF model for each objective
        # in a list


    # returned the evaluated individuals which have been evaluated or modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    # print(algorithm.n_gen, algorithm.evaluator.n_eval)

# obtain the result objective from the algorithm
res = algorithm.result()

# plot = Scatter()
# alpha makes transparent
plot.add(problem.pareto_front(), alpha=0.7, s=5)
# makes transparent
plot.add(res.F, facecolors='none', edgecolors='green') 
plot.show()




"""
Notes:

F - Get the objective function vector for an individual.
G - Get the inequality constraint vector for an individual.
H - Get the equality constraint vector for an individual.


# get the design space values of the algorithm
    X = pop.get("X")

    # implement your evluation. here ZDT1
    f1 = X[:, 0]
    v = 1 + 9.0 / (problem.n_var - 1) * np.sum(X[:, 1:], axis=1)
    f2 = v * (1 - np.power((f1 / v), 0.5))
    F = np.column_stack([f1, f2])

    static = StaticProblem(problem, F=F)
    Evaluator().eval(static, pop)

"""