import numpy as np
from problem_wrapper import pymooProblem
from reproblem import *

# pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pymoo.problems import get_problem
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance

import matplotlib.pyplot as plt
import random

POP_SIZE = 100
SBX_ETA = 15
SBX_PROB = 0.9
N_GEN = 10

problem = pymooProblem(RE24())
static_problem = problem.get_basic_problem()
n_objectives = problem.get_n_objectives()
n_variables = problem.get_n_variables()

# Uses Latin Hyper Cube Sampling as in most papers
# SBX crossover (can modify probability of crossover etc) and polynomial mutation as in Yu M
rbf_algorithm = NSGA2(pop_size=POP_SIZE, 
                sampling=LHS(), 
                crossover=SBX(eta=SBX_ETA, prob=SBX_PROB), 
                mutation=PolynomialMutation(),
                survival=RankAndCrowding())

# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
rbf_algorithm.setup(problem, termination=('n_gen', N_GEN), seed=1, verbose=False)

# To store points/individuals that have used the actual fitness evaluation
evaluated_archive = []
rbf_models=[]
err_plot = [0]

plot = Scatter()

# Until the algorithm has no terminated - we have set to when gets to certain number of generations in algorithm setup tuple
for n_gen in range(N_GEN):

    # Asks the algorithm for the next solution to be evaluated
    # Returns the pymoo.core.population.Population class
    pop = rbf_algorithm.ask()
    decision_space = pop.get("X") # numpy array of points dimension n_variables

    if n_gen ==0:
        evaluated_archive.extend(pop)

        # Evaluate all individuals using the algorithm's evaluator (contains count of evaluations for termination)
        rbf_algorithm.evaluator.eval(problem, pop, skip_already_evaluated=False)
        objective_space = pop.get("F") # numpy array of points of dimension n_objectives

        # Train initial RBF model
        for i in range(n_objectives):
            target_values = objective_space[:, i] # all rows, objective i column
            # Can pass in number of neighbors...also consider epsilon especially per objective
            obj_model = RBFInterpolator(decision_space, target_values, kernel='multiquadric', epsilon=0.1)
            rbf_models.append(obj_model)

        # print(f"{n_objectives}\n{n_variables}\n{decision_space[0:3]}\n{objective_space[0:3]}")

    else:
        # Evaluate population on the rbf models
        model_evaluations = []
        for i in range(n_objectives):
            obj_model = rbf_models[i]
            # Could remove values in the archive but leave for now (gets all predicted values)
            evaluation = obj_model(decision_space) # returns ndarray of predicted objective values
            model_evaluations.append(evaluation)
        model_F = np.column_stack(model_evaluations)

        # This gives the population their model predicted values
        static = StaticProblem(problem.get_basic_problem(), F=model_F)
        Evaluator().eval(static, pop)

        nds = NonDominatedSorting()
        fronts = nds.do(pop.get("F")) # lists of front with each entry being the point index in model_F

        # plot2 = Scatter()
        # for i, front in enumerate(fronts):
        #     plt.scatter(pop.get("F")[front, 0], pop.get("F")[front, 1])
        # plt.show()

        # Randomly choose points according to pareto front of predicted values
        # also choose depending on accuracy of predictions maybe - could look at neighbour influence
        sample_num_to_evaluate = 3
        try:
            new_archive_points_ind = random.sample(list(fronts[0]), sample_num_to_evaluate)
        except ValueError:
            print("Not enough front points")

        new_archive_points = Population([pop[i] for i in new_archive_points_ind])
        surrogate_vals = new_archive_points.get("F")
        rbf_algorithm.evaluator.eval(problem, new_archive_points, skip_already_evaluated=False)
        evaluated_archive.extend(list(new_archive_points))
        function_vals = new_archive_points.get("F")

        # Could work out distance between the predicted ones and actual ones to determine 
        # next strategy management selection
        distances = [distance.euclidean(surrogate_vals[i],function_vals[i]) for i in range(sample_num_to_evaluate)]
        err_plot.append(sum(distances))

        # print(f"Model predicted: {surrogate_vals}\n Actual value: {function_vals}")

        # --- Retrain model ---
        archive_descision_space = Population(evaluated_archive).get("X")
        archive_objective_space = Population(evaluated_archive).get("F")
        for i in range(n_objectives):
            target_values = archive_objective_space[:, i] # all rows, objective i column
            # Can pass in number of neighbors...also consider epsilon especially per objective
            obj_model = RBFInterpolator(archive_descision_space, target_values, kernel='multiquadric', epsilon=0.05, neighbors=20)
            rbf_models[i] = obj_model
        # -----

    # TODO: Now re-evaluate objective space to get actual values it found and plot
    # Ie will have line for predicted, line for actual obtained from model algorithm, one for without model 
    # algorithm and actual pareto front

    print(rbf_algorithm.n_gen, rbf_algorithm.evaluator.n_eval)
    # returned the evaluated individuals which have been evaluated or modified
    rbf_algorithm.tell(infills=pop)

# obtain the result objective from the model algorithm
rbf_res = rbf_algorithm.result()

# Get actual values from precited decision space
rbf_result_X = rbf_res.X
out = problem.evaluate(rbf_result_X, return_values_of=["F"], return_as_dictionary=True)
rbf_result_F = out["F"]

# Run again but all with actual evaluations
control_algorithm = NSGA2(pop_size=POP_SIZE, 
                sampling=LHS(), 
                crossover=SBX(eta=SBX_ETA, prob=SBX_PROB), 
                mutation=PolynomialMutation(),
                survival=RankAndCrowding())


# n_gen is termination tuple, seed is seeding random value
control_res = minimize(problem,
               control_algorithm,
               ('n_gen', N_GEN),
               seed=1,
               verbose=False)

# plot = Scatter()
# alpha makes transparent
plot.add(problem.pareto_front(), alpha=0.7, s=5)
# makes transparent
plot.add(rbf_result_F, facecolors='none', edgecolors='green') 
plot.add(control_res.F, facecolors='none', edgecolors='orange') 
plot.show()

plt.figure()
plt.plot(range(0, len(err_plot)), err_plot)
plt.show()


# TODO: for error distance, maybe sample 3 points on the hypercube objective space, get distances to their nearest
# neighbours, and then at least general average between different objective space points. At least gets the scale right

# Hyper-parameter is epsilon and maybe soemthing to do with the rate ill set
# work on spread bc its really bad

# should be best near beginining so get distances between the accuracy then. set this as a benhcmark
# and if worse then maybe sample next round of points biased to the worse one. if ok then don't need to 
# do that extra re-evaluation