from problem_wrapper import pymooProblem
from reproblem import *
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from rbf_NSGA_II import RBF_NSGA2 
import itertools 

# Kep constant
POP_SIZE = 100
N_GEN = 40

# problem_list = [RE24(), RE31(), RE32(), RE33(), RE34(), RE37(), RE41(), RE42(), RE61(), RE91()]
problem_list= [RE41()]

err_scalar_list = [None, 0.1, 0.3, 0.5, 0.7, 0.9]
trial_num = 5
const_sample_list = [1,3]
neighbours_scale = [1,3,5]

combinations = list(itertools.product(err_scalar_list, const_sample_list, neighbours_scale))
combinations=combinations[0:3]

for p in problem_list:

    pymoo_problem = pymooProblem(p)
    
    for comb in combinations:
        err_scale, const_sample, nbh_scale = comb

        for trial in range(1,trial_num+1):

            # Two instances of the NSGA2 base algorithm, one for out surrogate experiment and one control
            nsga2_base = NSGA2(pop_size=POP_SIZE, sampling=LHS(), mutation=PolynomialMutation())
            control_algorithm = NSGA2(pop_size=POP_SIZE, sampling=LHS(), mutation=PolynomialMutation())

            # Our algorithm
            rbf_nsga2_F, evals, err_plot = RBF_NSGA2(pymoo_problem, nsga2_base, N_GEN, 
                                                    err_scalar=err_scale,
                                                    const_sample=const_sample,
                                                    rand_seed=trial, 
                                                    nbh_scale=nbh_scale)

            # Control Algorithm
            control_res = minimize(pymoo_problem, control_algorithm, ('n_gen', N_GEN), seed=trial, verbose=False)

            if pymoo_problem.get_n_objectives() <= 3:
                plot = Scatter()
                plot.add(pymoo_problem.pareto_front(), alpha=0.7, s=5) # alpha makes transparent
                plot.add(control_res.F, facecolors='none', edgecolors='orange') 
                plot.add(rbf_nsga2_F, color="green")   
                plot.show()

                plt.figure()
                plt.plot(range(0, len(err_plot)), err_plot)
                plt.show()


            # Output parameters, graph and hupervolume and igd


# """
# Notes:

# Uses Latin Hyper Cube Sampling as in most papers
# SBX crossover (can modify probability of crossover etc) and polynomial mutation as in Yu M


# """