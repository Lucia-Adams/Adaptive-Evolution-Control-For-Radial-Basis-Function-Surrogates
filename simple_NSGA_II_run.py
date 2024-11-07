from problem_wrapper import pymooProblem
from reproblem import *

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


problem = pymooProblem(RE21())

# algorithm = NSGA2(pop_size=100)

# Uses Latin Hyper Cube Sampling as in most papers
# SBX crossover (can modify probability of crossover etc) and polynomial mutation as in Yu M
algorithm = NSGA2(pop_size=100, 
                sampling=LHS(), 
                crossover=SBX(eta=15, prob=0.9), 
                mutation=PolynomialMutation(),
                survival=RankAndCrowding())

# n_gen is termination tuple, seed is seeding random value
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
# alpha makes transparent
plot.add(problem.pareto_front(), alpha=0.7, s=5)
# makes transparent
plot.add(res.F, facecolors='none', edgecolors='green') 
plot.show()


# https://pymoo.org/algorithms/usage.html#nb-algorithms-usage