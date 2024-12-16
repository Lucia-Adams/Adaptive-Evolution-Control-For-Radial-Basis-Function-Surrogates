from problem_wrapper import pymooProblem
from reproblem import *

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from numpy.linalg import LinAlgError
from rbf_NSGA_II import RBF_NSGA2 
import itertools 

POP_SIZE = 100
N_GEN = 40

problem_list = [RE24(), RE31(), RE32(), RE33(), RE34(), RE37(), RE41(), RE42(), RE61(), RE91()]
err_scalar_list = [None, 0.1, 0.3, 0.5, 0.7, 0.9]
trial_num = 10
const_sample_list = [1,3,50]
neighbours_scale = [1,3,5]

combinations = list(itertools.product(err_scalar_list, const_sample_list, neighbours_scale))
graph_counter=0

for p in problem_list:

    pymoo_problem = pymooProblem(p)
    approx_nadir = pymoo_problem.pareto_front().max(axis=0)
    gdplus_ind = GDPlus(pymoo_problem.pareto_front())
    igdplus_ind = IGDPlus(pymoo_problem.pareto_front())
    # hv_ind = HV(ref_point=approx_nadir)

    control_results =[]

    for trial in range(trial_num):
        control_algorithm = NSGA2(pop_size=POP_SIZE, sampling=LHS(), mutation=PolynomialMutation())
        control_res = minimize(pymoo_problem, control_algorithm, ('n_gen', N_GEN), seed=(trial+1), verbose=False)

        c_gd_plus = gdplus_ind(control_res.F)
        c_igd_plus = igdplus_ind(control_res.F)
        # c_hv = hv_ind(control_res.F)
        c_evals = control_res.algorithm.evaluator.n_eval

        # control_results.append([control_res, c_gd_plus, c_igd_plus, c_hv, c_evals])
        control_results.append([control_res, c_gd_plus, c_igd_plus, 0, c_evals])

    for comb in combinations:
        err_scale, const_sample, nbh_scale = comb

        for trial in range(trial_num):
            try:
                nsga2_base = NSGA2(pop_size=POP_SIZE, sampling=LHS(), mutation=PolynomialMutation())

                # My algorithm
                rbf_nsga2_F, evals, err_plot = RBF_NSGA2(pymoo_problem, nsga2_base, N_GEN, 
                                                        err_scalar=err_scale,
                                                        const_sample=const_sample,
                                                        rand_seed=(trial+1), 
                                                        nbh_scale=nbh_scale)

                if p.n_objectives <= 3:
                    plot = Scatter(figsize=(12, 9))
                    plot.add(pymoo_problem.pareto_front(), alpha=0.3, s=5) # alpha makes transparent
                    plot.add(control_results[trial][0].F, facecolors='none', edgecolors='orange', s=10) 
                    plot.add(rbf_nsga2_F, facecolors='none', edgecolors='green')   
                    plot.save(f'graphs/problem_{p.problem_name}_{graph_counter}.png')
                    graph_counter+=1

                gd_plus = gdplus_ind(rbf_nsga2_F)
                igd_plus = igdplus_ind(rbf_nsga2_F)
                # hv = hv_ind(rbf_nsga2_F)

                err_plot_string = "+".join(map(str,err_plot))

                with open(f"Expr/{p.problem_name}_experiments.csv", "a") as exp_file: 
                    # exp_file.write(f"{POP_SIZE},{N_GEN},{err_scale},{const_sample},{nbh_scale},"
                    #             f"{evals},{gd_plus},{igd_plus},{hv},{err_plot_string},"
                    #             f"{c_evals},{gd_plus/c_gd_plus},{igd_plus/c_igd_plus},{hv/c_hv},{graph_counter}\n")
                    exp_file.write(f"{POP_SIZE},{N_GEN},{err_scale},{const_sample},{nbh_scale},"
                                  f"{evals},{gd_plus},{igd_plus},0,{err_plot_string},"
                                  f"{c_evals},{gd_plus/c_gd_plus},{igd_plus/c_igd_plus},0,{graph_counter}\n")

            except LinAlgError:
                print(f":trial_num={trial_num},const_sample={const_sample}, nbh_scale={nbh_scale}, err_scalar={err_scale}")

"""
Notes:

Uses Latin Hyper Cube Sampling as in most papers
SBX crossover (can modify probability of crossover etc) and polynomial mutation 
"""