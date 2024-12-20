import numpy as np
from numpy.linalg import LinAlgError

from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance
import random
import math

# pymoo imports
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def RBF_NSGA2(problem, base_moea, generations, err_scalar=None, const_sample=3, rand_seed=1, nbh_scale=3, rbf_epsilon=0.05):
    """
    problem (pymooProblem): Pymoo Problem, as returned by pymooProblem()
    base_moea (GeneticAlgorithm) : Base Pymoo MOEA, we use as returned by the NSGA2 function
    generations (int): generation count
    err_scalar (float): 0-1 scalar for model lenience. The higher the value the higher the model
                        accuracy at each stage
    const_sample (int): Given the err_scalar is specified, is the number of points sampled when in 
                        model acuracry bounds
    rand_seed (int): random seed
    nbh_scale (int): Fraction of neighbourhood considered in RBF function
    rbf_epsilon (float): RBF epsilon parameter
    """

    n_objectives = problem.get_n_objectives()
    base_moea.setup(problem, termination=('n_gen', generations), seed=rand_seed, verbose=False)
    pop_size = base_moea.pop_size
    evaluated_archive = []
    rbf_models = []
    err_plot = [0]
    err_margin = 0

    if err_scalar is None:
        sample_num_to_evaluate = const_sample
    else:
        sample_num_to_evaluate = 3 # start with 3 extra evaluations to get good initial error bound
    neighbours = pop_size//nbh_scale

    # Repeat algorithm for set number of generations
    for n_gen in range(generations): 
        pop = base_moea.ask()  # Asks the algorithm for the next solution to be evaluated (Population class)
        decision_space = pop.get("X") # numpy array of points dimension n_variables

        # --- First generation trains model ---
        if n_gen ==0:
            evaluated_archive.extend(pop)
            # Evaluate all individuals using the algorithm's evaluator (contains count of evaluations for termination)
            base_moea.evaluator.eval(problem, pop, skip_already_evaluated=False)
            objective_space = pop.get("F") # numpy array of points of dimension n_objectives

            # Train initial RBF model
            for i in range(n_objectives):
                target_values = objective_space[:, i] # all rows, objective i column
                obj_model = RBFInterpolator(decision_space, target_values, kernel='multiquadric', 
                                            epsilon=rbf_epsilon, neighbors=neighbours)
                rbf_models.append(obj_model)

        # --- All other generations ---
        else:
            # Evaluate population on the rbf models
            model_evaluations = []
            for i in range(n_objectives):
                obj_model = rbf_models[i]
                # Could remove values in the archive but leave for now (gets all predicted values)
                try:
                    evaluation = obj_model(decision_space) # returns ndarray of predicted objective values
                except LinAlgError:
                    print(f"Evaluation LinAlgError")
                model_evaluations.append(evaluation)
            model_F = np.column_stack(model_evaluations)
            # This gives the population their model predicted values
            static = StaticProblem(problem.get_basic_problem(), F=model_F)
            Evaluator().eval(static, pop)

            nds = NonDominatedSorting()
            fronts = nds.do(pop.get("F")) # lists of front with each entry being the point index in model_F
            # Randomly choose points according to pareto fronts of predicted values
            # Future work: add diversity here instead of random like crowding distance
            counter, fr = sample_num_to_evaluate, 0 # count how many points we have left to select and which front we are on
            new_archive_points_ind = []
            while counter > 0:
                front_len = len(fronts[fr])
                if counter <= front_len: # number of points left to sample are all in this front
                    new_archive_points_ind.extend(random.sample(list(fronts[fr]), counter))
                    break
                else:
                    new_archive_points_ind.extend(list(fronts[fr]))
                    counter -= front_len
                    fr +=1

            new_archive_points = Population([pop[i] for i in new_archive_points_ind])
            surrogate_vals = new_archive_points.get("F")
            base_moea.evaluator.eval(problem, new_archive_points, skip_already_evaluated=False)
            evaluated_archive.extend(list(new_archive_points))
            function_vals = new_archive_points.get("F")

            # Could work out distance between the predicted ones and actual ones to determine 
            # next strategy management selection
            err_distances = [distance.euclidean(surrogate_vals[i],function_vals[i]) for i in range(sample_num_to_evaluate)]
            log_avg_err_dist = math.log((sum(err_distances)/len(err_distances))+1)
            err_plot.append(log_avg_err_dist)

            # For this, all values in set have been evaluated so has lowest incremental error 
            if n_gen == 1 and err_scalar: err_margin = log_avg_err_dist/err_scalar
            # then if an err_scalar has been provided, adjust next sample points on error scale
            elif err_scalar and log_avg_err_dist > err_margin:
                extra_points = int(round((log_avg_err_dist-err_margin)*(pop_size-const_sample)/log_avg_err_dist, 0))
                # print(f"Gen{n_gen}: {round(log_avg_err_dist,3)} is above {round(err_margin,3)}: {extra_points}")
                sample_num_to_evaluate =const_sample + extra_points
            elif err_scalar:
                sample_num_to_evaluate = const_sample
            # Otherwise will take 3 samples each iteration

            # --- Retrain model ---
            if n_gen != generations-1:
                archive_descision_space = Population(evaluated_archive).get("X")
                archive_objective_space = Population(evaluated_archive).get("F")
                for i in range(n_objectives):
                    target_values = archive_objective_space[:, i] # all rows, objective i column
                    # Can pass in number of neighbors...also consider epsilon especially per objective
                    obj_model = RBFInterpolator(archive_descision_space, target_values, kernel='multiquadric', 
                                                epsilon=rbf_epsilon, neighbors=neighbours)
                    rbf_models[i] = obj_model
            # -----

        base_moea.tell(infills=pop)

    # obtain the result objective from the model algorithm
    rbf_res = base_moea.result()
    # Get actual values from predicted decision space
    rbf_result_X = rbf_res.X
    out = problem.evaluate(rbf_result_X, return_values_of=["F"], return_as_dictionary=True)
    rbf_result_F = out["F"]

    evaluations = base_moea.evaluator.n_eval + len(rbf_result_X)

    return rbf_result_F, evaluations, err_plot


