import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem

class pymooProblem(ElementwiseProblem):
    def __init__(self, re_problem):
        self.re_problem = re_problem
        # maps re problem naming to pymoo naming
        super().__init__(n_var=re_problem.n_variables, n_obj=re_problem.n_objectives,
                         xl=np.array(self.re_problem.lbound),
                         xu=np.array(self.re_problem.ubound))

        with open(f'approximated_Pareto_fronts/reference_points_{self.re_problem.problem_name}.dat', 'r') as data:
            self.pf = np.array([[np.float64(x) for x in l.strip('\n').split(' ')] for l in data.readlines()])
            data.close()

    def pareto_front(self):
        '''override to use imported (approx) pareto front'''
        return self.pf

    def _evaluate(self, x, out):
        '''set output using associated re problem evaluate func'''
        out["F"] = self.re_problem.evaluate(x)

    def get_basic_problem(self):
        '''returns basic problem version without predefined evaluation for adaption into static problem'''
        return Problem(n_var=self.re_problem.n_variables, n_obj=self.re_problem.n_objectives,
                        n_constr=0, xl=np.array(self.re_problem.lbound), xu=np.array(self.re_problem.ubound))

    def get_n_objectives(self):
        return self.re_problem.n_objectives
    
    def get_n_variables(self):
        return self.re_problem.n_variables

