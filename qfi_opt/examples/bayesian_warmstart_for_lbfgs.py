#!/usr/bin/env python3
import numpy as np
import PermSolver_methods as psm
import PermSolver_matrix as matrix
import random
import sys
from LBFGSB import LBFGSB
from bayes_opt import BayesianOptimization
from pounders_qfi import density_Ffun, qfi_hfun

def LBFGSB_wrapper(x, obj_params, get_jacobian=[]):

    x = np.squeeze(x.T)
    dim = len(x)
    qfi_grad = np.zeros(dim)

    rho = psm.simulate_layers(params=x,
                                     num_qubits=obj_params['N'],
                                     Hamiltonian_set=obj_params['Hmat_set'],
                                     dissipation_rates=obj_params['dissipation'] / np.pi)

    jacobian = get_jacobian(params=x,
                            num_qubits=obj_params['N'],
                            Hamiltonian_set=obj_params['Hmat_set'],
                            dissipation_rates=obj_params['dissipation'] / np.pi)

    qfi, new_grad = psm.compute_QFI(rho, x, jacobian, obj_params=obj_params, grad=qfi_grad)

    new_grad = np.expand_dims(new_grad, 0).T

    return -1.0 * qfi, -1.0 * new_grad

def bayes_wrapper(x, obj_params):

    Fvec = density_Ffun(x, obj_params)
    qfi = qfi_hfun(Fvec, obj_params)

    # this only returns zeroth-order info by design
    return qfi

def run_bayes_opt(x_opt, sim_params, bounds, random_seed=888):
    # This function is intended only to attempt global maximization over theta.
    # The high-level idea of why I'm providing it is to sanity check a solution to make sure that the max over theta in
    # the definition of CFI is not, in fact, only a local maximum (meaning the CFI definition is wrong).

    n = len(x_opt)

    # bayes_opt solver requires naming your variables like this:
    # (i personally believe they do this because it forces you to consider, explicitly, how large your problem is and
    # whether bayesian optimization is appropriate)

    pbounds = {}
    for t in range(n):
        pbounds['x[' + str(t) + ']'] = bounds[t]

    def bayes_wrapped(*args, **kwargs):
        n = len(kwargs)
        x = np.zeros(n)
        for t in range(n):
            x[t] = kwargs['x['+str(t)+']']
        return -1.0 * bayes_wrapper(x, sim_params)

    optimizer = BayesianOptimization(
        f=bayes_wrapped,
        pbounds=pbounds,
        random_state=random_seed,
        verbose=1
    )

    optimizer.maximize(
        init_points=n, # intuition: Latin hypercube sampling
        n_iter=10*n
    )

    qfi_value = -1.0 * optimizer.max['target']
    x_star = np.zeros(n)
    for t in range(n):
        x_star[t] = optimizer.max['params']['x[' + str(t) + ']']

    return qfi_value, x_star

N = int(sys.argv[1])
model = sys.argv[2]
exec(f"Hmat_set = matrix.{model}Mat(1.0, N//2)")
dissipation = float(sys.argv[3])
layers = int(sys.argv[4])

obj_params = {'G': psm.MatSz(N//2), 'N': N, 'dissipation': dissipation, 'layers': layers, 'Hmat_set': Hmat_set}

# set up initial vector, parameter bounds
x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * obj_params['layers'])] + [1])
      * np.random.rand(3 + 2 * obj_params['layers']))
bounds = [(0, 1/2) for _ in range(2)] + [(0,1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * obj_params['layers'])] + [(0, 1)]

num_params = 3 + 2 * layers

# finishing loading sim_params
Fvec, obj_params = density_Ffun(x0, obj_params)

# first run Bayesian optimization to get a good starting point
qfi_value, x0 = run_bayes_opt(x0, obj_params, bounds, random_seed=8)

print("QFI after Bayesian optimization: ", qfi_value)

# now set up LBFGS

get_jacobian = psm.get_jacobian_func(psm.simulate_layers)  # check if simulate_layers is the correct string here.

func = lambda x: LBFGSB_wrapper(x, obj_params, get_jacobian)

lower_bounds = np.expand_dims(-np.inf * np.ones(num_params), 0).T
upper_bounds = np.expand_dims(np.inf * np.ones(num_params), 0).T
x0 = np.expand_dims(x0, 0).T
x, xhist, exitflag = LBFGSB(func, x0, lower_bounds, upper_bounds, m=10, tol=1e-5, max_iters=50, display=True, xhistory=False)
print(exitflag)
