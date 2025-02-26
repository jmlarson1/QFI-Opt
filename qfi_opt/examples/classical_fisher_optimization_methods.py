import ibcdfo.pounders as pdrs
import numpy as np
from declare_hfun_and_combine_model_with_jax_CFI import hfun, combinemodels_jax, hfun_d
from bayes_opt import BayesianOptimization

def run_pounders(initial_point, Ffun, hfun, hfun_d, sim_params, m, delta_0=0.125, nf_max=500, g_tol=1e-4):

    n = len(initial_point)

    # need to scale hfun and hfun_d:
    N = sim_params['N']
    dphi = sim_params['dphi']

    def scaled_hfun(y):
        return hfun(y) / ((N * dphi) ** 2)

    def scaled_hfun_d(y, yd):
        resd = hfun_d(y, yd)
        resd = np.array(resd)
        for ctr in range(len(resd)):
            resd[ctr] = resd[ctr] / ((N * dphi) ** 2)
        return resd

    Opts = {
        "hfun": scaled_hfun,  # using structure
        "combinemodels": combinemodels_jax, # not actually used, just pulling from outer scope
        "hfun_d": scaled_hfun_d,  # using structure
        "printf": 1,  # for debugging.
        "spsolver": 4,
        "delta_min": 1e-8
    }

    Pars = [np.sqrt(n), 10.0, 0.001, 0.001] # the second number is forcing us to pick points closer to TR.
    Model = {"np_max": int((n + 1) * (n + 2) / 2), "Par": Pars}

    def wrapped_Ffun(x):
        return Ffun(x, sim_params)

    # don't actually bound the pounders run (function is periodic in all variables)
    bounds = [(-np.inf, np.inf) for _ in range(2 * layers + 3 + num_thetas)]
    Low = np.array([entry[0] for entry in bounds])
    Upp = np.array([entry[1] for entry in bounds])

    [X, F, hF, flag, xkin] = pdrs.pounders(wrapped_Ffun, initial_point, n, nf_max, g_tol, delta_0, m, Low, Upp,
                                           Options=Opts, Model=Model)

    return X, F, hF, flag, xkin


def run_bayes_opt(x_opt, num_thetas, bounds, rho, N, dphi, random_seed=888):
    # This function is intended only to attempt global maximization over theta.
    # The high-level idea of why I'm providing it is to sanity check a solution to make sure that the max over theta in
    # the definition of CFI is not, in fact, only a local maximum (meaning the CFI definition is wrong).

    n = len(x_opt)
    fixed_x = x_opt[:n - num_thetas]

    # bayes_opt solver requires naming your variables like this:
    pbounds = {}
    for t in range(num_thetas):
        pbounds['theta[' + str(t) + ']'] = bounds[n - num_thetas + t]

    def bayes_wrapped_hFfun_fixed_x(*args, **kwargs):
        num_thetas = len(kwargs)
        theta = np.zeros(num_thetas)
        for t in range(num_thetas):
            theta[t] = kwargs['theta['+str(t)+']']
        Fvec = Ffun(np.concatenate((fixed_x, theta)), sim_params, just_theta=True, rho=rho)
        return -1.0 * hfun(Fvec) / ((N * dphi) ** 2)

    optimizer = BayesianOptimization(
        f=bayes_wrapped_hFfun_fixed_x,
        pbounds=pbounds,
        random_state=random_seed,
        verbose=1
    )

    optimizer.maximize(
        init_points=num_thetas, # intuition: Latin hypercube sampling
        n_iter=np.maximum(2 ** num_thetas, 64)  # intuition: let an acquisition function at least explore the corners.
    )

    cfi_value = -1.0 * optimizer.max['target']
    theta_star = np.zeros(num_thetas)
    for t in range(num_thetas):
        theta_star[t] = optimizer.max['params']['theta[' + str(t) + ']']

    return cfi_value, theta_star

if __name__ == "__main__":
    ##  Define the problem. This could be passed in a number of ways
    N = 4
    model = 'XX'
    coupling_exponent = 0.0
    dissipation_rates = 0.1
    layers = 1
    cfi_type = 4

    if cfi_type == 1:
        from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
        num_thetas = 1
        bounds = [(0, 1 / 2), (0, 1 / 2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi / 2)]
    elif cfi_type == 2:
        from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_single_qubit_rotations_Ffun as Ffun
        num_thetas = N
        bounds = [(0, 1 / 2), (0, 1 / 2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi)]
    elif cfi_type == 3:
        from qfi_opt.examples.classical_fisher import compute_bitstring_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
        num_thetas = 1
        bounds = [(0, 1 / 2), (0, 1 / 2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi / 2)]
    elif cfi_type == 4:
        from qfi_opt.examples.classical_fisher import compute_bitstring_basis_CFI_for_single_qubit_rotations_Ffun as Ffun
        num_thetas = N
        bounds = [(0, 1 / 2), (0, 1 / 2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi)]

    # involved in CFI computation, unsure how much this should be exposed.
    dphi = 1e-5

    # create dictionary from simulation parameters
    sim_params = {'N': N, 'model': model, 'coupling_exponent': coupling_exponent, 'dissipation_rates': dissipation_rates, 'dphi': dphi}

    initial_point = np.array(2 * [1/4] + [1/4 if _ % 2 == 0 else 1/2 for _ in range(2 * layers)] + [1/2] + num_thetas * [0])
    rho = Ffun(initial_point, sim_params, just_return_rho=True)
    m = len(Ffun(initial_point, sim_params, just_theta=True, rho=rho))
    delta_0 = 0.125

    print("First testing pounders to find a stationary point of the composite objective function.")
    X, F, hF, flag, xkin = run_pounders(initial_point, Ffun, hfun, hfun_d, sim_params, m, delta_0, nf_max=500, g_tol=1e-4)

    print("Optimal CFI is ", -1.0 * hF[xkin])
    x_opt = X[xkin]

    print("Now probing the value of theta by Bayesian optimization to ensure that it's actually a global maximum.")
    rho = Ffun(x_opt, sim_params, just_return_rho=True)
    cfi, theta_star = run_bayes_opt(x_opt, num_thetas, bounds, rho, N, dphi)
    print("Optimal CFI after attempting to improve theta is: ", -1.0 * cfi)

    print("If the post-Bayes values of optimal CFI is significantly better than the pre-Bayes value, then pounders")
    print("unfortunately found a local maximum. You would ideally overwrite the last num_thetas entries of x_opt")
    print("with theta_star and restart pounders. Also, note that it's wholly possible that, for the fixed x variables,")
    print("theta is globally optimal. This simple test doesn't rule out this possibility. This is why restarts (or an ")
    print("attempt at global optimization in x, as well) are necessary to convince yourself that you have THE solution")




