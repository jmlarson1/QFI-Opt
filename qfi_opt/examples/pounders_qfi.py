import ibcdfo.pounders as pdrs
import numpy as np
import PermSolver_methods as methods
import PermSolver_matrix as matrix
import ipdb

# still need to write an hfun, an hfun_d, and an (empty) combinemodels

def density_Ffun(params, sim_params):

    # compute rho
    params = np.squeeze(params.T)

    rho = methods.simulate_layers(params=params,
                                  num_qubits=sim_params['N'],
                                  Hamiltonian_set=sim_params['Hmat_set'],
                                  dissipation_rates=sim_params['dissipation'])

    return_new_params_flag = False
    if 'Fdim' in sim_params:
        Fdim = sim_params['Fdim']
        shape_vector = sim_params['shape_vector']
        num_blocks = len(shape_vector)
    else:
        # let's populate these missing entries
        Fdim = 0
        num_blocks = len(rho)
        shape_vector = np.zeros(num_blocks, dtype='int')
        for nb in range(num_blocks):
            subdim = np.shape(rho[nb])[0]
            shape_vector[nb] = subdim
            Fdim += subdim * (subdim + 1) # this is the size of the upper triangular part * 2
        sim_params['Fdim'] = Fdim
        sim_params['shape_vector'] = shape_vector
        return_new_params_flag = True

    Fvec = np.zeros(Fdim)
    fdim_idx = 0
    for nb in range(num_blocks):
        subdim = np.shape(rho[nb])[0]
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)

        upper_triangular_indices = np.triu_indices(rho[nb].shape[0])
        upper_triangular_array = rho[nb][upper_triangular_indices]

        # store the real part of the array first
        Fvec[fdim_idx:fdim_idx_end] = np.real(upper_triangular_array)

        fdim_idx = int(fdim_idx + subdim * (subdim + 1) / 2)
        fdim_idx_end = int(fdim_idx_end + subdim * (subdim + 1) / 2)

        # now store the imaginary part of the array
        Fvec[fdim_idx:fdim_idx_end] = np.imag(upper_triangular_array)

        fdim_idx = fdim_idx_end

    if return_new_params_flag:
        return Fvec, sim_params
    else:
        return Fvec

def qfi_hfun(z, sim_params):

    # left off here. need to compute QFI grad by following these steps:

    # reconstruct rho
    shape_vector = sim_params['shape_vector']
    Jmax = len(shape_vector)

    fdim_idx = 0
    rho = np.empty((Jmax,), dtype=object)
    for jm in range(Jmax):
        subdim = shape_vector[jm]
        rho_block = np.zeros((subdim, subdim), dtype='complex128')

        # populate the real entries of the matrix:
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)
        upper_triangular_array = z[fdim_idx:fdim_idx_end]
        upper_triangular_indices = np.triu_indices(subdim)
        rho_block[upper_triangular_indices] = upper_triangular_array

        # populate the complex entries of the matrix:
        fdim_idx = fdim_idx_end
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)
        upper_triangular_array = z[fdim_idx:fdim_idx_end]
        rho_block[upper_triangular_indices] += 1j * upper_triangular_array

        # make rho_block Hermitian
        rho_block += rho_block.T.conj()
        rho_block[np.diag_indices_from(rho_block)] /= 2.0

        # append
        rho[jm] = rho_block

        # get ready for next iteration
        fdim_idx = fdim_idx_end

    # now that we've unpacked rho, we can compute qfi of rho
    # we could combine these two for-loops over Jmax, but i appreciate the legibility here.
    running_sum = 0.0
    for jm in range(Jmax):
        eigvals, eigvecs = methods.compute_eigendecomposition(rho[jm])
        num_vals = len(eigvals)

        G = sim_params["G"][jm]

        # There should never be negative eigenvalues, so their magnitude gives an
        # empirical estimate of the numerical accuracy of the eigendecomposition.
        # We discard any QFI terms denominators within an order of magnitude of
        # this value.
        tol = 1e-8

        for i in range(num_vals):
            for j in range(num_vals):
                denom = eigvals[i] + eigvals[j]
                diff = eigvals[i] - eigvals[j]
                if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                    f_quotient = methods.qfi_quotient(eigvals[i], eigvals[j], np.array([]))
                    eigenvector_bases = np.zeros((1, num_vals, num_vals), dtype="cdouble")
                    eigenvector_bases[0] = eigvecs
                    f_modulus = methods.qfi_modulus(G, np.array([]), i, j, eigenvector_bases)
                    running_sum += f_quotient * f_modulus

    const = (2 / (sim_params['N']**2))
    const = -1.0 * const # because we're minimizing !!!

    return const * running_sum


def qfi_hfun_d(z, direction, sim_params):

    # returns central finite difference approximation of directional derivative
    diff_param = 1e-8
    forward = qfi_hfun(z + diff_param * direction, sim_params)
    backward = qfi_hfun(z - diff_param * direction, sim_params)

    return [], (forward - backward) / (2.0 * diff_param)



def run_pounders(initial_point, Ffun, hfun, hfun_d, sim_params, m, delta_0=0.125, spsolver=4, Prior=None, nf_max=500, g_tol=1e-4):

    n = len(initial_point)

    def wrapped_hfun(y):
        return hfun(y, sim_params)

    def wrapped_hfun_d(y, yd):
        return hfun_d(y, yd, sim_params)

    if spsolver == 4:
        combinemodels = []
    else:
        combinemodels = pdrs.identity_combine

    Opts = {
        "hfun": wrapped_hfun,  # using structure
        "combinemodels": combinemodels, # not actually used, make sure this doesn't cause errors downstream
        "hfun_d": wrapped_hfun_d,  # using structure
        "printf": 1,  # for debugging.
        "spsolver": spsolver,
        "delta_min": 1e-6
    }

    Pars = [np.sqrt(n), 10.0, 0.001, 0.001] # the second number is forcing us to pick points closer to TR.
    #Model = {"np_max": int((n + 1) * (n + 2) / 2), "Par": Pars}
    Model = {"np_max": 2*n + 1, "Par": Pars}

    def wrapped_Ffun(x):
        return Ffun(x, sim_params)

    # don't actually bound the pounders run (function is periodic in all variables)
    bounds = [(-np.inf, np.inf) for _ in range(n)]
    Low = np.array([entry[0] for entry in bounds])
    Upp = np.array([entry[1] for entry in bounds])

    [X, F, hF, flag, xkin] = pdrs.pounders(wrapped_Ffun, initial_point, n, nf_max, g_tol, delta_0, m, Low, Upp,
                                           Options=Opts, Model=Model, Prior=Prior)

    return X, F, hF, flag, xkin


if __name__ == "__main__":
    ##  Define the problem. This could be passed in a number of ways
    N = 5
    model = 'OAT'
    assert model in ['OAT', 'TAT'], 'model choice not in available options.'
    exec(f"Hmat_set = matrix.{model}Mat(1.0, N//2)")
    coupling_exponent = 0.0
    dissipation = 0.01
    layers = 2

    # create dictionary from simulation parameters
    sim_params = {'G': methods.MatSz(N // 2),
                  'N': N,
                  'dissipation': dissipation,
                  'layers': layers,
                  'Hmat_set': Hmat_set}

    n = 3 + 2 * layers
    initial_point = np.array(2 * [1 / 4] + [1 / 4 if _ % 2 == 0 else 1 / 2 for _ in range(2 * layers)] + [1 / 2])

    # two output arguments because Fdim and shape_vector will be added to sim_params after this evaluation.
    initial_Fvec, sim_params = density_Ffun(initial_point, sim_params)

    qfi_value = qfi_hfun(initial_Fvec, sim_params)

    # two more parameters for pounders:
    delta_0 = 0.125
    Prior = None

    nf_max = 50*n  # this provides a lower bound on function evaluations per optimizer call

    # call pounder
    print("Running pounder (no s!) to find a stationary point of the composite objective function.")
    hF = lambda x, sim_params: qfi_hfun(density_Ffun(x, sim_params), sim_params)
    trivial_hfun = lambda F, sim_params: np.squeeze(F)
    spsolver = 2
    # two output arguments because Fdim and shape_vector will be added to sim_params after this evaluation.
    initial_Fvec = hF(initial_point, sim_params)

    m = 1
    X, F, hF, flag, xkin = run_pounders(initial_point, hF, trivial_hfun, [], sim_params, m, delta_0, spsolver,
                                        Prior=Prior, nf_max=nf_max, g_tol=1e-4)
    x_opt = X[xkin]
    qfi_after_pounders = -1.0 * hF[xkin]
    print("Estimate of optimal QFI: ", qfi_after_pounders)

    # call pounders
    print("Running pounders to find a stationary point of the composite objective function.")
    spsolver = 4
    # two output arguments because Fdim and shape_vector will be added to sim_params after this evaluation.
    initial_Fvec = density_Ffun(initial_point, sim_params)
    m = len(initial_Fvec)

    qfi_value = qfi_hfun(initial_Fvec, sim_params)

    # three more parameters for pounders:
    m = len(initial_Fvec)
    X, F, hF, flag, xkin = run_pounders(initial_point, density_Ffun, qfi_hfun, qfi_hfun_d, sim_params, m, delta_0, spsolver,
                                        Prior=Prior, nf_max=nf_max, g_tol=1e-4)
    x_opt = X[xkin]
    qfi_after_pounders = -1.0 * hF[xkin]
    print("Estimate of optimal QFI: ", qfi_after_pounders)