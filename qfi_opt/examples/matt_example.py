import numpy as np

import qfi_opt
from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_eigendecomposition, compute_QFI_simpler_api

np.set_printoptions(precision=4, linewidth=200)


def sim_wrapper_diffrax(x, qfi_grad, obj, obj_params, get_jacobian):
    rho = obj(x, obj_params["N"], dissipation_rates=obj_params["dissipation"])

    # force Hermitianness:
    rho = (rho + rho.conj().T) / 2.0
    # Compute eigendecomposition
    vals, vecs = compute_eigendecomposition(rho)

    qfi, new_grad = compute_QFI_simpler_api(vals, vecs, rho, x, qfi_grad, get_jacobian, obj_params)
    print(x, qfi, new_grad, flush=True)

    try:
        if qfi_grad.size > 0:
            qfi_grad[:] = -1.0 * new_grad
    except:
        qfi_grad[:] = []

    return -1.0 * qfi  # , -qfi_grad  # negative because we are maximizing


if __name__ == "__main__":

    N = 4
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    obj_params = {}
    obj_params["N"] = N
    obj_params["dissipation"] = 1.0
    obj_params["G"] = G

    seed = 88
    np.random.seed(seed)

    num_params = 5

    lb = np.zeros(num_params)
    ub = np.ones(num_params)

    x0 = np.random.uniform(lb, ub, num_params)
    model = "simulate_TAT"

    obj = getattr(spin_models, model)
    get_jacobian = spin_models.get_jacobian_func(obj)

    grad = np.zeros(num_params)
    out = sim_wrapper_diffrax(x0, grad, obj, obj_params, get_jacobian)
    print(out)
