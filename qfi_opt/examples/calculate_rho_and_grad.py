#!/usr/local/bin/ python3
import sys

import numpy as np
from scipy.io import savemat

from qfi_opt import spin_models

if __name__ == "__main__":
    sense = sys.argv[1]  # this should be "f" or "g"
    num_spins = int(sys.argv[2])
    dissipation = float(sys.argv[3])
    num_params = int(sys.argv[4])
    if num_params == 4:
        model = "simulate_OAT"  # , "simulate_ising_chain", "simulate_XX_chain"]
    elif num_params == 5:
        model = "simulate_TAT"  # , "simulate_local_TAT_chain"]

    # from command line input, the point we wish to evaluate:
    center = np.array([float(v) for v in sys.argv[5 : num_params + 5]])

    op = spin_models.collective_op(spin_models.PAULI_Z, num_spins) / (2 * num_spins)
    obj = getattr(spin_models, model)

    obj_params = {}
    obj_params["N"] = num_spins
    obj_params["dissipation"] = dissipation
    obj_params["G"] = op

    # Evaluate at center first.
    params = center
    rho = obj(params, num_spins, dissipation_rates=dissipation)
    mdic = {"rho": rho, "G": op}
    outputfile = "rho_center.mat"
    savemat(outputfile, mdic)

    if sense == "g":
        get_jacobian = spin_models.get_jacobian_func(obj)
        dA = get_jacobian(params, obj_params["N"], dissipation_rates=obj_params["dissipation"])
        dA = np.transpose(dA, (2, 0, 1))
        mdic = {"drho": dA}
        outputfile = "rho_grad.mat"
        savemat(outputfile, mdic)
