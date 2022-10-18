import itertools 
import numpy as np
from run_OAT import simulate_OAT

def compute_QFI(rho, G):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eig(rho)

    # Take the real part of the eigenvalues 
    eigvals = np.real(eigvals)

    n0 = len(eigvals)

    # Extract nonzero eigenvalues (and corresponding eigenvectors)
    zero_inds = np.isclose(eigvals, np.zeros(n0), rtol=1e-08, atol=1e-08)
    nonzero_inds = np.logical_not(zero_inds)
    n1 = sum(nonzero_inds)
    nonzero_eigvals = eigvals[nonzero_inds]
    nonzero_eigvecs = eigvecs[nonzero_inds]

    # Compute QFI
    running_sum = 0
    for i in range(n1):
        for j in range(i+1, n1):
            denom = nonzero_eigvals[i] + nonzero_eigvals[j] 
            if abs(denom) > 1e-12:
                numer = (nonzero_eigvals[i] - nonzero_eigvals[j])**2
                term = nonzero_eigvecs[i] @ G @ nonzero_eigvecs[j] 
                running_sum += (numer/denom)*np.linalg.norm(term)**2

    return running_sum


N = 4
noise = 0 
G = np.eye(2**N)

# Let's try calculating the QFI at all corner points of the domain: 
all_perms = [",".join(seq) for seq in itertools.product("01", repeat=4)]
for perm in all_perms:
    params = np.fromstring(perm, dtype=int, sep=",")
    rho = simulate_OAT(N, params, noise)
    qfi = compute_QFI(rho, G)
    print(f"QFI is {qfi} for {params}")

# Let's try calculating the QFI at some random points in the domain: 
np.random.seed(0)
for _ in range(10):
    params = np.random.uniform(0, 1, 4)
    rho = simulate_OAT(4, params, 0)
    qfi = compute_QFI(rho, G)
    print(f"QFI is {qfi} for {params}")




