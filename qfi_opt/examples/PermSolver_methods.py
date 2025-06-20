import os
import ipdb

USE_DIFFRAX = bool(os.getenv("USE_DIFFRAX"))

if USE_DIFFRAX:
    import diffrax
    import jax
    import jax.numpy as np
    from jax.scipy.linalg import expm
    from jax.numpy import linalg as LA
    jax.config.update("jax_enable_x64", True)
    COMPLEX_TYPE = np.complex128

else:
    import numpy as np  # type: ignore[no-redef]
    from scipy.linalg import expm
    from numpy import linalg as LA

###################################
## Dicke basis, ordered as |0,0>, |1,-1>,|1,0>... |J-1,J-1>,|J,-J>,...|J,J>
## Build the basis
## Initial state set as spin coherent state |0,0>
## Gate operation, Sx, Sy and Sz rotations
## Observable: Sx, Sy, Sz, Sx^2, Sy^2, Sz^2
## QFI covariance matrix


##Initial state
# initializes state in | j, +j >_z
def Init_rho(Jmax): #Jmax=N/2
    result=[]
    for i in range (0,Jmax+1):
        result.append(np.zeros(((2*i+1),(2*i+1)), dtype=np.complex128))
    #set the |J,J>=1, all other zero
    if USE_DIFFRAX == False:
        result[Jmax][2*Jmax][2*Jmax]=1
    else:
        result[Jmax] = result[Jmax].at[2*Jmax,2*Jmax].set(1)
    return result


##Rotations and unitary transformations
def MatSz(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j):
                    if USE_DIFFRAX == False:
                        result[mj][i][j]=i-mj
                    else:
                        result[mj] = result[mj].at[i,j].set(i-mj)
    return result


def MatSy(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j+1):
                    if USE_DIFFRAX == False:
                        result[mj][i][j]=-0.5*1j*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj))
                    else:
                        result[mj] = result[mj].at[i,j].set(-0.5*1j*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj)))
                elif(i==j-1):
                    if USE_DIFFRAX == False:
                        result[mj][i][j]=0.5*1j*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj))
                    else:
                        result[mj] = result[mj].at[i,j].set(0.5*1j*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj)))
    return result


def MatSx(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j+1):
                    if USE_DIFFRAX == False:
                        result[mj][i][j]=0.5*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj))
                    else:
                        result[mj] = result[mj].at[i,j].set(0.5*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj)))
                elif(i==j-1):
                    if USE_DIFFRAX == False:
                        result[mj][i][j]=0.5*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj))
                    else:
                        result[mj] = result[mj].at[i,j].set(0.5*np.sqrt(mj*(mj+1)-(j-mj)*(i-mj)))
    return result


def operator_moments(Jmax:float, return_first_moments_only:bool=False)->list:
    if return_first_moments_only:
        return [MatSx(Jmax), MatSy(Jmax), MatSz(Jmax)]
    else:
        return [MatSx(Jmax), MatSy(Jmax), MatSz(Jmax), MatSx2(Jmax), MatSy2(Jmax), MatSz2(Jmax)]


def unitary(rho,rotmat,theta):
    U = expm(-1j*rotmat*theta)
    result=U @ rho @ U.T.conj()
    return result


def UnitaryGate(rho, rotmat, theta, Jmax):
    result=[]
    for mj in range(0, Jmax+1):
        result.append(unitary(rho[mj],rotmat[mj],theta))
    return result


def flatrhomat(rho0, N):
    result=[]
    for i in range(0,N+1):
        for j in range(0,2*i+1):
            for k in range(j,2*i+1):
                result.append(rho0[i][j][k])
    return np.array(result)


def recoverrhomat(rho0, N):
    result=[]
    count=0
    for i in range(0,N+1):
        result.append(np.zeros((2*i+1,2*i+1),dtype=np.complex128))
        for mj1 in range(0,2*i+1):
            for mj2 in range(mj1,2*i+1):
                result[i][mj1][mj2]=rho0[count]
                if (mj1!=mj2):
                    result[i][mj2][mj1]=np.conj(rho0[count])
                count+=1
    return result


def recoverrhomat2(rho0, N, Nstep):
    result=[]

    # print(f'time = [0, {Nstep + 1}], count = [0, {N + 1}], rho0/shape = {len(rho0)}, {len(rho0[0])}')
    # print(rho0)
    for time in range (0, Nstep+1):
        result.append([])
        count=0
        for i in range(0,N+1):
            result[time].append(np.zeros((2*i+1,2*i+1),dtype=np.complex128))
            for mj1 in range(0,2*i+1):
                for mj2 in range(mj1,2*i+1):
                    if USE_DIFFRAX == False:
                        result[time][i][mj1][mj2]=rho0[count][time]
                        if (mj1!=mj2):
                            result[time][i][mj2][mj1]=np.conj(rho0[count][time])
                    else:
                        result[time][i] = result[time][i].at[mj1,mj2].set(rho0[count][time])
                        if mj1!=mj2:
                            result[time][i] = result[time][i].at[mj2, mj1].set(np.conj(rho0[count][time]))
                    
                    count+=1
    return result


def SzCalculation(rho0,N):
    resultSz=0
    resultSz2=0
    for i in range(0,N+1):
        for mj1 in range(0,2*i+1):
            resultSz+=(mj1-i)*rho0[i][mj1][mj1]
            resultSz2+=(mj1-i)*(mj1-i)*rho0[i][mj1][mj1]
    return resultSz,resultSz2
    

def Expectation(rho0, optr, N):
    result=0
    for i in range(0,N+1):
        mat=rho0[i]@optr[i]
        result+=np.trace(mat)
    return result


def MatSx2(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j+2):
                    result[mj][i][j]=0.25*np.sqrt(mj*(mj+1)-(i-mj)*(i-1-mj))*np.sqrt(mj*(mj+1)-(i-1-mj)*(i-2-mj))
                elif(i==j-2):
                    result[mj][i][j]=0.25*np.sqrt(mj*(mj+1)-(j-mj)*(j-1-mj))*np.sqrt(mj*(mj+1)-(j-1-mj)*(j-2-mj))
                elif(i==j):
                    result[mj][i][j]=0.25*(2*mj*(mj+1)-2*(j-mj)*(j-mj))
    return result


def MatSy2(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j+2):
                    result[mj][i][j]=-0.25*np.sqrt(mj*(mj+1)-(i-mj)*(i-1-mj))*np.sqrt(mj*(mj+1)-(i-1-mj)*(i-2-mj))
                elif(i==j-2):
                    result[mj][i][j]=-0.25*np.sqrt(mj*(mj+1)-(j-mj)*(j-1-mj))*np.sqrt(mj*(mj+1)-(j-1-mj)*(j-2-mj))
                elif(i==j):
                    result[mj][i][j]=0.25*(2*mj*(mj+1)-2*(j-mj)*(j-mj))
    return result


def MatSz2(Jmax):
    result=[]
    for mj in range (0,Jmax+1):
        result.append(np.zeros(((2*mj+1),(2*mj+1)), dtype=np.complex128))
        for i in range (0,2*mj+1):
            for j in range (0,2*mj+1):
                if(i==j):
                    result[mj][i][j]=(j-mj)*(j-mj)
    return result


def QFIcalculation(rho, Jmax, optrSx, optrSy, optrSz, tol=1e-10):
    Resultmat=np.zeros((3,3))
    for mj in range(0,Jmax+1):
        eigenvales, eigenvectors=LA.eigh(rho[mj], UPLO='L')
        Sx=eigenvectors.conj().T@optrSx[mj]@eigenvectors
        Sy=eigenvectors.conj().T@optrSy[mj]@eigenvectors
        Sz=eigenvectors.conj().T@optrSz[mj]@eigenvectors
        for i in range(0,2*mj+1):
            for j in range(0,2*mj+1):
                if(eigenvales[i]<=tol and eigenvales[j]<=tol):
                    continue

                Resultmat[0][0]+=2*(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sx[i][j]*Sx[j][i])
                Resultmat[0][1]+=(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sx[i][j]*Sy[j][i] + Sy[i][j]*Sx[j][i])
                Resultmat[0][2]+=(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sx[i][j]*Sz[j][i] + Sz[i][j]*Sx[j][i])
                Resultmat[1][1]+=2*(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sy[i][j]*Sy[j][i])
                Resultmat[1][2]+=(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sy[i][j]*Sz[j][i] + Sz[i][j]*Sy[j][i])
                Resultmat[2][2]+=2*(eigenvales[i]-eigenvales[j])**2/(eigenvales[i]+eigenvales[j])*np.real(Sz[i][j]*Sz[j][i])
        del eigenvales,eigenvectors,Sx,Sy,Sz
    Resultmat[1][0]=Resultmat[0][1]
    Resultmat[2][0]=Resultmat[0][2]
    Resultmat[2][1]=Resultmat[1][2]
        
    return Resultmat


# single generator QFI
def calc_QFI(rho, Jmax, optrSz, tol=1e-10)->float:
    result = 0
    for mj in range(0, Jmax + 1):
        vals, vecs = LA.eigh(rho[mj], UPLO='L')
        Sz = vecs.conj().T @ optrSz[mj] @ vecs
        for i in range(0, 2 * mj + 1):
            for j in range(0, 2 * mj + 1):
                denominator = (vals[i] + vals[j])
                if (denominator <= tol):
                    continue

                result += (vals[i] - vals[j])**2 * np.abs(Sz[i][j])**2 / denominator

        del vals, vecs, Sz

    return result * 2


import PermSolver_matrix as matrix

def simulate_layers(params:np.ndarray, num_qubits:int, Hamiltonian_set:list, dissipation_rates:tuple|float=0.0, dissipation_format:str='XYZ'):
    assert dissipation_format in ['XYZ', 'PMZ'], "dissipation format distinct from preset formats"
    if len(params) < 5 or not len(params) % 2:
        raise ValueError(f"The number of parameters should be an odd number >=5, not {len(params)}.")

    Jmax = num_qubits//2
    Nsteps = 0
    Sx, Sy, Sz = operator_moments(Jmax, return_first_moments_only=True)

    # set all spins down
    rho_init = UnitaryGate(Init_rho(Jmax), Sx, np.pi, Jmax)

    dissipation_rates = dissipation_rates / np.pi
    if dissipation_format == "XYZ":
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            dissipation_matrix_set = matrix.XYZ_DisMat(dissipation_rates, dissipation_rates, dissipation_rates, Jmax)
        else:
            dissipation_matrix_set = matrix.XYZ_DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                               Jmax)
    else:
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            dissipation_matrix_set = matrix.PMZ_DisMat(dissipation_rates, dissipation_rates, dissipation_rates,
                                                     0, 0, 0, Jmax)
        else:
            dissipation_matrix_set = matrix.PMZ_DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                     0, 0, 0, Jmax)

    # set up initial rotation axis -> params[1]
    axis_factor = 1
    cos_component = np.cos(axis_factor * np.pi * params[1])
    sin_component = np.sin(axis_factor * np.pi * params[1])
    Sphi1 = []
    for jj in range(Jmax + 1):
        Sphi1.append(Sx[jj] * cos_component + Sy[jj] * sin_component)

    # rotate state for angle params[0] * pi about axis set by params[1] * pi
    state = UnitaryGate(rho_init, Sphi1, params[0] * np.pi, Jmax)

    # Entangle-rotate
    for pp in range(2, len(params) - 1, 2):

        # if entangling time is not zero, entangle
        if params[pp] > 0:
            state_f = flatrhomat(state, Jmax)
            sol = matrix.Perm_solver(state_f, params[pp] * np.pi, *dissipation_matrix_set[:-1], *Hamiltonian_set, Nsteps)
            state = recoverrhomat2(sol, Jmax, Nsteps)[-1]
            # out_state = ent1[-1]
        state = UnitaryGate(state, Sx, params[pp + 1] * np.pi, Jmax)

    # final rotation about Y
    state = UnitaryGate(state, Sy, params[-1] * np.pi, Jmax)

    return state

def get_jacobian_func(simulate_func):
    """Convert a simulation method into a function that returns its Jacobian."""

    if USE_DIFFRAX:
        #print("USE_DIFFRAX and FORWARD_MODE")
        # forward-mode automatic differentiation

        def get_jacobian(params, *args: object, **kwargs: object) -> np.ndarray:
            _simulate_func = lambda params: simulate_func(params, *args, **kwargs)
            _get_jacobian = jax.jacfwd(_simulate_func, argnums=0, holomorphic=True)
            return _get_jacobian(np.array(params, dtype=COMPLEX_TYPE))

        return get_jacobian

    #def get_jacobian_manually(params: Sequence[float], *args: object, **kwargs: object) -> np.ndarray:
    def get_jacobian_manually(params, *args: object, **kwargs: object) -> np.ndarray:
        step_sizes = kwargs.get("step_sizes", 1e-10)
        if isinstance(step_sizes, float):
            param_step_sizes = [step_sizes] * len(params)
        assert len(param_step_sizes) == len(params)

        result_at_params = simulate_func(params, *args, **kwargs)
        shifted_results = [[] for i in range(len(result_at_params))]
        #shifted_results1 = [ ]
        #shifted_results2 = [ ]
        #shifted_results = [ shifted_results1, shifted_results2]
        #print(result_at_params[0].shape, result_at_params[1].shape )
        for idx, step_size in enumerate(param_step_sizes):
            new_params = list(params)
            new_params[idx] += step_size
            result_at_params_with_step = simulate_func(new_params, *args, **kwargs)
            #print("len(result_at_params_with_step)", len(result_at_params_with_step))
            for i in range(len(result_at_params_with_step)):
                shifted_results[i].append((result_at_params_with_step[i] - result_at_params[i])/ step_size)
            #shifted_results.append(res)
        #print(shifted_results[1])
        #return shifted_results
        #return np.stack(shifted_results, axis=-1)
        #print("result_at_params[0].shape, params.shape", (params.shape + result_at_params[0].shape))
        #return [np.array(shifted_results[0]).reshape((params.shape + result_at_params[0].shape )),
        #        np.array(shifted_results[1]).reshape((params.shape + result_at_params[1].shape)) ]
        return [np.array(shifted_results[res]).reshape((params.shape + result_at_params[res].shape)) for res in range(len(result_at_params))]

    return get_jacobian_manually

def print_jacobian(jacobian: np.ndarray, precision: int = 3, linewidth: int = 200) -> None:
    np.set_printoptions(precision=precision, suppress=True, linewidth=linewidth)
    params = jacobian.shape[2]
    for pp in range(params):
        print(f"d(final_state/d(params[{pp}]):")
        print(jacobian[:, :, pp])

def print_jacobian_manual(jacobian: np.ndarray, precision: int = 3, linewidth: int = 200) -> None:
    np.set_printoptions(precision=precision, suppress=True, linewidth=linewidth)
    params = jacobian.shape[0]
    for pp in range(params):
        print(f"d(final_state/d(params[{pp}]):")
        print(jacobian[pp, :, :])

def compute_eigendecomposition(rho: np.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    # consistent sorting:
    eigvals = np.real(eigvals)
    sort_inds = np.argsort(eigvals)
    eigvals = eigvals[sort_inds]
    eigvecs = eigvecs[sort_inds]
    return eigvals, eigvecs


def compute_QFI(rho: np.ndarray, params: np.ndarray, jacobian: np.ndarray, obj_params, tol: float = 1e-8, etol_scale: float = 10, grad=np.empty(0)):

    Jmax = len(rho)
    num_params = len(params)

    # Initialize QFI and grad
    running_sum = 0
    if grad.size > 0:
        grad[:] = np.zeros(num_params)

    for jm in range(Jmax):
        eigvals, eigvecs = compute_eigendecomposition(rho[jm])
        # Note: The eigenvectors must be rows of eigvecs
        num_vals = len(eigvals)

        G = obj_params["G"][jm]

        # There should never be negative eigenvalues, so their magnitude gives an
        # empirical estimate of the numerical accuracy of the eigendecomposition.
        # We discard any QFI terms denominators within an order of magnitude of
        # this value.
        tol = max(tol, -etol_scale * np.min(eigvals))

        if grad.size > 0:
            dA = jacobian[jm]
            if USE_DIFFRAX:
                if np.shape(dA)[0] == 1:
                    # for any blocks that may be 1D, they will contribute nothing to QFI.
                    continue
                dA = np.transpose(dA, (2, 0, 1))
            else:
                # for any blocks that may be 1D, they will contribute nothing to QFI.
                if np.shape(dA)[1] == 1:
                    continue
            psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")
            lambda_grads = np.zeros((num_params, num_vals))
            eigenvector_bases = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

            for k in range(num_params):
                # compute gradients of each eigenvalue
                psi_grad_k, lambda_grad_k, basis_k = get_matrix_grads_rotate(rho[jm], dA[k], eigvals, eigvecs, tol)
                if not USE_DIFFRAX:
                    psi_grads[k] = psi_grad_k
                else:
                    psi_grads = psi_grads.at[k].set(psi_grad_k)
                if not USE_DIFFRAX:
                    lambda_grads[k] = lambda_grad_k
                else:
                    lambda_grads = lambda_grads.at[k].set(lambda_grad_k)
                if not USE_DIFFRAX:
                    eigenvector_bases[k] = basis_k
                else:
                    eigenvector_bases = eigenvector_bases.at[k].set(basis_k)


        # NOW COMPUTE
        for i in range(num_vals):
            for j in range(i + 1, num_vals):
                denom = eigvals[i] + eigvals[j]
                diff = eigvals[i] - eigvals[j]
                if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                    if grad.size > 0:
                        f_quotient, g_quotient = qfi_quotient(eigvals[i], eigvals[j], lambda_grads[:, [i, j]])
                        f_modulus, g_modulus = qfi_modulus(G, psi_grads, i, j, eigenvector_bases)
                    else:
                        eigenvector_bases = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")
                        eigenvector_bases[0] = eigvecs
                        f_quotient = qfi_quotient(eigvals[i], eigvals[j], np.array([]))
                        f_modulus = qfi_modulus(G, np.array([]), i, j, eigenvector_bases)
                    running_sum += f_quotient * f_modulus
                    if grad.size > 0:
                        grad[:] += f_quotient * g_modulus + f_modulus * g_quotient

    const = (4 / (obj_params['N']**2))
    if grad.size > 0:
        return const * running_sum, const * grad
    else:
        return const * running_sum, []


def check_close_entries(arr, tol):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diff = arr[j] - arr[i]
            if np.isclose(diff, 0, atol=tol, rtol=tol):
                return True
    return False


def get_matrix_grads_rotate(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    if np.linalg.matrix_rank(rho) < dim:
        return psi_grads, lambda_grads, eigvecs # skip this term of the matrix

    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2], eigvals[ind1], atol=tol, rtol=tol):
                    break  # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            group_set = np.arange(ind1, ind2)

            if group_set.size == 0:
                group_set = [ind2]

            # Two cases - either the eigenvalue has multiplicity one or it doesn't.
            if len(group_set) > 1:
                Lambda_prime = eigvecs[group_set].conj() @ dA @ eigvecs[group_set].T
                H = Lambda_prime @ Lambda_prime
                eigvalsH, eigvecsH = np.linalg.eigh(H)
                rotated_eigvecs = eigvecs[group_set].T @ eigvecsH
                lhs = rho - eigvals[group_set[-1]] * np.eye(dim)
                lhs = np.hstack((lhs, -rotated_eigvecs))
                lhs_row2 = np.hstack((-rotated_eigvecs.T.conj(), np.zeros((len(group_set), len(group_set)))))
                lhs = np.vstack((lhs, lhs_row2))
                rhs = -dA @ rotated_eigvecs
                rhs = np.vstack((rhs, np.zeros((len(group_set), len(group_set)))))
                sol = np.linalg.solve(lhs, rhs)
                if not USE_DIFFRAX:
                    psi_grads[group_set] = sol[:dim, :].T
                else:
                    psi_grads = psi_grads.at[group_set].set(np.squeeze(sol[:dim, :].T))
                Lambda_prime = sol[dim:, :]
                if not USE_DIFFRAX:
                    lambda_grads[group_set] = np.real(np.diag(Lambda_prime))
                else:
                    lambda_grads = lambda_grads.at[group_set].set(np.squeeze(np.real(np.diag(Lambda_prime))))

                # key: let the routine that called this subroutine know we rotated the eigvecs
                if not USE_DIFFRAX:
                    eigvecs[group_set] = rotated_eigvecs.T
                else:
                    eigvecs = eigvecs.at[group_set].set(np.squeeze(rotated_eigvecs.T))
            else: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                M = np.hstack((rho - eigvals[ind1] * np.eye(dim), -np.expand_dims(eigvecs[ind1].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[ind1].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[ind1].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                if not USE_DIFFRAX:
                    psi_grads[ind1] = np.squeeze(sol[:dim])
                else:
                    psi_grads = psi_grads.at[ind1].set(np.squeeze(sol[:dim]))
                if not USE_DIFFRAX:
                    lambda_grads[ind1] = np.real(sol[dim])
                else:
                    lambda_grads = lambda_grads.at[ind1].set(np.squeeze(np.real(sol[dim])))

    return psi_grads, lambda_grads, eigvecs


def qfi_quotient(lambda_i, lambda_j, lambda_grads):

    dim = np.shape(lambda_grads)[0]

    diff = lambda_i - lambda_j
    sum = lambda_i + lambda_j

    f = diff ** 2 / sum

    if lambda_grads.size == 0:
        return f

    g = np.zeros(dim)
    for k in range(dim):
        dk_lambda_i = lambda_grads[k, 0]
        dk_lambda_j = lambda_grads[k, 1]

        if not USE_DIFFRAX:
            g[k] = np.real((2 * diff * sum * (dk_lambda_i - dk_lambda_j) - (dk_lambda_i + dk_lambda_j) * diff ** 2) / (sum ** 2))
        else:
            g = g.at[k].set(np.real((2 * diff * sum * (dk_lambda_i - dk_lambda_j) - (dk_lambda_i + dk_lambda_j) * diff ** 2) / (sum ** 2)))

    return f, g


def qfi_modulus(G, psi_grads, i, j, eigenvectors):

    dim = np.shape(psi_grads)[0]
    g = np.zeros(dim)

    # WLOG:
    psi_i = eigenvectors[0, i]
    psi_j = eigenvectors[0, j]
    ip = psi_i.conj() @ G @ psi_j.T

    f = np.absolute(ip) ** 2

    if psi_grads.size == 0:
        return f

    for k in range(dim):
        d_xk_psi_i = psi_grads[k, i]
        d_xk_psi_j = psi_grads[k, j]
        psi_i = eigenvectors[k, i]
        psi_j = eigenvectors[k, j]
        der_product = d_xk_psi_i.conj() @ G @ psi_j.T + psi_i.conj() @ G @ d_xk_psi_j.T
        if not USE_DIFFRAX:
            g[k] = 2 * np.real(ip) * np.real(der_product) + 2 * np.imag(ip) * np.imag(der_product)
        else:
            g = g.at[k].set(2 * np.real(ip) * np.real(der_product) + 2 * np.imag(ip) * np.imag(der_product))

    return f, g
