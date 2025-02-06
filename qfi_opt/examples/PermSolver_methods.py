import os
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
    result=expm(1j*rotmat*theta)@rho@expm(-1j*rotmat*theta)
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
            for k in range(0,2*i+1):
                result.append(rho0[i][j][k])
    return result


def recoverrhomat(rho0, N):
    result=[]
    count=0
    for i in range(0,N+1):
        result.append(np.zeros((2*i+1,2*i+1),dtype=np.complex128))
        for mj1 in range(0,2*i+1):
            for mj2 in range(0,2*i+1):
                result[i][mj1][mj2]=rho0[count]
                # if (mj1!=mj2):
                #     result[i][mj2][mj1]=np.conj(rho0[count])
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
                for mj2 in range(0,2*i+1):
                    if USE_DIFFRAX == False:
                        result[time][i][mj1][mj2]=rho0[count][time]
                    else:
                        result[time][i] = result[time][i].at[mj1,mj2].set(rho0[count][time])
                    # if (mj1!=mj2):
                    #     result[time][i][mj2][mj1]=np.conj(rho0[count][time])
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
def simulate(params:np.ndarray, num_qubits:int, Hamiltonian_set:list, dissipation_rates:tuple|float=0.0, dissipation_format:str='XYZ'):
    assert dissipation_format in ['XYZ', 'PMZ'], "dissipation format distinct from preset formats"
    Jmax = num_qubits//2
    Nsteps = 101
    Sx, Sy, Sz, Sx2, Sy2, Sz2 = operator_moments(Jmax)
    Hmat, Hmatloc, dimension = Hamiltonian_set[0], Hamiltonian_set[1], Hamiltonian_set[2]

    # start all spins down
    rho_init = UnitaryGate(Init_rho(Jmax), Sx, np.pi, Jmax)

    if dissipation_format == "XYZ":
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            Dmat, Dmatloc, dimension = matrix.isotropic_DisMat(dissipation_rates, dissipation_rates, dissipation_rates, Jmax)
        else:
            Dmat, Dmatloc, dimension = matrix.isotropic_DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                               Jmax)
    else:
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            Dmat, Dmatloc, dimension = matrix.DisMat(dissipation_rates, dissipation_rates, dissipation_rates,
                                                     0, 0, 0, Jmax)
        else:
            Dmat, Dmatloc, dimension = matrix.DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                     0, 0, 0, Jmax)

    # set up initial rotation axis -> params[1]
    axis_factor = 1
    cos_component = np.cos(axis_factor * np.pi * params[1])
    sin_component = np.sin(axis_factor * np.pi * params[1])
    Sphi1 = []
    for jj in range(Jmax + 1):
        Sphi1.append(Sx[jj] * cos_component + Sy[jj] * sin_component)

    # rotate state for angle params[0] * pi about axis set by params[1] * pi
    rot1 = UnitaryGate(rho_init, Sphi1, -params[0] * np.pi, Jmax)


    # Entangle!
    # if entangling time is not zero, entangle
    if params[2] != 0:
        rho1_f = flatrhomat(rot1, Jmax)
        sol = matrix.Perm_solver(rho1_f, params[2] * np.pi, Dmat, Dmatloc, Hmat, Hmatloc, dimension, Nsteps)
        ent1 = recoverrhomat2(sol.y, Jmax, Nsteps)
        out_state = ent1[-1]


    else:
        out_state = rot1

    # set up final rotation axis -> params[4]
    cos_component = np.cos(axis_factor * np.pi * params[4])
    sin_component = np.sin(axis_factor * np.pi * params[4])
    Sphi2 = []
    for jj in range(Jmax + 1):
        Sphi2.append(Sx[jj] * cos_component + Sy[jj] * sin_component)

    # rotate state for angle params[3] * pi about axis set by params[4] * pi
    return UnitaryGate(out_state, Sphi2, params[3] * np.pi, Jmax)


def simulate_layers(params:np.ndarray, num_qubits:int, Hamiltonian_set:list, dissipation_rates:tuple|float=0.0, dissipation_format:str='XYZ'):
    assert dissipation_format in ['XYZ', 'PMZ'], "dissipation format distinct from preset formats"
    if len(params) < 5 or not len(params) % 2:
        raise ValueError(f"The number of parameters should be an odd number >=5, not {len(params)}.")

    Jmax = num_qubits//2
    Nsteps = 11
    Sx, Sy, Sz = operator_moments(Jmax, return_first_moments_only=True)
    Hmat, Hmatloc, dimension = Hamiltonian_set[0], Hamiltonian_set[1], Hamiltonian_set[2]

    # set all spins down
    rho_init = UnitaryGate(Init_rho(Jmax), Sx, np.pi, Jmax)

    if dissipation_format == "XYZ":
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            Dmat, Dmatloc, dimension = matrix.isotropic_DisMat(dissipation_rates, dissipation_rates, dissipation_rates, Jmax)
        else:
            Dmat, Dmatloc, dimension = matrix.isotropic_DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                               Jmax)
    else:
        if type(dissipation_rates) == float or type(dissipation_rates) == np.float64:
            Dmat, Dmatloc, dimension = matrix.DisMat(dissipation_rates, dissipation_rates, dissipation_rates,
                                                     0, 0, 0, Jmax)
        else:
            Dmat, Dmatloc, dimension = matrix.DisMat(dissipation_rates[0], dissipation_rates[1], dissipation_rates[2],
                                                     0, 0, 0, Jmax)

    # set up initial rotation axis -> params[1]
    axis_factor = 1
    cos_component = np.cos(axis_factor * np.pi * params[1])
    sin_component = np.sin(axis_factor * np.pi * params[1])
    Sphi1 = []
    for jj in range(Jmax + 1):
        Sphi1.append(Sx[jj] * cos_component + Sy[jj] * sin_component)

    # rotate state for angle params[0] * pi about axis set by params[1] * pi
    state = UnitaryGate(rho_init, Sphi1, -params[0] * np.pi, Jmax)

    # Entangle-rotate
    for pp in range(2, len(params) - 1, 2):

        # if entangling time is not zero, entangle
        if params[pp] > 0:
            state_f = flatrhomat(state, Jmax)
            sol = matrix.Perm_solver(state_f, params[pp] * np.pi, Dmat, Dmatloc, Hmat, Hmatloc, dimension, Nsteps)
            if USE_DIFFRAX == False:
                state = recoverrhomat2(sol.y, Jmax, Nsteps)[-1]
            else:
                state = recoverrhomat2(sol, Jmax, Nsteps)[-1]
            # out_state = ent1[-1]

        state = UnitaryGate(state, Sx, -params[pp + 1] * np.pi, Jmax)

    # final rotation about Y
    state = UnitaryGate(state, Sy, -params[-1] * np.pi, Jmax)

    return state

#def simulate_layers(params:np.ndarray, num_qubits:int, Hamiltonian_set:list, dissipation_rates:tuple|float=0.0, dissipation_format:str='XYZ'):
#def simulate_OAT(
#    params: Sequence[float] | np.ndarray,
#    num_qubits: int,
#    *,
#    dissipation_rates: float | tuple[float, float, float] = 0.0,
#    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
#)
def get_jacobian_func(simulate_func):
    """Convert a simulation method into a function that returns its Jacobian."""

    if USE_DIFFRAX:
        print("USE_DIFFRAX and FORWARD_MODE")
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
        shifted_results1 = [ ] 
        shifted_results2 = [ ] 
        shifted_results = [ shifted_results1, shifted_results2] 
        print(result_at_params[0].shape, result_at_params[1].shape )
        for idx, step_size in enumerate(param_step_sizes):
            new_params = list(params)
            new_params[idx] += step_size
            result_at_params_with_step = simulate_func(new_params, *args, **kwargs)
            #print("len(result_at_params_with_step)", len(result_at_params_with_step))
            for i in range(len(result_at_params_with_step)):
                shifted_results[i].append((result_at_params_with_step[i] - result_at_params[i])/ step_size)
            #shifted_results.append(res)
        print(shifted_results[1])
        #return shifted_results
        #return np.stack(shifted_results, axis=-1)
        print("result_at_params[0].shape, params.shape", (params.shape + result_at_params[0].shape))
        return [np.array(shifted_results[0]).reshape((params.shape + result_at_params[0].shape )), 
                np.array(shifted_results[1]).reshape((params.shape + result_at_params[1].shape)) ]

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