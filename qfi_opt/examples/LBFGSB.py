import numpy as np
import ipdb

def LBFGSB(func, x0, l, u, m=10, tol=1e-5, max_iters=20, display=True, xhistory=False):
    # Validate inputs
    #x0 = validate_inputs(func, x0, l, u)

    # Initialize BFGS variables

    n = len(x0)
    Y = np.zeros((n, 0))
    S = np.zeros((n, 0))
    W = np.zeros((n, 1))
    M = np.zeros((1, 1))
    theta = 1

    # Initialize objective variables
    x = x0
    f, g = func(x)

    # Initialize quasi-Newton iterations
    k = 0

    # Print out useful information if specified
    if display:
        print(' iter        f(x)          optimality')
        print('-------------------------------------')
        opt = get_optimality(x, g, l, u)
        print(f'{k:3d} {f:16.8f} {opt:16.8f}')

    # Save the x history if specified
    xhist = []
    if xhistory:
        xhist.append(x0)

    # Perform quasi-Newton iterations
    while (get_optimality(x, g, l, u) > tol) and (k < max_iters):
        # Update search information
        x_old = x
        g_old = g

        # Compute the new search direction
        xc, c = get_cauchy_point(x, g, l, u, theta, W, M)
        xbar, line_search_flag = subspace_min(x, g, l, u, xc, c, theta, W, M)

        #if line_search_flag: # this commented line only makes sense for strong Wolfe.
        d = xbar - x
        c1, c2 = 1e-4, 1 - 1e-4
        alpha_lo, alpha_hi = 0, np.inf
        alpha, f_new, f_old = 1.0, np.inf, f
        dphi0 = g.T @ d
        while alpha > 1e-8 and np.abs(f_new - f_old) > 1e-12:
            xtrial = x + alpha * d
            f_new, g_new = func(xtrial)
            dphiplus = g_new.T @ d
            if f_new > f + alpha * c1 * dphi0:  # Failed to find decrease
                alpha_hi = alpha
                print("Line search continues. Insufficient decrease found. f0 = ", f, 'fnew = ', f_new)
            elif dphiplus < c2 * dphi0:
                alpha_lo = alpha
                print("Line search continues. Weak Wolfe condition not satisfied. f0 = ", f, 'fnew = ', f_new)
            else:
                x = xtrial
                f, g = f_new, g_new
                break

            if alpha_hi < np.inf:
                alpha = (alpha_hi + alpha_lo) / 2
            else:
                alpha = 2 * alpha_lo

        if alpha <= 1e-8 or np.abs(f_new - f_old) <= 1e-12:
            print('Stopping because line search failed - gradient error or nonsmoothness likely to blame.')
            return x, xhist

        # Update LBFGS data structures
        y = g - g_old
        s = x - x_old
        curv = np.dot(s.T, y)  # Keep sign of curvature
        if curv < np.finfo(float).eps:
            ipdb.set_trace()
            print('Warning: negative curvature detected, skipping L-BFGS update')
            k += 1

            continue

        if Y.shape[1] < m:
            Y = np.hstack([Y, y.reshape(-1, 1)])
            S = np.hstack([S, s.reshape(-1, 1)])
        else:
            Y[:, :m - 1] = Y[:, 1:m]
            S[:, :m - 1] = S[:, 1:m]
            Y[:, m-1] = y.T
            S[:, m-1] = s.T

        theta = np.dot(y.T, y) / np.dot(y.T, s)
        W = np.hstack([Y, theta * S])
        A = np.dot(S.T, Y)
        L = np.tril(A, -1)
        D = -np.diag(np.diag(A))
        MM = np.vstack([np.hstack([D, L.T]), np.hstack([L, theta * np.dot(S.T, S)])])
        M = np.linalg.inv(MM)

        # Update the iteration count
        k += 1
        if xhistory:
            xhist.append(x)

        # Print useful information
        if display:
            opt = get_optimality(x, g, l, u)
            print(f'{k:3d} {f:16.8f} {opt:16.8f}')

    if k == max_iters:
        print('Warning: maximum number of iterations reached')
        return x, np.array(xhist)

    if get_optimality(x, g, l, u) < tol:
        print('Stopping because convergence tolerance met!')
        return x, np.array(xhist)



def validate_inputs(func, x0, l, u):
    if not callable(func):
        raise ValueError("Input func must be callable")
    if len(x0.shape) != 1:
        raise ValueError("Input x0 must be a column vector")
    if len(l.shape) != 1 or len(u.shape) != 1:
        raise ValueError("Input l and u must be column vectors")
    if len(l) != len(x0) or len(u) != len(x0):
        raise ValueError("l and u must be of the same length as x0")

    # Project x0 into feasible space
    x0 = np.maximum(np.minimum(x0, u), l)
    return x0


def get_optimality(x, g, l, u):
    projected_g = x - g
    for ind in range(len(x)):
        if projected_g[ind] < l[ind]:
            projected_g[ind] = l[ind]
        elif projected_g[ind] > u[ind]:
            projected_g[ind] = u[ind]
    projected_g = projected_g - x
    return np.max(np.abs(projected_g))


def get_breakpoints(x, g, l, u):
    """
    Compute the breakpoint variables needed for the Cauchy point.
    INPUTS:
        x : [n,1] design vector.
        g : [n,1] objective gradient.
        l : [n,1] lower bound vector.
        u : [n,1] upper bound vector.
    OUTPUTS:
        t : [n,1] breakpoint vector.
        d : [n,1] search direction vector.
        F : [n,1] the indices that sort t from low to high.
    """

    n = len(x)
    t = np.zeros(n)
    d = -g.copy()

    for i in range(n):
        if g[i] < 0:
            t[i] = (x[i] - u[i]) / g[i]
        elif g[i] > 0:
            t[i] = (x[i] - l[i]) / g[i]
        else:
            t[i] = np.finfo(float).max  # Equivalent to MATLAB's realmax

        if t[i] < np.finfo(float).eps:
            d[i] = 0.0

    F = np.argsort(t)

    return t, d, F


def get_cauchy_point(x, g, l, u, theta, W, M):
    """
    Compute the generalized Cauchy point.
    Algorithm CP (as described in the MATLAB code).

    INPUTS:
        x : [n,1] design vector.
        g : [n,1] objective function gradient.
        l : [n,1] lower bound vector.
        u : [n,1] upper bound vector.
        theta : positive BFGS scaling.
        W : [n,2m] BFGS matrix storage.
        M : [2m,2m] BFGSB matrix storage.

    OUTPUTS:
        xc : [n,1] the generalized Cauchy point.
        c  : [2m,1] initialization vector for subspace minimization.
    """

    # Initialization step
    tt, d, F = get_breakpoints(x, g, l, u)
    xc = np.copy(x)
    p = np.dot(W.T, d)
    c = np.zeros((W.shape[1], 1))
    fp = -np.dot(d.T, d)
    fpp = -theta * fp - np.dot(p.T, np.dot(M, p))
    fpp0 = -theta * fp
    dt_min = -fp / fpp
    t_old = 0

    # Initial loop to find 'b'
    for j in range(len(x)):
        i = j
        if F[i] > 0:
            break

    b = F[i]
    t = tt[b]
    dt = t - t_old

    # Examine the subsequent segments
    while (dt_min > dt) and (i < len(x)):
        if d[b] > 0:
            xc[b] = u[b]
        elif d[b] < 0:
            xc[b] = l[b]

        zb = xc[b] - x[b]
        c += dt * p
        gb = g[b]
        wbt = np.expand_dims(W[b, :], 0)

        fp += dt * fpp + gb ** 2 + theta * gb * zb - gb * np.dot(wbt, np.dot(M, c))
        fpp = fpp - theta * gb ** 2 - 2.0 * gb * np.dot(wbt, np.dot(M, p)) - gb ** 2 * np.dot(wbt, np.dot(M, wbt.T))
        fpp = max(np.finfo(float).eps * fpp0, fpp)

        p += gb * wbt.T
        d[b] = 0.0
        dt_min = -fp / fpp
        t_old = t

        i += 1
        if i < len(x):
            b = F[i]
            t = tt[b]
            dt = t - t_old

    # Final updates
    dt_min = max(dt_min, 0)
    t_old += dt_min

    for j in range(i, len(xc)):
        idx = F[j]
        xc[idx] = x[idx] + t_old * d[idx]

    c += dt_min * p
    return xc, c


def subspace_min(x, g, l, u, xc, c, theta, W, M):
    """
    Subspace minimization for the quadratic model over free variables.
    Direct Primal Method.

    INPUTS:
        x : [n,1] design vector.
        g : [n,1] objective gradient vector.
        l : [n,1] lower bound vector.
        u : [n,1] upper bound vector.
        xc : [n,1] generalized Cauchy point.
        c : [2m,1] minimization initialization vector.
        theta : positive LBFGS scaling parameter.
        W : [n,2m] LBFGS matrix storage.
        M : [2m,2m] LBFGS matrix storage.

    OUTPUTS:
        xbar : [n,1] the result of the subspace minimization.
        line_search_flag : bool, indicates if a line search is needed.
    """

    # Set the line search flag to True
    line_search_flag = True

    # Compute the free variables (those not at bounds)
    free_vars_idx = []
    for i in range(len(xc)):
        if (xc[i] != u[i]) and (xc[i] != l[i]):
            free_vars_idx.append(i)

    num_free_vars = len(free_vars_idx)

    if num_free_vars == 0:
        xbar = xc.copy()
        line_search_flag = False
        return xbar, line_search_flag

    # Compute W^T Z, the restriction of W to free variables
    WTZ = np.zeros((len(c), num_free_vars))  # len(c) = 2*m
    if num_free_vars == 1 or len(c) == 0:
        ipdb.set_trace()
    for i in range(num_free_vars):
        WTZ[:, i] = W[free_vars_idx[i], :]

    # Compute the reduced gradient of mk restricted to free variables
    rr = g + theta * (xc - x) - W @ (M @ c)
    r = np.zeros(num_free_vars)
    for i in range(num_free_vars):
        r[i] = rr[free_vars_idx[i]]

    # Form intermediate variables
    invtheta = 1.0 / theta
    v = M @ (WTZ @ r)
    N = invtheta * WTZ @ WTZ.T
    N = np.eye(N.shape[0]) - M @ N
    v = np.linalg.solve(N, v)
    du = -invtheta * r - invtheta ** 2 * WTZ.T @ v
    du = np.squeeze(du)

    # Find alpha star
    alpha_star = find_alpha(l, u, xc, du, free_vars_idx)

    # Compute the subspace minimization
    d_star = alpha_star * du
    xbar = xc.copy()
    for i in range(num_free_vars):
        idx = free_vars_idx[i]
        xbar[idx] = xbar[idx] + d_star[i]

    return xbar, line_search_flag


def find_alpha(l, u, xc, du, free_vars_idx):
    """
    Finds the optimal step size alpha_star based on Equation (5.8).

    INPUTS:
        l : [n,1] lower bound constraint vector.
        u : [n,1] upper bound constraint vector.
        xc : [n,1] generalized Cauchy point.
        du : [num_free_vars,1] solution of unconstrained minimization.
        free_vars_idx : list of indices corresponding to free variables.

    OUTPUT:
        alpha_star : positive scaling parameter.
    """

    alpha_star = 1.0  # Initialize alpha_star to 1
    n = len(free_vars_idx)

    for i in range(n):
        idx = free_vars_idx[i]
        if du[i] > 0:
            alpha_star = min(alpha_star, (u[idx] - xc[idx]) / du[i])
        elif du[i] < 0:
            alpha_star = min(alpha_star, (l[idx] - xc[idx]) / du[i])

    return alpha_star

