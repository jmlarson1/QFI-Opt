global counter allF allX

counter = 0; allF = []; allX = [];

addpath('../../../IBCDFO/pounders/m/');
addpath('../../../IBCDFO/minq/m/minq5/')
fun = @(x)qfi_objective(x, num_params, [], 0);
X0 = x0;
npmax = 2 * num_params + 1;
nfmax = 10 * (num_params + 1);
gtol = 1e-8;
delta = 0.1;
nfs = 1;
m = 1;
F0 = fun(X0);
xkin = 1;
L = -Inf(1, num_params);
U = Inf(1, num_params);
printf = 1;
spsolver = 2;
hfun = @(F)F;
combinemodels = @identity_combine;

[X, F, flag, xkin] = ...
    pounders(fun, X0, num_params, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);