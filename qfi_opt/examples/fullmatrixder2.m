function [dLambda, dV, V, eigvals] = fullmatrixder2(rho, drho, tol)
    
    n = size(rho, 1);
    dLambda = zeros(1, n);
    dV = zeros(n);

    if nargin == 1
        tol = 1e-8;
    end

    % rho and its derivatives are generally not exactly Hermitian 
    % (but they should be, in theory)
    rho = (rho + rho')/2;
    drho = (drho + drho')/2;

    [V, D] = eig(rho);
    eigvals = real(diag(D)); % this assumes rho hermitian.
    [eigvals, sort_inds] = sort(eigvals);
    V = V(:, sort_inds);

    % group the sorted eigvals
    current_ind = 1;
    for i = 1:n
        if current_ind == i
            for j = (i + 1):n
                if eigvals(j) - eigvals(i) > tol
                    break % break for loop
                end
            end
            % just broke loop, so:
            current_ind = j;
            % special case handling:
            if isempty(j)
                j = n + 1;
            end
            [dLambdasub, dVsub] = sylvestersolve(V, D, drho, i:(j - 1));
            dLambda(i:(j - 1)) = dLambdasub;
            dV(:, i:(j - 1)) = dVsub;
        end
    end

end