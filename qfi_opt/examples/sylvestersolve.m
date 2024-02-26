function [dLambda, dV] = sylvestersolve(V, D, drho, inds)

    n = size(D, 1);
    notinds = setdiff(1:n, inds);
    A1 = D(inds, inds);
    A2 = D(notinds, notinds);
    C = V(:,inds)' * drho * V(:,notinds);
    S = sylvester(A1, -A2, C);
    D2 = S * V(:, notinds)';

    dV = D2'; 

    % average eigenvalue derivative:
    r = length(inds);
    dLambda = (1/r) * trace(V(:, inds)' * drho * V(:, inds));
    dLambda = dLambda * ones(1, r);

end