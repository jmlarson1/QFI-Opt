function [returned_val, returned_val2] = qfi_objective(x, num_params, emptything, output)

    %SLQP-GS only:
    %x = x';
    global num_spins dissipation allX allF counter

    % get the density matrix and the approximate derivatives
    if output >= 1
        [rho0, G, drho] = get_rho_and_grad(x, "g", num_spins, dissipation); 
        %[rho0, G, drho] = get_rho(x, "g");
    elseif output == 0
        [rho0, G] = get_rho_and_grad(x, "f", num_spins, dissipation);
        %[rho0, G] = get_rho(x, "f");
    end

    [~, N, ~] = size(rho0); %[num_params, N, ~] = size(drho);

    lambda = zeros(num_params, N);
    psi = zeros(num_params, N, N);
    if output >= 1
        lambda_grads = zeros(num_params, N);
        psi_grads = zeros(num_params, N, N);
        for k = 1:num_params
            %[dLambda, dV, V, eigvals] = fullmatrixder(rho0, squeeze(drho(k,:,:)), ...
             %   squeeze(d2rho(k,:,:)), 1e-8);
            [dLambda, dV, V, eigvals] = fullmatrixder2(rho0, squeeze(drho(k,:,:)), 1e-8); 
            %[dLambda, dV, V, eigvals] = fullmatrixderlazy(rho0, squeeze(drho(k,:,:)));
            lambda(k, :) = eigvals;
            psi(k, :, :) = V;
            lambda_grads(k, :) = dLambda;
            psi_grads(k, :, :) = dV;
        end
    elseif output == 0
        rho0 = (rho0 + rho0') / 2;
        [V, D] = eig(rho0);
        [eigvals, sort_indices] = sort(real(diag(D)));
        psi(1, :, :) = V(:, sort_indices);
        lambda(1, :) = eigvals;
        
    end

    f = 0;
    g = zeros(1, num_params);

    for i = 1:N
        for j = (i+1):N
            denom = lambda(1, i) + lambda(1, j);
            diff = lambda(1, i) - lambda(1, j);
            if denom > 1e-8 && abs(diff) > 1e-8
                % quotient term:
                numer = (diff)^2;
                quotient = numer / denom;

                % squared modulus term:
                % somewhat arbitrary, but use the first coord's basis of psi:
                % need to check difference of using different bases
                ip_term = psi(1,:,i) * G * psi(1,:,j)';                
                squared_modulus = norm(ip_term)^2;

                f = f + quotient * squared_modulus; 
                if output >= 1
                    for k = 1:num_params
                        g(k) = g(k) + kth_partial(quotient, squared_modulus, ...
                            lambda(1,i), lambda(1,j), lambda_grads(k,i), lambda_grads(k,j),...
                            psi(k, :, i), psi(k, :, j), psi_grads(k, :, i), psi_grads(k, :, j), G);
                    end
                end
            end
        end % for j loop
    end % for i loop

    if output == 0
        returned_val = -4 * real(f); 
        returned_val2 = 0;
    elseif output == 1
        returned_val = -4 * real(g)';
        returned_val2 = 0;
    elseif output == 2
        returned_val = -4 * real(f);
        returned_val2 = -4 * real(g)';
    end

    % global manipulations
    counter = counter + 1;
    allX(counter,:) = x;
    allF(counter) = 4 * real(f);
    
end

function der = kth_partial(quotient, squared_modulus, lambda_i, lambda_j, dlambda_i, dlambda_j, psi_i, psi_j, dpsi_i, dpsi_j, G)
    quotient_der = quotient_partial(lambda_i, lambda_j, dlambda_i, dlambda_j);
    modulus_der = modulus_partial(psi_i, psi_j, dpsi_i, dpsi_j, G);

    der = quotient * modulus_der + squared_modulus * quotient_der;
end

function der = quotient_partial(lambda_i, lambda_j, dlambda_i, dlambda_j)
    squared_diff = (lambda_i - lambda_j)^2;
    fprimeg = 2 * (lambda_i^2 - lambda_j^2) * (dlambda_i - dlambda_j);
    gprimef = squared_diff * (dlambda_i + dlambda_j);
    der = (fprimeg - gprimef) / (lambda_i + lambda_j)^2;
end

function der = modulus_partial(psi_i, psi_j, dpsi_i, dpsi_j, G)
    ip = psi_i * G * psi_j';
    left_der = dpsi_i * G * psi_j';
    right_der = psi_i * G * dpsi_j';

    real_der = 2 * real(ip) * real(left_der + right_der);
    imag_der = 2 * imag(ip) * imag(left_der + right_der);

    der = real_der + imag_der;
end