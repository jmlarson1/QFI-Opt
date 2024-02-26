global num_spins dissipation allF allX

% set these problem-defining globals (i know, ick)
num_spins = 4;

num_dissipations = 50;
least_dissipation = 0.1;
greatest_dissipation = 5;
dissipations = linspace(least_dissipation, greatest_dissipation, num_dissipations);

% must be set to 4 or 5. for now, 4 uses simulate_OAT and 5 uses
% simulate_TAT
num_params = 4;

% an initial point 
x0 = 0.5*ones(1, num_params);

% for storing data
budget = 10 * (num_params + 1);
H = NaN * ones(budget, num_dissipations, 2);

for j = 1:num_dissipations    
    dissipation = dissipations(j);
    fprintf('Dissipation: %.2f \n', dissipation)
    
    run_pounders;
    num_evals = length(allF);
    if num_evals <= budget
        H(1:num_evals, j, 1) = allF;
    else
        H(1:budget, j, 1) = allF(1:budget);
    end

    run_fminunc;
    num_evals = length(allF);
    if num_evals <= budget
        H(1:num_evals, j, 2) = allF;
    else
        H(1:num_evals, j, 2) = allF(1:budget);
    end
    save('lbfgs_vs_pounders.mat', 'H');

    % start next run from fminunc's solution. 
    x0 = x; 
end