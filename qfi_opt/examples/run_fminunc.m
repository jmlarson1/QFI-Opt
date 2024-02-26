global counter allF allX

counter = 0; allF = []; allX = [];

options = optimoptions("fminunc");
options.SpecifyObjectiveGradient = true;
options.Display = 'iter';

fun = @(x)qfi_objective(x, num_params, [], 2);

[x,f,flag,outstruct,g] = fminunc(fun,x0,options);