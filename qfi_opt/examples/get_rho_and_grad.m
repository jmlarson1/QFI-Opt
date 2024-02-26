function [rho0, G, drho] = get_rho_and_grad(x, sense, num_spins, dissipation)

num_params = length(x);
str = '/usr/local/bin/python3 calculate_rho_and_grad.py "' + sense + '" ';
str = str + num2str(num_spins) + ' ' + num2str(dissipation) + ' ' + num2str(num_params) + ' ';
str = str + num2str(x,'%16.16f ');
system(str);

filename = 'rho_center.mat';
load(filename, 'rho', 'G');
rho0 = rho;
system(['rm ' filename]);


if strcmp(sense, "g")  
    filename = strcat('rho_grad.mat');
    load(filename, 'drho');
    system(['rm ' filename]);
end

end