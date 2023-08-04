#!/usr/bin/env python3
import matplotlib.pyplot as plt

import qfi_opt

params = [0.85255953, 1.0, 2.75901551, 1.66809236, 1.00000226]
num_qubits = 4
dissipation_rates = 0

state = qfi_opt.spin_models.simulate_TAT(params, num_qubits, dissipation_rates=dissipation_rates)

cat_state_fidelity = 0.5 * sum(abs(state[ii, jj]) for ii in [0, -1] for jj in [0, -1])
print("cat state fidelity:", cat_state_fidelity)


qfi_opt.plot.husimi(state)
qfi_opt.plot.histogram(state)
plt.show()
