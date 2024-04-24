import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    N = 4
    seed = 88
    for dissipation_rate in np.linspace(0.1, 5, 20):
        for num_params in [5]:  # [4, 5]:
            match num_params:
                case 4:
                    models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
                    # models = ["simulate_OAT"]
                case 5:
                    models = ["simulate_TAT", "simulate_local_TAT_chain"]
                    # models = ["simulate_TAT"]

            for model in models:
                f1 = np.loadtxt(f"results/vals_pounder_N={N}_seed={seed}_model={model}_dissipation={dissipation_rate}")
                f2 = np.loadtxt(f"results/vals_nlopt_N={N}_seed={seed}_model={model}_dissipation={dissipation_rate}")

                plt.figure()  # This ensures a new figure is used for each loop iteration
                plt.plot(f1, label="pounder")
                plt.plot(f2, label="nlopt")
                plt.legend()
                plt.savefig(f"figures/N={N}_seed={seed}_model={model}_dissipation={dissipation_rate}.png", dpi=300)
                plt.close()
