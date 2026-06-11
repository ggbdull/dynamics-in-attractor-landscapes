import numpy as np
import matplotlib.pyplot as plt


def plot_results(region_bests, greedy_bests):

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)

    plt.bar(
        ['Region', 'Greedy'],
        [
            np.mean(region_bests),
            np.mean(greedy_bests)
        ],
        yerr=[
            np.std(region_bests),
            np.std(greedy_bests)
        ],
        capsize=5
    )

    plt.title("Comparison")

    plt.subplot(1, 2, 2)

    plt.plot(region_bests, 'o-', label='Region')

    plt.plot(greedy_bests, 's-', label='Greedy')

    plt.legend()

    plt.title("Scores per Run")

    plt.tight_layout()

    plt.savefig("figures/comparison.png", dpi=300)

    plt.show()