import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e

def fitness(pop):
    return np.array([ackley(ind) for ind in pop])

def discrete_recombination(parents):
    num_parents, dim = parents.shape
    offspring = np.zeros_like(parents)
    for i in range(num_parents):
        p1, p2 = parents[np.random.choice(num_parents, 2, replace=False)]
        mask = np.random.rand(dim) > 0.5
        offspring[i] = np.where(mask, p1, p2)
    return offspring

def self_adaptive_mutation(pop, sigma):
    tau = 1 / np.sqrt(2 * np.sqrt(pop.shape[1]))
    tau_prime = 1 / np.sqrt(2 * pop.shape[1])
    new_sigma = sigma * np.exp(tau_prime * np.random.randn(*sigma.shape) + tau * np.random.randn(*sigma.shape))
    pop_mutated = pop + new_sigma * np.random.randn(*pop.shape)
    return pop_mutated, new_sigma

def evolution_strategy(n, pop_size=40, max_iter=100, domain=(-5, 5), return_history=False):
    pop = np.random.uniform(domain[0], domain[1], (pop_size, n))
    sigma = np.random.uniform(0.1, 0.5, (pop_size, n))
    best_fitness = []
    history = []

    for _ in range(max_iter):
        offspring = discrete_recombination(pop)
        offspring, sigma_offspring = self_adaptive_mutation(offspring, sigma)
        combined = np.vstack((pop, offspring))
        combined_sigma = np.vstack((sigma, sigma_offspring))
        fit_vals = fitness(combined)
        best_indices = np.argsort(fit_vals)[:pop_size]
        pop = combined[best_indices]
        sigma = combined_sigma[best_indices]
        best_fitness.append(fit_vals[best_indices[0]])

        if return_history:
            history.append((pop.copy(), fit_vals[best_indices].copy()))

    if return_history:
        return pop[0], best_fitness, history
    else:
        return pop[0], best_fitness






def plot_ackley():
    x = y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley(np.array([i, j])) for i, j in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Suprafață 3D Ackley
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')

    # Punctul de minim global (0, 0)
    ax.scatter(0, 0, ackley(np.array([0, 0])), color='red', s=80, marker='*', label='Global Minimum')

    ax.set_title("Ackley Function Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.legend()
    plt.tight_layout()
    plt.show()



# Plot fitness evolution
def plot_fitness_evolution(all_fitnesses, dimensions, target=1e-3):
    plt.figure(figsize=(10, 5))
    for fit, n in zip(all_fitnesses, dimensions):
        fit = np.array(fit)
        improvement = fit[0] - fit
        plt.plot(improvement, label=f'n={n}')

        # adaugă linie verticală când atinge target
        gen_hit = generation_to_target(fit, target)
        if gen_hit is not None:
            plt.axvline(gen_hit, linestyle='--', color='gray', alpha=0.6)
            plt.text(gen_hit + 1, improvement[gen_hit],
                     f'{gen_hit} gen', fontsize=9,
                     color='gray', rotation=90, va='bottom')

    plt.title(f"Evolution of fitness over generations\n")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Improvement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_fitness_boxplot(dimensions, all_fitnesses):
    sns.set(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("Set2")

    for i, (dim, fit_list) in enumerate(zip(dimensions, all_fitnesses)):
        data = pd.DataFrame({"Fitness": fit_list})

        # Calcul statistici
        mean_val = np.mean(fit_list)
        median_val = np.median(fit_list)
        std_val = np.std(fit_list, ddof=1)
        q1 = np.percentile(fit_list, 25)
        q3 = np.percentile(fit_list, 75)

        # Figură individuală
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(fit_list, vert=False, patch_artist=True, widths=0.6,
                   boxprops=dict(facecolor=palette[i % len(palette)], color='black'),
                   medianprops=dict(color='black'))

        # Linie pentru medie
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        # Linie pentru mediană
        ax.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.4f}')

        # Legendă
        ax.legend(loc='upper right')

        # Casetă cu statistici
        stats_text = (
            f"Mean: {mean_val:.4f}\n"
            f"Median: {median_val:.4f}\n"
            f"Standard Deviation: {std_val:.4f}\n"
            f"Q1: {q1:.4f}\n"
            f"Q3: {q3:.4f}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=props)

        # Titluri și etichete
        ax.set_title(f"Boxplot of Fitness (n = {dim})", fontsize=13)
        ax.set_xlabel("Fitness")
        ax.set_yticks([1])
        ax.set_yticklabels([""])

        plt.tight_layout()
        plt.show()

def generation_to_target(fitness_history, target=1e-3):
    for i, val in enumerate(fitness_history):
        if val <= target:
            return i
    return None


def print_convergence_table(all_fitnesses, dimensions, target=1e-3):
    print("\n Convergense Table for Ackley Function")
    print("-" * 50)
    print(f"{'n':>5} | {'Generation target':>20}")
    print("-" * 50)
    for dim, fit in zip(dimensions, all_fitnesses):
        gen = generation_to_target(fit, target)
        if gen is not None:
            print(f"{dim:>5} | {gen:>20} ")
        else:
            print(f"{dim:>5} | {'Not reached':>20}")
    print("-" * 50)



dimensions = [2, 3, 30]
all_fitnesses = []
history_n2 = None

for dim in dimensions:
    if dim == 2:
        best_sol, fitnesses, history = evolution_strategy(dim, return_history=True)
        history_n2 = history
    else:
        best_sol, fitnesses = evolution_strategy(dim)
    all_fitnesses.append(fitnesses)



plot_ackley()  # Vizualizare statică 3D
plot_fitness_evolution(all_fitnesses, dimensions)
plot_fitness_boxplot(dimensions, all_fitnesses)
print_convergence_table(all_fitnesses, dimensions)



