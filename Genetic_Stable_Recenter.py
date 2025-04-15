import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import multiprocessing as mp
from Simulation_Stable_Recenter import gillespie_simulation_outline

population_size = 250
top_k = 20
total_generations = 2000
num_processes = mp.cpu_count()

upsampled_data = pd.read_csv('upsampled_distributions.csv')

high_total_ecdna = np.array(upsampled_data['top'].dropna().tolist())
sortedcntrl_total_ecdna = np.array(upsampled_data['middle'].dropna().tolist())
low_total_ecdna = np.array(upsampled_data['bottom'].dropna().tolist())

all_total_ecdna = np.concatenate((high_total_ecdna, sortedcntrl_total_ecdna, low_total_ecdna)).astype(int)

def gillespie_simulation_wrapper(params):
    return gillespie_simulation_outline(
        high_total_ecdna,
        sortedcntrl_total_ecdna,
        low_total_ecdna,
        False,
        **params
    ) 

hyperparameters = [
    {
        'death_mean': random.uniform(200, 300),
        'death_dispersion': random.uniform(80, 120),
        'death_scale': random.uniform(0.002, 0.01),
        'death_bias': 0,
        'division_mean': random.uniform(200, 300),
        'mean_division_time_scale': random.uniform(80, 120),
        'division_scale': random.uniform(0.01, 0.05),
        'timed_death_without_replication': 200,
        'split_probability': random.uniform(0.35, 0.65)
    } for _ in range(population_size)
]

population = hyperparameters
csv_data = []

average_fitness_per_generation = []
execution_times = []

for generation in range(total_generations):
    start_time = time.time()
    print(f"Generation: {generation}")
    
    with mp.Pool(num_processes) as pool:
        results = pool.map(gillespie_simulation_wrapper, population)
    results = [res for res in results if np.isfinite(res[0]) and res[0] < 0]

    if not results:
        print("Change KWARGS: not valid")
        break

    results.sort(reverse=True, key=lambda x: x[0])
    top_k_individuals = results[:top_k]

    if len(top_k_individuals) < 2:
        print("Not enough valid people to create next gen")
        break

    avg_fitness = np.mean([ind[0] for ind in top_k_individuals])
    max_fitness = top_k_individuals[0][0]
    best_params = top_k_individuals[0][1]

    average_fitness_per_generation.append(avg_fitness)

    for res in results:
        fitness_function, kwargs = res
        csv_data.append({
            'generation': generation,
            'kwargs': kwargs,
            'fitness': fitness_function,
        })

    end_time = time.time()
    generation_time = end_time - start_time
    execution_times.append(generation_time)

    print(f"Generation {generation} completed in {generation_time:.2f} seconds.")
    print(f"Max Fitness: {max_fitness}, Best Params: {best_params}")

    new_population = []
    for _ in range(population_size):
        parent1 = random.choice(top_k_individuals[:top_k // 2])  
        parent2 = random.choice(top_k_individuals[top_k // 2:])
        
        parent1_kwargs, parent2_kwargs = parent1[1], parent2[1]
        new_individual = {k: (parent1_kwargs.get(k, 0) + parent2_kwargs.get(k, 0)) / 2 for k in parent1_kwargs}
        mutation_strength = 0.05
        for param in new_individual:
            if param != 'split_probability':
                new_individual[param] *= np.random.uniform(1 - mutation_strength, 1 + mutation_strength)
            else:
                new_value = new_individual[param] * np.random.uniform(1 - mutation_strength, 1 + mutation_strength)
                new_individual[param] = min(max(new_value, 0), 1)

        new_population.append(new_individual)

    population = new_population

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('genetic_algorithm_results_new.csv', index=False)

plt.figure()
plt.plot(average_fitness_per_generation)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Average Fitness per Generation')
plt.savefig('average_fitness_per_generation_recentered.png')
plt.close()

plt.figure()
plt.plot(execution_times)
plt.xlabel('Generation')
plt.ylabel('Time (seconds)')
plt.title('Time per Generation')
plt.savefig('execution_time_per_generation_recentered.png')
plt.close()