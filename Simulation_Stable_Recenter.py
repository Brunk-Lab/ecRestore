import numpy as np
import time
from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.special import expit
import concurrent.futures

def death_rate(ecDNA, **kwargs):
    death_mean = kwargs.get('death_mean', 250)
    death_dispersion = kwargs.get('death_dispersion', 100)
    death_bias = kwargs.get('death_bias', 0.01)
    death_scale = kwargs.get('death_scale', 0.1)
    
    if death_bias < 0:
        death_bias = 0

    death_rate = death_bias + death_scale * (1 - np.exp(-((ecDNA - death_mean) ** 2) / (2 * death_dispersion ** 2)))

    return death_rate


def division_rate(ecDNA, time_alive, **kwargs):
    division_mean = kwargs.get('division_mean', 250)
    mean_division_time_scale = kwargs.get('mean_division_time_scale', 31)
    division_scale = kwargs.get('division_scale', 0.02)

    normal_division_rate = 1 / mean_division_time_scale
    pre_normalized = expit(normal_division_rate * (ecDNA - division_mean))
    division_rate = division_scale * pre_normalized

    return division_rate


def gillespie_simulation(resources, neutral,  **kwargs):

    max_count = kwargs.get('max_count', 100000)
    death_enabled = kwargs.get('death_enabled', True)
    input_cells = kwargs.get('input', None)
    timed_death_without_replication = kwargs.get('timed_death_without_replication', 72) # for low ecDNA
    instant_death = kwargs.get('instant_death', 10000000) #no instant death parameter for high ecDNA
    split_probability = kwargs.get('split_probability', 0.5)

    time_limit = 48

    time = 1
    cells = np.column_stack((resources, np.zeros(len(resources))))

    events = []

    while time < time_limit and len(cells) > 0:
        cells = cells[(time - cells[:, 1]) <= timed_death_without_replication]

        if len(cells) == 0:
            break

        division_rates = division_rate(cells[:, 0], time_alive=1, **kwargs)
        death_rates = death_rate(cells[:, 0], **kwargs) if death_enabled else np.zeros_like(cells[:, 0])

        total_rate = division_rates.sum() + death_rates.sum()
        if total_rate <= 0:
            print(f"Change kwargs: need positive rate.")
            break 

        tau = np.random.exponential(1 / total_rate)
        time += tau

        if np.random.rand() < division_rates.sum() / total_rate:
            chosen_index = np.random.choice(len(cells), p=division_rates / division_rates.sum())
            chosen_cell = cells[chosen_index]

            resource = chosen_cell[0]
            new_resource = 2 * resource
            resource_distribution = np.random.binomial(new_resource, split_probability)
            daughter1_resource = resource_distribution
            daughter2_resource = new_resource - resource_distribution

            if daughter1_resource < instant_death:
                cells = np.vstack([cells, [daughter1_resource, time]])
            if daughter2_resource < instant_death:
                cells = np.vstack([cells, [daughter2_resource, time]])

            cells = np.delete(cells, chosen_index, axis=0)

        else:
            chosen_index = np.random.choice(len(cells), p=death_rates / death_rates.sum())
            cells = np.delete(cells, chosen_index, axis=0) 

        events.append((time, len(cells)))

        if len(cells) > max_count:
            return cells, events

    return cells, events


def calculate_binned_kl_divergence(dist1, dist2, bins=100, epsilon=1e-10):
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    if dist1.size == 0 or dist2.size == 0:
        print("Empty Distribution: Overflow")
        return np.inf  

    min_val = min(dist1.min(), dist2.min())
    max_val = max(dist1.max(), dist2.max())

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    hist1, _ = np.histogram(dist1, bins=bin_edges, density=False)
    hist2, _ = np.histogram(dist2, bins=bin_edges, density=False)

    hist1 = hist1.astype(float) + epsilon
    hist2 = hist2.astype(float) + epsilon

    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    kl_divergence = entropy(hist1, hist2)

    if not np.isfinite(kl_divergence):
        kl_divergence = 1e6 

    return kl_divergence

def gillespie_simulation_outline(high, middle, low, neutral, replicates=10, **kwargs):
    
    def run_replicate(_):
        running_loss = 0
        middle_init = middle
        high_init = high
        low_init = low

        for gen in range(3):
            middle_out, _ = gillespie_simulation(middle_init, neutral, **kwargs)
            high_out, _ = gillespie_simulation(high_init, neutral, **kwargs)
            low_out, _ = gillespie_simulation(low_init, neutral, **kwargs)

            if middle_out.size == 0 or high_out.size == 0 or low_out.size == 0:
                return np.inf 
            
            middle_out = middle_out[:, 0]
            high_out = high_out[:, 0]
            low_out = low_out[:, 0]
            
            final_distribution = np.concatenate((middle_out, high_out, low_out))
            initial_distribution = np.concatenate((middle, high, low))
            
            if len(final_distribution) < len(initial_distribution):
                running_loss += 100000
            elif len(final_distribution) < 1.5 * len(initial_distribution):
                running_loss += 15000
            elif len(final_distribution) > 4 * len(initial_distribution):
                running_loss += 100000

            middle_init = middle_out
            high_init = high_out
            low_init = low_out

            running_loss += 1000 * calculate_binned_kl_divergence(initial_distribution, final_distribution)
            
            kl_div_high_middle = calculate_binned_kl_divergence(high_out, middle_out)
            kl_div_middle_low = calculate_binned_kl_divergence(low_out, middle_out)
            
            kl_div_initial_middle = calculate_binned_kl_divergence(middle_out, middle)
    
            running_loss += (5000 * kl_div_initial_middle + 500 * kl_div_high_middle + 500 * kl_div_middle_low)

        return running_loss
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_replicate, range(replicates)))
    
    return -(sum(results) / replicates), kwargs

