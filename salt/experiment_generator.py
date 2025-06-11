def experiments_generator(
    selected_domains=['NAB', 'YAHOO'],
    nof_experiments_per_category=3,
    public_root="./data/",
    normality_level_min=2,
    normality_level_max=4, # inclusive
    random_seed=42,
):
    """
    Generates a list of experiments with time series data from specified (2) domains.
    Each experiment consists of a list of file paths to the time series data.
    """
    import os
    import random

    experiments = []
    random.seed(random_seed)

    # experiments with concept drift within domains
    for selected_domain in selected_domains:
        dom_path = os.path.join(public_root, selected_domain)
        files = sorted(f for f in os.listdir(dom_path) if f.endswith('.out'))
        for normality_levels in range(normality_level_min, normality_level_max + 1):
            for _ in range(nof_experiments_per_category):
                if len(files) < normality_levels:
                    print(f"Not enough files in {selected_domain} for normality level {normality_level}.")
                    continue
                experiment = []
                for normality_level in range(normality_levels):
                    experiment.append(os.path.join(dom_path, random.choice(files)))
                experiments.append(experiment)

    files_of_first_domain = [f for f in os.listdir(os.path.join(public_root, selected_domains[0])) if f.endswith('.out')]
    files_of_second_domain = [f for f in os.listdir(os.path.join(public_root, selected_domains[1])) if f.endswith('.out')]

    # Experiments with concept drift across domains
    for _ in range(nof_experiments_per_category):
        first_second_drift_experiment = [
            os.path.join(public_root, selected_domains[0], random.choice(files_of_first_domain)),
            os.path.join(public_root, selected_domains[1], random.choice(files_of_second_domain))
        ]
        experiments.append(first_second_drift_experiment)

        second_first_drift_experiment = [
            os.path.join(public_root, selected_domains[1], random.choice(files_of_second_domain)),
            os.path.join(public_root, selected_domains[0], random.choice(files_of_first_domain))
        ]
        experiments.append(second_first_drift_experiment)

    return experiments
