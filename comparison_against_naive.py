import matplotlib.pyplot as plt
import numpy as np
from db_utils import connect_to_db

conn, cursor = connect_to_db()
cursor.execute('''
SELECT e1.experiment_id, e1.method_name, e2.method_name, e1.vus_roc, e2.vus_roc, e1.vus_roc - e2.vus_roc AS difference
FROM experiments e1 JOIN experiments e2
ON e1.files = e2.files
WHERE e1.experiment_id != e2.experiment_id
AND e1.method_name = 'IForest'
AND e2.method_name = 'IForest_PCA'
ORDER BY difference DESC
-- GROUP BY experiment_id
LIMIT 2000
''')

results = cursor.fetchall()
iforest_pca = []
experiment_ids = set()
for experiment_id, method_name1, method_name2, vus_roc1, vus_roc2, _ in results:
    if experiment_id not in experiment_ids:
        iforest_pca.append([vus_roc2, vus_roc1])
    experiment_ids.add(experiment_id)


cursor.execute('''
SELECT e1.experiment_id, e1.method_name, e2.method_name, e1.vus_roc, e2.vus_roc, e1.vus_roc - e2.vus_roc AS difference
FROM experiments e1 JOIN experiments e2
ON e1.files = e2.files
WHERE e1.experiment_id != e2.experiment_id
AND e1.method_name = 'IForest'
AND e2.method_name = 'IForest_TabPFN'
ORDER BY difference DESC
-- GROUP BY experiment_id
LIMIT 250
''')

results = cursor.fetchall()
iforest_tabPFN = []
experiment_ids = set()
for experiment_id, method_name1, method_name2, vus_roc1, vus_roc2, _ in results:
    if experiment_id not in experiment_ids:
        iforest_tabPFN.append([vus_roc2, vus_roc1])
    experiment_ids.add(experiment_id)


cursor.execute('''
SELECT e1.experiment_id, e1.method_name, e2.method_name, e1.vus_roc, e2.vus_roc, e1.vus_roc - e2.vus_roc AS difference
FROM experiments e1 JOIN experiments e2
ON e1.files = e2.files
WHERE e1.experiment_id != e2.experiment_id
AND e1.method_name = 'LOF'
AND e2.method_name = 'LOF_PCA'
ORDER BY difference DESC
-- GROUP BY experiment_id
LIMIT 2000
''')

results = cursor.fetchall()
lof_pca = []
experiment_ids = set()
for experiment_id, method_name1, method_name2, vus_roc1, vus_roc2, _ in results:
    if experiment_id not in experiment_ids:
        lof_pca.append([vus_roc2, vus_roc1])
    experiment_ids.add(experiment_id)


cursor.execute('''
SELECT e1.experiment_id, e1.method_name, e2.method_name, e1.vus_roc, e2.vus_roc, e1.vus_roc - e2.vus_roc AS difference
FROM experiments e1 JOIN experiments e2
ON e1.files = e2.files
WHERE e1.experiment_id != e2.experiment_id
AND e1.method_name = 'LOF'
AND e2.method_name = 'LOF_TabPFN'
ORDER BY difference DESC
-- GROUP BY experiment_id
LIMIT 250
''')

results = cursor.fetchall()
lof_tabPFN = []
experiment_ids = set()
for experiment_id, method_name1, method_name2, vus_roc1, vus_roc2, _ in results:
    if experiment_id not in experiment_ids:
        lof_tabPFN.append([vus_roc2, vus_roc1])
    experiment_ids.add(experiment_id)


# Create a figure with two subplots side by side
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Adjust figure size as needed

# Function to create a single plot
def create_plot(ax, data, xlabel, title, min_value=0.3, max_value:float=1.0):
    # Scatter plot
    x = data[:, 0]
    y = data[:, 1]
    ax.scatter(x, y, color='darkorange', s=80, edgecolor='black', linewidth=1)

    # Diagonal line
    ax.plot([min_value, max_value], [min_value, max_value], color='gray', linestyle='--', linewidth=1)

    # Shaded region
    ax.fill_between([min_value, max_value], [min_value, max_value], max_value, color='lightblue', alpha=0.3, label='SALT better')
    ax.fill_between([min_value, max_value], min_value, [min_value, max_value], color='lightcoral', alpha=0.3, label='Naive batching better')

    # Set axis limits and labels
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("VUS ROC (SALT)", fontsize=16)
    ax.set_title(title, fontsize=18) # Adjusted title position

    # Ticks and grid
    ax.set_xticks(np.arange(min_value, max_value, 0.1))
    ax.set_yticks(np.arange(min_value, max_value, 0.1))
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', length=4, width=1, color='gray')

    # Legend
    ax.legend(loc='upper left', fontsize=16, frameon=False)

# Create the two plots
create_plot(axs[0][0], np.array(iforest_pca), "VUS ROC (naive)", "(a) IForest + PCA", min_value=0.3, max_value=0.9)
create_plot(axs[0][1], np.array(iforest_tabPFN), "VUS ROC (naive)", "(b) IForest + TabPFN", min_value=0.3, max_value=0.85)
create_plot(axs[1][0], np.array(lof_pca), "VUS ROC (naive)", "(c) LOF + PCA", min_value=0.3, max_value=1)
create_plot(axs[1][1], np.array(lof_tabPFN), "VUS ROC (naive)", "(d) LOF + TabPFN", min_value=0.3, max_value=1)

# Add a caption (optional)
# fig.text(0.5, 0.01, "Figure: Comparison of quality before and after transformation.\nPoints above the diagonal indicate improved quality, while points below indicate degradation.", ha='center', va='center', fontsize=14)

plt.tight_layout()  # Adjust layout to make room for the caption\
plt.savefig("comparison_against_naive.png", dpi=700, bbox_inches='tight')  # Save the figure
plt.show()