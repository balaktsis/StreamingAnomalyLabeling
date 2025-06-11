import matplotlib.pyplot as plt
import numpy as np
from db_utils import connect_to_db

conn, cursor = connect_to_db()

sand_vus_roc, iforest_vus_roc, lof_vus_roc = [], [], []
sand_runtime, iforest_runtime, lof_runtime = [], [], []
metric = 'recall'

for normality_level in [2, 3, 4]:
    sand_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'SAND'
    AND tag = 'batch'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 10)
    '''

    iforest_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'IForest_PCA'
    AND tag = 'online'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 100)
    '''

    lof_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'LOF_PCA'
    AND tag = 'online'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 100)
    '''

    cursor.execute(sand_query)
    sand_result = cursor.fetchone()
    sand_vus_roc.append(sand_result[0])

    cursor.execute(iforest_query)
    iforest_result = cursor.fetchone()
    iforest_vus_roc.append(iforest_result[0])

    cursor.execute(lof_query)
    lof_result = cursor.fetchone()
    lof_vus_roc.append(lof_result[0])

metric = 'execution_time'
for normality_level in [2, 3, 4]:
    sand_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'SAND'
    AND tag = 'batch'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 10)
    '''

    iforest_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'IForest_PCA'
    AND tag = 'online'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 100)
    '''

    lof_query = f'''
    SELECT AVG({metric}) from ( 
    SELECT {metric}
    from experiments 
    WHERE method_name = 'LOF_PCA'
    AND tag = 'online'
    AND normality_levels = {normality_level}
    AND {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT 100)
    '''

    cursor.execute(sand_query)
    sand_result = cursor.fetchone()
    sand_runtime.append(sand_result[0])

    cursor.execute(iforest_query)
    iforest_result = cursor.fetchone()
    iforest_runtime.append(iforest_result[0])

    cursor.execute(lof_query)
    lof_result = cursor.fetchone()
    lof_runtime.append(lof_result[0])


colors = ['#4878D0', '#EE854A', '#6ACC64', '#9d6acc', '#956CB4', '#8C613C', '#DC7EC0', '#797979', '#D5BB67', '#82C6E2']
plt.figure(figsize=(12, 3))


plt.subplot(121)
plt.plot([2, 3, 4], sand_vus_roc, label='SAND', marker='o', color=colors[0], linewidth=2, linestyle='--', markersize=6)
plt.plot([2, 3, 4], iforest_vus_roc, label='IForest PCA', marker='^', color=colors[1], linewidth=2)
plt.plot([2, 3, 4], lof_vus_roc, label='LOF PCA', marker='s', color=colors[2], linewidth=2)

plt.xlabel('Normality Level')
plt.ylabel(f'Average Recall')
# plt.suptitle('Comparison between SALT and SAND', fontsize=16)
plt.title('Recall Comparison')
plt.xticks([2, 3, 4], ['2', '3', '4'])
# plt.legend()

plt.subplot(122)
plt.plot([2, 3, 4], sand_runtime, label='SAND', marker='o', color=colors[0], linewidth=2, linestyle='--', markersize=6)
plt.plot([2, 3, 4], iforest_runtime, label='IForest PCA', marker='^', color=colors[1], linewidth=2)
plt.plot([2, 3, 4], lof_runtime, label='LOF PCA', marker='s', color=colors[2], linewidth=2)

plt.xlabel('Normality Level')
plt.ylabel(f'Avg. Exec. Time (ms)')
plt.title('Execution Time Comparison')
plt.xticks([2, 3, 4], ['2', '3', '4'])
plt.legend(loc='lower center', bbox_to_anchor=(-0.5, -1.2), ncol=3, fontsize=12, frameon=False)




plt.tight_layout()  # Adjust layout to make room for the caption\
plt.savefig("comparison_against_SAND.png", dpi=700, bbox_inches='tight')  # Save the figure
plt.show()