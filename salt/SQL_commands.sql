SELECT method_name, COUNT(*) FROM experiments GROUP BY method_name;

SELECT AVG(vus_roc) AS avg_sand from experiments
WHERE method_name = 'SAND'
AND tag = 'batch'
AND normality_levels = 3
AND vus_roc IS NOT NULL
;

SELECT AVG(vus_roc) AS avg_IFPCA from
(
SELECT vus_roc
from experiments
WHERE method_name = 'IForest_PCA'
AND tag = 'online'
AND normality_levels = 3
AND vus_roc IS NOT NULL
ORDER BY vus_roc DESC
LiMIT 10
)
;

SELECT AVG(vus_roc) AS avg_LOF from experiments
WHERE method_name = 'LOF'
AND tag = 'batch'
AND normality_levels = 3
AND vus_roc IS NOT NULL
;


SELECT e1.experiment_id, e1.method_name, e2.method_name, e1.vus_roc, e2.vus_roc, e1.vus_roc - e2.vus_roc AS difference
FROM experiments e1 JOIN experiments e2
ON e1.files = e2.files
WHERE e1.experiment_id != e2.experiment_id
AND e1.method_name = 'IForest'
AND e2.method_name = 'IForest_PCA'
ORDER BY difference DESC
-- GROUP BY experiment_id
LIMIT 3
;


SELECT AVG(vus_roc) AS avg_iforest_static
from experiments
where method_name = 'IForest'
and tag = 'offline'
and vus_roc IS NOT NULL
;

SELECT AVG(vus_roc) AS avg_iforest_batch
from experiments
where method_name = 'IForest'
and tag = 'batch'
and vus_roc IS NOT NULL
;

SELECT AVG(vus_roc) AS avg_iforest_TabPFN
from
(
SELECT * FROM experiments
where method_name = 'IForest_TabPFN'
and tag = 'online'
and vus_roc IS NOT NULL
ORDER BY vus_roc DESC
--LIMIT 15
)
;

SELECT AVG(vus_roc) AS avg_iforest_PCA
-- SELECT extra_info->>'n_clusters'
from
(
SELECT * FROM experiments
where method_name = 'IForest_PCA'
and tag = 'online'
and vus_roc IS NOT NULL
and CAST(extra_info->>'n_clusters' AS INTEGER) <= 10
and CAST(extra_info->>'state_size' AS INTEGER) = 1000
and CAST(extra_info->>'n_dimensions' AS INTEGER) = 3
-- ORDER BY vus_roc DESC
-- LIMIT 10
-- LIMIT (SELECT COUNT(*)/2 FROM experiments WHERE method_name = 'IForest_PCA' and tag = 'online' and vus_roc IS NOT NULL)
)
;

--------------------

SELECT AVG(vus_roc) AS avg_LOF_static
from experiments
where method_name = 'LOF'
and tag = 'offline'
and vus_roc IS NOT NULL
;

SELECT AVG(vus_roc) AS avg_LOF_batch
from experiments
where method_name = 'LOF'
and tag = 'batch'
and vus_roc IS NOT NULL
;

SELECT AVG(vus_roc) AS avg_LOF_TabPFN
from
(
SELECT * FROM experiments
where method_name = 'LOF_TabPFN'
and tag = 'online'
and vus_roc IS NOT NULL
ORDER BY vus_roc DESC
-- LIMIT 15
)
;

SELECT AVG(vus_roc) AS avg_LOF_PCA
-- SELECT extra_info->>'n_clusters'
from
(
SELECT * FROM experiments
where method_name = 'LOF_PCA'
and tag = 'online'
and vus_roc IS NOT NULL
and CAST(extra_info->>'n_clusters' AS INTEGER) <= 10
and CAST(extra_info->>'state_size' AS INTEGER) = 1000
and CAST(extra_info->>'n_dimensions' AS INTEGER) = 3
-- ORDER BY vus_roc DESC
-- LIMIT 10
-- LIMIT (SELECT COUNT(*)/2 FROM experiments WHERE method_name = 'IForest_PCA' and tag = 'online' and vus_roc IS NOT NULL)
)
;
