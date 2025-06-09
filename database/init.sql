DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_tables WHERE tablename = 'experiments') THEN
        CREATE TABLE experiments (
            experiment_id UUID PRIMARY KEY,
            normality_levels SMALLINT NOT NULL,
            files JSONB NOT NULL,
            series_length INTEGER NOT NULL,
            nof_anomalies INTEGER NOT NULL,
            method_name VARCHAR(255) NOT NULL,
            tag VARCHAR(255) NOT NULL,
            execution_time NUMERIC NOT NULL,

            -- evaluation metrics from TSB
            -- https://github.com/TheDatumOrg/TSB-UAD/blob/313f0fdeba14292b9db4e1aa94c74a983a25de31/TSB_UAD/vus/metrics.py#L21
            AUC_ROC NUMERIC,
            AUC_PR NUMERIC,
            Precision NUMERIC,
            Recall NUMERIC,
            F NUMERIC,
            Precision_at_k NUMERIC,
            Rprecision NUMERIC,
            Rrecall NUMERIC,
            RF NUMERIC,
            R_AUC_ROC NUMERIC,
            R_AUC_PR NUMERIC,
            VUS_ROC NUMERIC,
            VUS_PR NUMERIC,
            Affiliation_Precision NUMERIC,
            Affiliation_Recall NUMERIC,

            extra_info JSONB,
            evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    END IF;

    CREATE INDEX IF NOT EXISTS idx_experiment_id ON experiments (experiment_id);
END $$;