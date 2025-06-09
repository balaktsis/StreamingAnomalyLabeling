from GLaSSDetector import GLaSSDetector

import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding


# Generalized Latent Space Streaming Detector (GLaSSDetector)
class TabPFNGLaSSDetector(GLaSSDetector):
    def __init__(self,
                 batch_frac=0.1,
                 window_length=None,
                 overlap=1,
                 n_clusters=10,
                 state_size=3000,
                 random_state=42,
                 model=None,
                 tabpfn_device='cuda',
                 ):
        super().__init__(
            batch_frac=batch_frac,
            window_length=window_length,
            overlap=overlap,
            n_clusters=n_clusters,
            random_state=random_state,
            model=model,
            state_size=state_size,
            max_batch_size=self.max_batch_size,
            max_sliding_window_length=self.max_sliding_window_length,
        )
        self.tabpfn_reg = TabPFNRegressor(n_estimators=1, random_state=self.random_state, device=tabpfn_device)


    @property
    def max_batch_size(self):
        return 5000

    @property
    def max_sliding_window_length(self):
        return 250 # 250 * 2 = 500, which is the operational limit of TabPFN


    def get_latent_representation(self, data, debug=False):
        X_train = np.array(data)
        y_train = X_train[:, -1]  # assuming last column is the target variable
        X_train = X_train[:, :-1]
        embedding_extractor = TabPFNEmbedding(
            tabpfn_reg=self.tabpfn_reg,  # we only need a regressor since time series are numerical
            n_fold=0
        )

        if debug:
            # Add a debug mode to make sure that the rest rationale is correct
            print(f"Debug mode is set to True. Embeddings are randomly generated.")
            np.random.seed(self.random_state)
            return np.random.randn(len(X_train), 128)

        embeddings = embedding_extractor.get_embeddings(
            X_train=X_train,
            y_train=y_train,
            X=X_train,
            data_source="train",
        )[0]
        return self.z_normalize(embeddings)
