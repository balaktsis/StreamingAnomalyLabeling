from GLaSSDetector import GLaSSDetector

import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding


# Generalized Latent Space Streaming Detector (GLaSSDetector)
class PCAGLaSSDetector(GLaSSDetector):
    def __init__(self,
                 batch_frac=0.1,
                 window_length=None,
                 overlap=1,
                 n_clusters=10,
                 state_size=3000,
                 random_state=42,
                 model=None,
                 n_dimensions=2, ):
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
        self.n_dimensions = n_dimensions


    def get_latent_representation(self, data, debug=False):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        lower_dim = PCA(n_components=2)
        scaler = StandardScaler()

        # Fit and transform
        return scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(data)),
        )
