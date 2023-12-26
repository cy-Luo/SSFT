from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


from tqdm import tqdm

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KCenterGreedyBase:

    def __init__(self, embedding: Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: Tensor
        self.min_distances: Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        if isinstance(self.min_distances, Tensor):
            probabilities = F.softmax(self.min_distances, dim=0)
            top_k = 5
            probabilities = probabilities.squeeze()
            top_k_indices = torch.topk(probabilities, top_k).indices
            mask = torch.zeros_like(probabilities)
            mask.scatter_(0, top_k_indices, 1)
            probabilities = probabilities * mask
            idx = int(torch.multinomial(probabilities, num_samples=1).item())

        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        
        for _ in tqdm(range(self.coreset_size)):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:


        idxs = self.select_coreset_idxs(selected_idxs)
      
        coreset = self.embedding[idxs]

        return coreset
