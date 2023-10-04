"""create_dataloader function, that returns weighted dataloader or normal dataloader dependent on argument"""

from typing import Any, Optional, Union

import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from stapler.datamodule.components.train_dataset import STAPLERDataset


def create_dataloader(
    dataset: STAPLERDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    weighted_class_sampling: bool = False,
    weighted_epitope_sampling: bool = False,
) -> DataLoader:
    """Create dataloader for training

    Args:
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 4.
        pin_memory (bool, optional): Pin memory. Defaults to True.
        persistent_workers (bool, optional): Persistent workers. Defaults to True.
        weighted_class_sampling (bool, optional): Whether to use weighted class sampling. Defaults to False.
        weighted_epitope_sampling (bool, optional): Whether to use weighted epitope sampling. Defaults to False.

    Returns:
        DataLoader: Dataloader
    """

    weights = [1 for _ in range(len(dataset))]

    if weighted_class_sampling:
        # Calculate the weights for each sample
        labels = dataset.labels * 1
        class_counts = Counter(labels)
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}

        # add the class weights to the weights list
        class_weights_list = [class_weights[label] for label in labels]
        weights = [w1 * w2 for w1, w2 in zip(weights, class_weights_list)]

    if weighted_epitope_sampling:
        # Calculate the weights for each sample
        epitopes = list(dataset.epitope_df)
        epitopes_counts = Counter(epitopes)
        # calulate IDF for each epitope (inverse document frequency; IDF = log2(1 / (frequency / total)))
        epitopes_weights = {epitope: np.log2(1 / (count/len(epitopes))) for epitope, count in epitopes_counts.items()}

        # create a list of weights for each sample
        epitope_weights_list = [epitopes_weights[epitope] for epitope in epitopes]
        weights = [w1 * w2 for w1, w2 in zip(weights, epitope_weights_list)]

    if weighted_class_sampling or weighted_epitope_sampling:
        # Create a WeightedRandomSampler with the calculated weights
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        sampler=sampler,
    )

    return dataloader
