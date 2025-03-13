from collections import Counter, defaultdict
from typing import Iterator, List, Union
import numpy as np
import random

def smart_sample(array, k):
    """Sample n_samples items from given list. If array contains at least n_samples items, sample without repetition;
    otherwise take all the unique items and sample n_samples - len(array) ones with repetition.

    Args:
        array: list of unique elements to sample from
        k: number of items to sample

    Returns:
        sampled_items: list of sampled items
    """
    array_size = len(array)
    if array_size < k:
        sampled = (
            np.random.choice(array, size=array_size, replace=False).tolist()
            + np.random.choice(array, size=k - array_size, replace=True).tolist()
        )
    else:
        sampled = np.random.choice(array, size=k, replace=False).tolist()
    return sampled

class TrainValSampler:

    def __init__(self, labels: Union[List[int], np.ndarray], n_labels: int, n_instances: int, train: bool, train_ratio: float):
        """
        Args:
            labels: List of the labels for each element in the dataset
            n_labels: The desired number of labels in a batch, should be > 1
            n_instances: The desired number of instances of each label in a batch, should be > 1

        """
        from .utils import fix_seed
        fix_seed(0)
        unq_labels = set(labels)

        assert isinstance(n_labels, int) and isinstance(n_instances, int)
        assert (1 < n_labels <= len(unq_labels))
        assert all(n > 1 for n in Counter(labels).values()), "Each label should contain at least 2 samples"
        unq_labels_l = list(unq_labels)
        n_train_labels = int(len(unq_labels_l) * train_ratio)
        random.shuffle(unq_labels_l)
        if train:
            self._unq_labels = set(unq_labels_l[:n_train_labels])
        else:
            self._unq_labels = set(unq_labels_l[n_train_labels:])
        self.n_labels = n_labels
        self.n_instances = n_instances

        self._batch_size = self.n_labels * self.n_instances

        lbl2idx = defaultdict(list)

        for idx, label in enumerate(labels):
            lbl2idx[label].append(idx)

        self.lbl2idx = dict(lbl2idx)

        self._batches_in_epoch = len(self._unq_labels) // self.n_labels

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._batches_in_epoch

    def __iter__(self) -> Iterator[List[int]]:
        inds_epoch = []

        labels_rest = self._unq_labels.copy()

        for _ in range(len(self)):
            ids_batch = []

            labels_for_batch = set(
                np.random.choice(list(labels_rest), size=min(self.n_labels, len(labels_rest)), replace=False)
            )
            labels_rest -= labels_for_batch

            for cls in labels_for_batch:
                cls_ids = self.lbl2idx[cls]
                if self.n_instances == 1:
                    selected_inds = cls_ids
                else:
                    selected_inds = smart_sample(cls_ids, self.n_instances)
                ids_batch.extend(selected_inds)

            inds_epoch.append(ids_batch)
        return iter(inds_epoch)
