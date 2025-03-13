import random
import torch
import numpy as np

class RealFakeQuadrupletMiner:
    def __init__(self):
        pass

    def __call__(self, embeddings, labels, real_fake_labels):
        """
        Custom quadruplet miner that ensures quadruplets follow a Real1-Real1-Real2-Fake1 structure.
        """
        batch_size = embeddings.shape[0]
        ids_anchor = []
        ids_pos = []
        ids_neg_fake = []
        ids_neg_real = []
        for i in range(batch_size):
            if real_fake_labels[i] == 0: # real
                # Find another real from the same person
                pos_indices = [j for j in range(batch_size) if labels[j] == labels[i] and real_fake_labels[j] == 0 and j != i]
                # Find another real from a different person
                neg_real_indices = [j for j in range(batch_size) if labels[j] != labels[i] and real_fake_labels[j] == 0]
                # Find a fake from the same person
                neg_fake_indices = [j for j in range(batch_size) if labels[j] == labels[i] and real_fake_labels[j] == 1]
                if not (pos_indices and neg_fake_indices and neg_real_indices):
                    continue
                ids_anchor.append(i)
                ids_pos.append(random.choice(pos_indices))
                ids_neg_real.append(random.choice(neg_real_indices))
                ids_neg_fake.append(random.choice(neg_fake_indices))
        
        embeddings_anchor = embeddings[ids_anchor]
        embeddings_pos = embeddings[ids_pos]
        embeddings_neg_real = embeddings[ids_neg_real]
        embeddings_neg_fake = embeddings[ids_neg_fake]
        return embeddings_anchor, embeddings_pos, embeddings_neg_real, embeddings_neg_fake

class OnlyRealMiner:
    def __init__(self):
        pass

    def __call__(self, embeddings, labels, real_fake_labels):
        real_embeddings = embeddings[real_fake_labels == 0]
        real_labels = labels[real_fake_labels == 0]
        labels_range = np.arange(real_labels.shape[0])
        index2label = {i: label for i, label in enumerate(real_labels)}
        x, y = np.meshgrid(labels_range, labels_range)

        # Stack the pairs and filter out the pairs where elements are the same
        pairs = np.column_stack((x.ravel(), y.ravel()))
        indices = pairs[pairs[:, 0] != pairs[:, 1]]
        labels_anchor = [index2label[i] for i in indices[:, 0]]
        labels_real = [index2label[i] for i in indices[:, 1]]
        labels_identical = [1 if labels_anchor[i] == labels_real[i] else 0 for i in range(len(labels_anchor))]
        embeddings_anchor = real_embeddings[indices[:, 0], :]
        embeddings_pos_neg = real_embeddings[indices[:, 1], :]
        embeddings_pos = embeddings_pos_neg[labels_identical == 1, :].reshape(-1, embeddings_pos_neg.shape[1])
        embeddings_neg = embeddings_pos_neg[labels_identical == 0, :].reshape(-1, embeddings_pos_neg.shape[1])
        embeddings_anchor_pos = embeddings_anchor[labels_identical == 1, :].reshape(-1, embeddings_anchor.shape[1])
        embeddings_anchor_neg = embeddings_anchor[labels_identical == 0, :].reshape(-1, embeddings_anchor.shape[1])
        return embeddings_anchor_pos, embeddings_pos, embeddings_anchor_neg, embeddings_neg
