import random
import torch

class RealFakeTripletMiner:
    def __call__(self, embeddings, labels, real_fake_labels):
        """
        Custom triplet miner that ensures triplets follow a Real-Real-Fake or Real-Fake-Real structure.
        It prevents Fake-Fake-Fake triplets.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.squeeze().tolist()
        if isinstance(real_fake_labels, torch.Tensor):
            real_fake_labels = real_fake_labels.squeeze().tolist()
        triplets = []
        batch_size = embeddings.shape[0]

        for i in range(batch_size):
            if real_fake_labels[i] == "real":
                # Find another real from the same person
                pos_indices = [j for j in range(batch_size) if labels[j] == labels[i] and real_fake_labels[j] == "real" and j != i]
                # Find a fake from the same person
                neg_indices = [j for j in range(batch_size) if labels[j] == labels[i] and real_fake_labels[j] == "fake"]

                if pos_indices and neg_indices:
                    triplets.append((i, random.choice(pos_indices), random.choice(neg_indices)))

        return triplets

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
                label_i = labels[i].item()
                if not (pos_indices and neg_fake_indices and neg_real_indices):
                    # print("Could not find quadruplet for label", label_i)
                    # if not neg_fake_indices:
                    #     print("No fake found for label", label_i)
                    # if not neg_real_indices:
                    #     print("No negative real found for label", label_i)
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