from einops import einsum
import numpy as np

embedding_seed = 42
if embedding_seed is None:
    embeddings = np.load("embeddings.npy")
    interactions = np.load("interactions.npy")
else:
    embeddings = np.load(f"embeddings/{embedding_seed}/embeddings.npy")
    interactions = np.load(f"embeddings/{embedding_seed}/interactions.npy")
pickup = (interactions & 1) > 0
drop = (interactions & 2) > 0

#from probe_network import MLPProbe
#probe = MLPProbe(out_dim=2)

pickups = embeddings[pickup]
not_pickups = embeddings[np.logical_not(pickup)]
drops = embeddings[drop]
not_drops = embeddings[np.logical_not(drop)]

def linear_classifier(positives, negatives, direction=None):
    if direction is None:
        direction = positives.mean(axis=0) - negatives.mean(axis=0)
        direction /= np.linalg.norm(direction)

    positives = einsum(positives, direction, 'n x, x -> n')
    negatives = einsum(negatives, direction, 'n x, x -> n')
    true_positive = np.sum(positives > 0)
    true_negative = np.sum(negatives < 0)
    print(f"True positives: {true_positive}/{len(positives)} ({true_positive / len(positives):.5f})")
    print(f"True negatives: {true_negative}/{len(negatives)} ({true_negative / len(negatives):.5f})")
    return direction

load_seed = 43
if load_seed is None:
    pickup_vec = None
    drop_vec = None
else:
    pickup_vec = np.load(f"embeddings/{load_seed}/pickup_vec.npy")
    drop_vec = np.load(f"embeddings/{load_seed}/drop_vec.npy")

print(f"Embedding seed={embedding_seed}, Load seed={load_seed}")
print("Pickup available classifier:")
pickup_vec = linear_classifier(pickups, not_pickups, direction=pickup_vec)
print("Drop available classifier:")
drop_vec = linear_classifier(drops, not_drops, direction=drop_vec)

if load_seed is None:
    np.save(f"embeddings/{embedding_seed}/pickup_vec.npy", pickup_vec)
    np.save(f"embeddings/{embedding_seed}/drop_vec.npy", drop_vec)
