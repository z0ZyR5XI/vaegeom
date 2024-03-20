from .random import RandomCentroids

CENTROIDS = {
    'RandomCentroids': RandomCentroids
}

def build_centroid(name: int, num_centroids: int, *args, **kwargs):
    return CENTROIDS[name](num_centroids=num_centroids, **kwargs)