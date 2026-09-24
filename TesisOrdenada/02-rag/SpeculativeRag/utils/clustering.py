"""
Clustering utilities for multi-perspective sampling
"""
from collections import defaultdict
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from .inbedder import InBedder


def multi_perspective_sampling(embeddings, k=3, seed=42):
    """
    Realizar multi-perspective sampling en los embeddings
    
    Args:
        embeddings: Array de embeddings de documentos
        k: Número de clusters a crear
        seed: Semilla para reproducibilidad
        
    Returns:
        Diccionario con clusters y información adicional
    """
    np.random.seed(seed)
    
    # Asegurarse de que tenemos suficientes documentos
    n_docs = len(embeddings)
    if n_docs < k:
        # Si hay menos documentos que clusters, ajustar k
        k = max(1, n_docs)
    
    # Clustering con K-means
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Organizar documentos por cluster
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    
    # Eliminar clusters vacíos
    clusters = [c for c in clusters if len(c) > 0]
    
    return {
        "clusters": clusters,
        "labels": cluster_labels.tolist(),
        "k": len(clusters)
    }

