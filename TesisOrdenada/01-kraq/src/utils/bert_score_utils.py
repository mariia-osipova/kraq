import torch
from bert_score import score
import numpy as np
from typing import List, Dict, Tuple

def calculate_bert_score(candidates: List[str], references: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Forzar el uso de GPU1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    P, R, F1 = score(candidates, references, lang="es", verbose=False, device=device)
    return P.numpy(), R.numpy(), F1.numpy()

def get_bert_score_statistics(f1_scores: np.ndarray) -> Dict:
    return {
        "media": float(np.mean(f1_scores)),
        "mediana": float(np.median(f1_scores)),
        "desviacion_estandar": float(np.std(f1_scores)),
        "maximo": float(np.max(f1_scores)),
        "minimo": float(np.min(f1_scores)),
        "percentil_25": float(np.percentile(f1_scores, 25)),
        "percentil_75": float(np.percentile(f1_scores, 75))
    }

def get_top_bottom_scores(preguntas_originales: List[str], 
                         preguntas_similares: List[str], 
                         scores: np.ndarray,
                         top_k: int = 10) -> Dict:
    indices_ordenados = np.argsort(scores)
    
    peores = indices_ordenados[:top_k]
    mejores = indices_ordenados[-top_k:][::-1]
    
    return {
        "mejores_coincidencias": [
            {
                "pregunta_original": preguntas_originales[idx],
                "pregunta_similar": preguntas_similares[idx],
                "bert_score": float(scores[idx])
            } for idx in mejores
        ],
        "peores_coincidencias": [
            {
                "pregunta_original": preguntas_originales[idx],
                "pregunta_similar": preguntas_similares[idx],
                "bert_score": float(scores[idx])
            } for idx in peores
        ]
    } 