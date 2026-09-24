"""
Utilidades para serialización JSON, incluyendo soporte para tipos NumPy
"""
import json
import numpy as np
from datetime import datetime

def convert_numpy_types(obj):
    """
    Convierte tipos de NumPy a tipos nativos de Python para serialización JSON.
    
    Args:
        obj: El objeto a convertir
        
    Returns:
        Objeto convertido a tipos nativos de Python
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder personalizado que maneja tipos de NumPy y datetime
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyJSONEncoder, self).default(obj)

def dump_json(obj, file_path, indent=2, ensure_ascii=False):
    """
    Función de utilidad para guardar datos en formato JSON, compatible con NumPy
    
    Args:
        obj: Objeto a serializar
        file_path: Ruta del archivo para guardar
        indent: Indentación para el archivo JSON
        ensure_ascii: Si se garantiza que el resultado contiene solo caracteres ASCII
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(obj), f, indent=indent, 
                  ensure_ascii=ensure_ascii, cls=NumpyJSONEncoder) 