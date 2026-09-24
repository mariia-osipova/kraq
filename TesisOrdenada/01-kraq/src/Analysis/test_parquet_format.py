import pandas as pd
import json
import numpy as np
import os # Importar os para manejo de rutas
from typing import List, Dict, Any

def count_findings_in_row(findings) -> int:
    """
    Cuenta cuántos hallazgos hay en un registro específico,
    manejando los diferentes tipos de datos posibles.
    
    Returns:
        int: Número de hallazgos en este registro
    """
    # Si es None o todos son NaN
    if findings is None:
        return 0
    
    # Para arrays NumPy, verificar si es NaN usando np.isnan
    if isinstance(findings, np.ndarray):
        # Si es un array de objetos, simplemente contar elementos no nulos
        if findings.dtype == np.dtype('O'):
            return len(findings)
        # Si es un array numérico, verificar NaN
        elif np.issubdtype(findings.dtype, np.number):
            return len(findings) - np.isnan(findings).sum()
        else:
            return len(findings)
    
    # Para valores escalares, usar pd.isna
    if pd.isna(findings):
        return 0
        
    if isinstance(findings, str):
        try:
            # Intentar decodificar si es una cadena JSON
            findings_list = json.loads(findings)
            return len(findings_list) if isinstance(findings_list, list) else 0
        except json.JSONDecodeError:
            return 1  # Si es una string simple, contar como 1 hallazgo
    elif isinstance(findings, list):
        return len(findings)
    else:
        try:
            return len(list(findings))  # Intentar convertir a lista si es iterable
        except (TypeError, ValueError):
            return 0  # Si no se puede contar, asumir 0

def examine_parquet():
    print("\n=== Examinando formato del archivo parquet ===")
    
    # Construir la ruta relativa al archivo Parquet
    # __file__ es la ruta de este script (src/Analysis/test_parquet_format.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir dos niveles para llegar a la raíz del proyecto
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    # Nombre del archivo esperado (ajustar si es necesario)
    parquet_filename = "community_reports.parquet"
    db_path = os.path.join(project_root, 'output', parquet_filename)

    print(f"Intentando cargar el archivo Parquet desde: {db_path}")

    # Verificar si el archivo existe
    if not os.path.exists(db_path):
        print(f"\nError: El archivo Parquet no se encontró en la ruta esperada.")
        print("Asegúrate de que el archivo exista en el directorio 'output/'.")
        return # Salir si el archivo no existe

    try:
        # Cargar el archivo
        df = pd.read_parquet(db_path)
    except Exception as e:
        print(f"\nError al cargar el archivo Parquet: {e}")
        return # Salir si hay error al cargar

    # Mostrar los primeros 3 registros con formato detallado
    print("\nPrimeros 3 registros:")
    for idx, row in df.head(3).iterrows():
        print("\n" + "="*80)
        print(f"ID: {idx} (Tipo: {type(idx)})") # Mostrar tipo de índice
        print(f"Level: {row['level']} (Tipo: {type(row['level'])})")
        
        print("\n=== RESUMEN GENERAL ===")
        print(f"{row['summary']} (Tipo: {type(row['summary'])})")
        
        print("\n=== FINDINGS ===")
        findings = row['findings']
        print(f"Tipo de 'findings': {type(findings)}")

        # Ajuste para manejar diferentes posibles tipos de 'findings'
        if isinstance(findings, str):
             try:
                 # Intentar decodificar si es una cadena JSON
                 findings = json.loads(findings)
                 print("(Interpretado como JSON string)")
             except json.JSONDecodeError:
                 print("(Interpretado como string simple)")
                 findings = [{'summary': findings, 'explanation': 'N/A'}] # Tratar como texto simple si no es JSON
        elif isinstance(findings, np.ndarray):
            findings = findings.tolist()
            print("(Interpretado como NumPy array)")
        elif not isinstance(findings, list): # Si no es lista, intentar convertir
             print(f"(Tipo inesperado {type(findings)}, intentando iterar...)")
             try:
                 findings = list(findings) # Intentar convertir a lista
             except TypeError:
                 findings = [] # Fallback a lista vacía

        if isinstance(findings, list):
            for i, finding in enumerate(findings, 1):
                print(f"\nFinding {i}:")
                if isinstance(finding, dict):
                    print(f"Resumen del finding: {finding.get('summary', 'N/A')}")
                    print(f"Explicación: {finding.get('explanation', 'N/A')}")
                else:
                     print(f"Contenido del finding (no es dict): {finding}") # Mostrar si no es un diccionario
                print("-" * 40)
        else:
             print("No se pudieron procesar los findings.")
        
        print("="*80)
    
    # Mostrar estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS GENERALES")
    print("="*80)
    
    total_registros = len(df)
    print(f"Total de registros: {total_registros}")
    
    if 'level' in df.columns:
        print("\nRegistros por nivel:")
        level_counts = df['level'].value_counts().sort_index()
        print(level_counts)
        
        # Calcular cuántas preguntas main se generarán por nivel
        print("\nPreguntas Main a generar por nivel:")
        for level, count in level_counts.items():
            print(f"  Nivel {level}: {count} preguntas main")
    else:
        print("\nColumna 'level' no encontrada.")

    # Analizar los hallazgos
    if 'findings' in df.columns:
        findings_null = df['findings'].isna().sum()
        findings_not_null = total_registros - findings_null
        print(f"\nRegistros con findings:")
        print(f"  Con findings válidos: {findings_not_null} ({findings_not_null/total_registros*100:.1f}%)")
        print(f"  Con findings nulos/NaN: {findings_null} ({findings_null/total_registros*100:.1f}%)")
        
        # Contar el número total de hallazgos
        total_findings = 0
        findings_counts = []
        
        for _, row in df.iterrows():
            findings_count = count_findings_in_row(row['findings'])
            findings_counts.append(findings_count)
            total_findings += findings_count
        
        # Calcular estadísticas sobre hallazgos
        findings_series = pd.Series(findings_counts)
        avg_findings = findings_series.mean()
        max_findings = findings_series.max()
        min_findings = findings_series.min()
        median_findings = findings_series.median()
        
        print("\nEstadísticas de Hallazgos:")
        print(f"  Total de hallazgos en todos los registros: {total_findings}")
        print(f"  Promedio de hallazgos por registro: {avg_findings:.2f}")
        print(f"  Mediana de hallazgos por registro: {median_findings}")
        print(f"  Máximo de hallazgos en un registro: {max_findings}")
        print(f"  Mínimo de hallazgos en registros no nulos: {min_findings}")
        
        # Desglose de hallazgos por nivel si está disponible
        if 'level' in df.columns:
            print("\nHallazgos por nivel:")
            level_findings = {}
            for _, row in df.iterrows():
                level = row['level']
                findings_count = count_findings_in_row(row['findings'])
                
                if level not in level_findings:
                    level_findings[level] = []
                level_findings[level].append(findings_count)
            
            # Calcular estadísticas por nivel
            for level, counts in sorted(level_findings.items()):
                total_level_findings = sum(counts)
                avg_level_findings = np.mean(counts)
                count_not_null = sum(1 for c in counts if c > 0)
                
                print(f"  Nivel {level}:")
                print(f"    Total hallazgos: {total_level_findings}")
                print(f"    Promedio hallazgos por registro: {avg_level_findings:.2f}")
                print(f"    Registros con al menos 1 hallazgo: {count_not_null} de {len(counts)}")
        
        # Distribución de cantidad de hallazgos
        print("\nDistribución de cantidad de hallazgos por registro:")
        findings_distribution = findings_series.value_counts().sort_index()
        for count, occurrences in findings_distribution.items():
            print(f"  {count} hallazgos: {occurrences} registros ({occurrences/total_registros*100:.1f}%)")
        
        # Estimación de preguntas finding a generar
        print("\nEstimación de preguntas Finding a generar:")
        print(f"  Total preguntas finding (una por hallazgo): {total_findings}")
        if 'level' in df.columns:
            for level, counts in sorted(level_findings.items()):
                total_level_findings = sum(counts)
                print(f"  Nivel {level}: {total_level_findings} preguntas finding")
        
        # Estimación total de preguntas (main + finding)
        total_questions = total_registros + total_findings
        print(f"\nTotal estimado de preguntas a generar (main + finding): {total_questions}")
        if 'level' in df.columns:
            for level, count in level_counts.items():
                level_findings_total = sum(level_findings.get(level, [0]))
                level_total = count + level_findings_total
                print(f"  Nivel {level}: {level_total} preguntas totales ({count} main + {level_findings_total} finding)")
        
    else:
        print("\nColumna 'findings' no encontrada.")

    print("\nTipos de datos de las columnas:")
    print(df.dtypes)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    examine_parquet() 