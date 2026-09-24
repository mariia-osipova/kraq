import lancedb
import pandas as pd
import numpy as np
import ast
import os # Importar os para manejo de rutas

# --- Construir la ruta relativa a la base de datos LanceDB ---
# __file__ es la ruta de este script (src/Analysis/read_summaries.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Subir dos niveles para llegar a la raíz del proyecto
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Nombre del directorio de la base de datos LanceDB
lancedb_dirname = "lancedb"
db_path = os.path.join(project_root, 'output', lancedb_dirname)
# --- Fin de la construcción de la ruta ---

print(f"Intentando conectar a LanceDB en: {db_path}")

# Verificar si el directorio existe (LanceDB lo crea si no existe al conectar/crear tabla)
# No es estrictamente necesario verificar antes de connect, pero puede dar un mensaje más claro.
if not os.path.isdir(db_path):
    print(f"Warning: El directorio de LanceDB '{db_path}' no existe aún. LanceDB debería crearlo si es necesario.")
    # Podrías decidir crear el directorio aquí si prefieres: os.makedirs(db_path, exist_ok=True)

try:
    db = lancedb.connect(db_path)

    # Nombre de la tabla esperado
    table_name = "default-community-full_content"
    print(f"Intentando abrir la tabla: {table_name}")

    # Verificar si la tabla existe antes de abrirla
    table_names = db.table_names()
    if table_name not in table_names:
         print(f"\nError: La tabla '{table_name}' no se encontró en la base de datos LanceDB.")
         print(f"Tablas disponibles: {table_names}")
         # Salir o manejar el error como prefieras
         # exit() # Opcional: salir si la tabla no existe
         df_summaries = pd.DataFrame() # Crear DataFrame vacío para evitar errores posteriores
    else:
        # Abrir la tabla de summaries
        community_summaries = db.open_table(table_name)
        df_summaries = community_summaries.to_pandas()
        print(f"Tabla '{table_name}' abierta exitosamente.")

except Exception as e:
    print(f"\nError al conectar o abrir la tabla de LanceDB: {e}")
    # Crear DataFrame vacío para evitar errores en el resto del script si falla la carga
    df_summaries = pd.DataFrame()

def extract_level_from_attributes(attributes):
    try:
        # Manejar NaN o None
        if pd.isna(attributes):
            return None
        if isinstance(attributes, str):
            # Intentar evaluar la cadena; si falla, podría ser texto simple
            try:
                attributes = ast.literal_eval(attributes)
            except (ValueError, SyntaxError):
                 # Podría ser una cadena simple o mal formada, devolver None o manejar diferente
                 return None # Opcional: registrar un warning
        # Asegurarse de que sea una lista de diccionarios después de evaluar
        if isinstance(attributes, list):
            return next((attr.get('level') for attr in attributes if isinstance(attr, dict) and 'level' in attr), None)
        else:
             # Si no es una lista (ej. un diccionario directamente), buscar 'level'
             if isinstance(attributes, dict) and 'level' in attributes:
                 return attributes['level']
             return None # No es el formato esperado
    except Exception as e:
        # Capturar otros posibles errores durante la extracción
        print(f"Error extrayendo nivel de atributos '{attributes}': {e}")
        return None

def print_level_statistics(df):
    print("\n=== Estadísticas por Nivel ===")

    if df.empty:
        print("El DataFrame está vacío. No se pueden calcular estadísticas.")
        return

    # Verificar si la columna 'attributes' existe
    if 'attributes' not in df.columns:
        print("Error: La columna 'attributes' no se encuentra en el DataFrame.")
        return

    # Extraer nivel de los atributos
    df['level'] = df['attributes'].apply(extract_level_from_attributes)

    # Filtrar niveles nulos/NaN antes de procesar
    valid_levels_df = df.dropna(subset=['level'])
    if valid_levels_df.empty:
        print("No se encontraron niveles válidos en la columna 'attributes'.")
        return

    # Intentar convertir a numérico para ordenar, si es posible, si no, tratar como string
    try:
        levels = sorted(valid_levels_df['level'].unique().astype(float))
    except ValueError:
        levels = sorted(valid_levels_df['level'].unique().astype(str))

    print("\nDistribución de resúmenes por nivel:")
    for level in levels:
        # Comparar con el tipo correcto (numérico o string)
        level_df = valid_levels_df[valid_levels_df['level'] == level]
        print(f"Nivel {level}: {len(level_df)} resúmenes")

        # Estadísticas adicionales por nivel
        if 'vector' in level_df.columns and not level_df['vector'].isnull().all():
            try:
                # Filtrar vectores nulos antes de apilar
                valid_vectors = level_df['vector'].dropna()
                if not valid_vectors.empty:
                    vectors = np.stack(valid_vectors.values)
                    print(f"  - Dimensión promedio de vectores: {vectors.shape[1]}")
                else:
                    print("  - No hay vectores válidos en este nivel.")
            except Exception as e:
                print(f"  - Error procesando vectores para nivel {level}: {e}")
        else:
            print("  - Columna 'vector' no encontrada o todos los valores son nulos en este nivel.")


        if 'text' in level_df.columns and not level_df['text'].isnull().all():
            text_lengths = level_df['text'].dropna().str.len()
            if not text_lengths.empty:
                print(f"  - Longitud promedio de texto: {text_lengths.mean():.2f} caracteres")
            else:
                 print("  - No hay textos válidos en este nivel.")
        else:
            print("  - Columna 'text' no encontrada o todos los valores son nulos en este nivel.")

        print()

def print_sample_summaries(df, samples_per_level=1):
    print("\n=== Muestras de Resúmenes por Nivel ===")

    if df.empty:
        print("El DataFrame está vacío. No se pueden mostrar muestras.")
        return

    # Extraer nivel de los atributos si aún no se ha hecho
    if 'level' not in df.columns:
         if 'attributes' not in df.columns:
              print("Error: Columna 'attributes' necesaria para extraer niveles no encontrada.")
              return
         df['level'] = df['attributes'].apply(extract_level_from_attributes)

    valid_levels_df = df.dropna(subset=['level'])
    if valid_levels_df.empty:
        print("No se encontraron niveles válidos para mostrar muestras.")
        return

    try:
        levels = sorted(valid_levels_df['level'].unique().astype(float))
    except ValueError:
        levels = sorted(valid_levels_df['level'].unique().astype(str))

    for level in levels:
        level_df = valid_levels_df[valid_levels_df['level'] == level]
        print(f"\nNivel {level}:")
        print("="*80)

        # Tomar muestras aleatorias del nivel
        if not level_df.empty:
            sample_indices = np.random.choice(level_df.index,
                                            size=min(samples_per_level, len(level_df)),
                                            replace=False)

            for idx in sample_indices:
                row = df.loc[idx]
                print(f"ID: {idx}") # Usar el índice original del DataFrame
                # Mostrar texto, manejando posible NaN/None
                text_content = row.get('text', 'N/A')
                print(f"Text: {str(text_content)[:200]}...")
                # Mostrar atributos, manejando posible NaN/None
                attributes_content = row.get('attributes', 'N/A')
                print(f"Attributes: {attributes_content}")
                print("-"*80)
        else:
             print("No hay datos para este nivel.")


def analyze_summaries():
    # Verificar y mostrar información básica
    print("=== Información General ===")
    if df_summaries.empty:
        print("No se cargaron datos de LanceDB.")
        return

    print(f"Total de resúmenes: {len(df_summaries)}")
    print("\nColumnas disponibles:")
    print(df_summaries.columns.tolist())

    # Mostrar estadísticas por nivel
    print_level_statistics(df_summaries.copy()) # Usar copia para evitar SettingWithCopyWarning

    # Mostrar ejemplos de cada nivel
    print_sample_summaries(df_summaries.copy(), samples_per_level=1) # Usar copia

    # Análisis adicional de la estructura
    print("\n=== Análisis de Estructura ===")
    for column in df_summaries.columns:
        non_null = df_summaries[column].count()
        print(f"{column}: {non_null} valores no nulos "
              f"({(non_null/len(df_summaries))*100:.1f}% de completitud)")

if __name__ == "__main__":
    analyze_summaries() 