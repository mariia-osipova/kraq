import sys
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Añadir el directorio raíz al sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuración para Qdrant
QDRANT_DB_PATH = os.getenv("QDRANT_DB_PATH", "./qdrant_client")
VECTOR_DIMENSION = 768  # Dimensión para nomic-embed-text
DISTANCE_METRIC = "Cosine"

# Nombres de las colecciones que podemos manejar
COLLECTION_NAMES = {
    "finetuned": "questions-index-finetuned",
    "base": "questions-index-base",
    "random": "questions-index-random",
    "chunks": "chunks",
    "benchmark": "questions-benchmark",
    "default": "questions-index"
}

def get_qdrant_client():
    """Obtiene una instancia del cliente Qdrant configurado para almacenamiento on-disk."""
    db_path = project_root / "qdrant_client"
    print(f"Initializing Qdrant client with on-disk storage at: {os.path.abspath(db_path)}")
    return QdrantClient(path=db_path)

def get_existing_collections():
    """Obtiene la lista de colecciones existentes en Qdrant."""
    client = get_qdrant_client()
    collections_info = client.get_collections()
    return [col.name for col in collections_info.collections]

def clear_collection(collection_name):
    """Elimina y recrea una colección específica."""
    client = get_qdrant_client()
    
    try:
        print(f"Eliminando colección '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        print(f"Colección '{collection_name}' eliminada correctamente.")
    except Exception as e:
        print(f"Error al eliminar colección '{collection_name}': {e}")
        return False
    
    try:
        print(f"Recreando colección '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
        )
        print(f"Colección '{collection_name}' recreada exitosamente.")
        return True
    except Exception as e:
        print(f"Error al recrear colección '{collection_name}': {e}")
        return False

def interactive_clear():
    """Interfaz interactiva para elegir qué colección limpiar."""
    print("\n=== Limpieza de Colecciones Qdrant ===")
    
    # Obtener colecciones existentes
    existing_collections = get_existing_collections()
    
    # Filtrar nuestras colecciones conocidas que existen
    our_collections = {}
    for key, name in COLLECTION_NAMES.items():
        if name in existing_collections:
            our_collections[key] = name
    
    if not our_collections:
        print("No se encontraron colecciones en Qdrant para limpiar.")
        other_collections = [c for c in existing_collections if c not in COLLECTION_NAMES.values()]
        if other_collections:
            print(f"Existen otras colecciones no relacionadas: {', '.join(other_collections)}")
        return
    
    # Mostrar opciones
    print("\nColecciones disponibles:")
    options = []
    
    for i, (key, name) in enumerate(our_collections.items(), 1):
        print(f"{i}. {name} (tipo: {key})")
        options.append(name)
    
    print(f"{len(options) + 1}. TODAS las colecciones")
    print("0. Cancelar operación")
    
    # Solicitar elección
    choice = None
    while choice is None:
        try:
            choice_input = input("\nIngresa el número de la colección a limpiar (0 para cancelar): ")
            choice = int(choice_input)
            
            if choice == 0:
                print("Operación cancelada.")
                return
            elif choice > 0 and choice <= len(options):
                selected = options[choice - 1]
                confirm = input(f"\n¿Estás seguro de que quieres limpiar la colección '{selected}'? (s/n): ")
                if confirm.lower() == 's':
                    print(f"\nLimpiando colección: {selected}")
                    if clear_collection(selected):
                        print(f"\nLa colección '{selected}' ha sido limpiada exitosamente.")
                else:
                    print("Operación cancelada.")
            elif choice == len(options) + 1:
                confirm = input("\n¿Estás seguro de que quieres limpiar TODAS las colecciones? (s/n): ")
                if confirm.lower() == 's':
                    success_count = 0
                    for col_name in options:
                        print(f"\nLimpiando colección: {col_name}")
                        if clear_collection(col_name):
                            success_count += 1
                    print(f"\nSe limpiaron exitosamente {success_count} de {len(options)} colecciones.")
                else:
                    print("Operación cancelada.")
            else:
                print("Opción inválida. Por favor ingresa un número válido.")
                choice = None
        except ValueError:
            print("Entrada inválida. Por favor ingresa un número.")
            choice = None

def clear_qdrant_collection_cli():
    """Función principal para CLI."""
    # Si se pasa un argumento, usarlo como nombre de colección
    if len(sys.argv) > 1 and sys.argv[1] in COLLECTION_NAMES:
        collection_name = COLLECTION_NAMES[sys.argv[1]]
        print(f"Limpiando colección específica: {collection_name}")
        clear_collection(collection_name)
    # Si se pasa "all" como argumento, limpiar todas
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        print("Limpiando todas las colecciones...")
        existing = get_existing_collections()
        for name in COLLECTION_NAMES.values():
            if name in existing:
                clear_collection(name)
    # De lo contrario, usar la interfaz interactiva
    else:
        interactive_clear()

if __name__ == "__main__":
    clear_qdrant_collection_cli()
