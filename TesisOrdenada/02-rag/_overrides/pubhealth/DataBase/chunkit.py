import os

def create_chunks(input_dir, output_dir, chunk_size=300, overlap=50):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterar sobre cada archivo en el directorio de entrada
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # Verificar si es un archivo (no un directorio)
        if not os.path.isfile(file_path):
            continue
            
        try:
            # Leer el contenido del archivo
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Tokenización simple por espacios
            tokens = text.split()
            
            # Crear los chunks
            chunks = []
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk = tokens[i:i + chunk_size]
                if len(chunk) < 20:  # Ignorar chunks muy pequeños
                    continue
                chunks.append(chunk)
            
            # Guardar los chunks en archivos separados
            for idx, chunk in enumerate(chunks):
                chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{idx}.txt"
                chunk_path = os.path.join(output_dir, chunk_filename)
                with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                    chunk_file.write(' '.join(chunk))
                    
            print(f"Procesado {filename}: {len(chunks)} chunks generados")
            
        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")

# Directorios de entrada y salida
input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')   # antes: ruta absoluta de la maquina original
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chunks')  # antes: ruta absoluta de la maquina original

# Ejecutar la función para crear los chunks
create_chunks(input_directory, output_directory)