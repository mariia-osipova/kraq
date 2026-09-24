import os
import nltk
from nltk.tokenize import word_tokenize

def analyze_documents():
    # Download required NLTK data
    print("Inicializando NLTK...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Descargando recursos de NLTK...")
        nltk.download('punkt')
    
    # Path to the documents directory
    docs_dir = 'primeras_1000_preguntas'
    print(f"\nBuscando documentos en: {docs_dir}")
    
    # Count txt files
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
    document_count = len(txt_files)
    
    print(f"\nResultados del análisis:")
    print(f"------------------------")
    print(f"Número total de documentos .txt: {document_count}")
    
    # Analyze first 10 txt files
    files_to_analyze = min(10, document_count)
    if files_to_analyze > 0:
        print(f"\nAnalizando los primeros {files_to_analyze} archivos:")
        total_tokens = 0
        token_counts = []
        
        for i in range(files_to_analyze):
            current_file = txt_files[i]
            print(f"\n{i+1}. Analizando archivo: {current_file}")
            
            try:
                with open(os.path.join(docs_dir, current_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    tokens = word_tokenize(content)
                    token_count = len(tokens)
                    token_counts.append(token_count)
                    total_tokens += token_count
                    
                    print(f"   - Tokens encontrados: {token_count}")
                    print(f"   - Primeros 5 tokens: {tokens[:5]}")
                    
            except Exception as e:
                print(f"Error al leer el archivo {current_file}: {str(e)}")
                continue
        
        # Calculate and display average
        average_tokens = total_tokens / files_to_analyze
        print(f"\nResumen del análisis:")
        print(f"------------------------")
        print(f"Total de tokens en {files_to_analyze} archivos: {total_tokens}")
        print(f"Promedio de tokens por archivo: {average_tokens:.2f}")
        print(f"Token count por archivo: {token_counts}")
        
    else:
        print("\nNo se encontraron archivos .txt para analizar")

if __name__ == "__main__":
    print("Iniciando análisis de documentos...")
    analyze_documents()
    print("\nAnálisis completado.")