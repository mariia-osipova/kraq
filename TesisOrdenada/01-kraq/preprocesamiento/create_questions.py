import os
import json
import sys
import shutil

def create_questions_json(num_questions, output_file='questions.json'):
    # Get all question numbers from existing files in documentos directory
    questions_info = []
    for filename in os.listdir('documentos'):
        if filename.endswith('.txt'):
            question_num = int(filename.split('_')[0])
            questions_info.append((question_num, filename))
    
    # Get unique question numbers and sort them
    sorted_questions = sorted(list(set([q[0] for q in questions_info])))
    
    # Ensure we're only selecting questions that have already been copied to primeras_X_preguntas
    output_dir = f'primeras_{num_questions}_preguntas'
    
    # If the directory doesn't exist, create it using copy_questions logic
    if not os.path.exists(output_dir):
        print(f"La carpeta {output_dir} no existe. Creándola primero...")
        
        # This is similar to the logic in copy_questions.py
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove if exists
        os.makedirs(output_dir)
        
        # Sort questions to get first X
        selected_questions = sorted_questions[:num_questions]
        
        # Copy files for selected questions
        files_copied = 0
        for filename in os.listdir('documentos'):
            if filename.endswith('.txt'):
                question_num = int(filename.split('_')[0])
                if question_num in selected_questions:
                    src_path = os.path.join('documentos', filename)
                    dst_path = os.path.join(output_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    files_copied += 1
        
        print(f"- Se seleccionaron las primeras {num_questions} preguntas")
        print(f"- Se copiaron {files_copied} documentos")
    else:
        # If the directory exists, get question numbers from there
        print(f"La carpeta {output_dir} ya existe. Obteniendo preguntas de allí...")
        selected_questions_set = set()
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt'):
                question_num = int(filename.split('_')[0])
                selected_questions_set.add(question_num)
        
        selected_questions = sorted(list(selected_questions_set))
        print(f"- Se encontraron {len(selected_questions)} preguntas únicas en la carpeta")
    
    # Verify that each selected question has all its supporting evidence files
    print("Verificando que todas las preguntas tengan su evidencia correspondiente...")
    
    # Create mapping of question numbers to their document files in 'documentos'
    question_docs_map = {}
    for qnum, filename in questions_info:
        if qnum not in question_docs_map:
            question_docs_map[qnum] = []
        question_docs_map[qnum].append(filename)
    
    # Verify each selected question
    for question_num in selected_questions:
        if question_num in question_docs_map:
            orig_docs = set(question_docs_map[question_num])
            copied_docs = set()
            
            # Check files in primeras_X_preguntas for this question
            for filename in os.listdir(output_dir):
                if filename.endswith('.txt') and int(filename.split('_')[0]) == question_num:
                    copied_docs.add(filename)
            
            # Compare sets
            if orig_docs != copied_docs:
                print(f"ADVERTENCIA: La pregunta {question_num} no tiene todos sus documentos copiados.")
                print(f"  - Originales: {len(orig_docs)}, Copiados: {len(copied_docs)}")
                missing = orig_docs - copied_docs
                if missing:
                    print(f"  - Documentos faltantes: {missing}")
    
    # Load the processed questions file to get question text and answers
    try:
        with open('processed_questions.json', 'r', encoding='utf-8') as f:
            processed_questions = json.load(f)
    except FileNotFoundError:
        print("Error: No se pudo encontrar el archivo processed_questions.json")
        sys.exit(1)
    
    # Create a mapping from question_index to question data
    question_map = {q['index']: q for q in processed_questions}
    
    questions_data = []
    for question_num in selected_questions:
        if question_num in question_map:
            q_data = question_map[question_num]
            
            # Extract question from body
            question = q_data['body']
            
            # Get ideal answer (join if it's a list)
            if isinstance(q_data['ideal_answer'], list):
                ideal_answer = ' '.join(q_data['ideal_answer'])
            else:
                ideal_answer = q_data['ideal_answer']
            
            # Get exact answer as is
            exact_answer = q_data['exact_answer']
            
            # Get question type
            question_type = q_data['type']
            
            questions_data.append({
                "id": f"jp_{question_num}",
                "question": question,
                "ideal_answer": ideal_answer,
                "exact_answer": exact_answer,
                "type": question_type
            })
        else:
            # If question not found in processed_questions.json
            print(f"Advertencia: No se encontró la pregunta {question_num} en processed_questions.json")
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResumen:")
    print(f"- Se creó archivo JSON con {len(questions_data)} preguntas")
    print(f"- Las preguntas tienen IDs: {['jp_'+str(q) for q in selected_questions[:5]]}{'...' if len(selected_questions) > 5 else ''}")
    print(f"- Archivo generado: {output_file}")
    print(f"- Los documentos de evidencia están en: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python create_questions.py <numero_de_preguntas>")
        print("Ejemplo: python create_questions.py 5")
        sys.exit(1)
    
    try:
        num_questions = int(sys.argv[1])
        if num_questions <= 0:
            raise ValueError("El número debe ser positivo")
        create_questions_json(num_questions)
    except ValueError as e:
        print(f"Error: Por favor ingrese un número válido y positivo")
        sys.exit(1)
