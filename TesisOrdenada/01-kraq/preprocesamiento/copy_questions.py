import os
import shutil
import sys

def copy_first_x_questions(num_questions):
    # Create output directory
    output_dir = f'primeras_{num_questions}_preguntas'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove if exists
    os.makedirs(output_dir)
    
    # Get all question numbers from existing files
    questions = set()
    for filename in os.listdir('documentos'):
        if filename.endswith('.txt'):
            question_num = int(filename.split('_')[0])
            questions.add(question_num)
    
    # Sort questions to get first X
    sorted_questions = sorted(list(questions))
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
    
    print(f"Resumen:")
    print(f"- Se seleccionaron las primeras {num_questions} preguntas: {selected_questions}")
    print(f"- Se copiaron {files_copied} documentos")
    print(f"- Los archivos están en la carpeta: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <numero_de_preguntas>")
        print("Ejemplo: python script.py 5")
        sys.exit(1)
    
    try:
        num_questions = int(sys.argv[1])
        if num_questions <= 0:
            raise ValueError("El número debe ser positivo")
        copy_first_x_questions(num_questions)
    except ValueError as e:
        print(f"Error: Por favor ingrese un número válido y positivo")
        sys.exit(1)