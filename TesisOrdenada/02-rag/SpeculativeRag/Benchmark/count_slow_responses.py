import json
import os

# Ruta absoluta al archivo full_results.json
results_path = "SpeculativeRag/Benchmark/benchmark_results_qdrant/ideal_answers/full_results.json"

# Cargar los datos JSON
with open(results_path, "r") as f:
    results = json.load(f)

# Contar entradas con total_time > 8 segundos
slow_entries = [entry for entry in results if entry.get("total_time", 0) > 8]
count = len(slow_entries)

# Contar respuestas incorrectas entre las lentas
incorrect_slow = [entry for entry in slow_entries if entry.get("is_correct", 1) == 0]
incorrect_count = len(incorrect_slow)

# Imprimir resultados
print(f"Total de entradas: {len(results)}")
print(f"Entradas que tomaron más de 8 segundos: {count}")
print(f"Porcentaje: {count/len(results)*100:.2f}%")
print(f"Respuestas lentas e incorrectas (is_correct=0): {incorrect_count}")
print(f"Porcentaje de respuestas lentas que son incorrectas: {incorrect_count/count*100:.2f}% si hay entradas lentas")

# Opcional: Listar las entradas lentas con sus IDs, tiempos y corrección
print("\nEntradas lentas:")
for entry in slow_entries:
    correct_status = "Correcta" if entry.get("is_correct", 1) == 1 else "Incorrecta"
    print(f"ID: {entry['question_id']}, Tiempo: {entry['total_time']:.2f}s, {correct_status}, Pregunta: {entry['question'][:50]}...")

# Opcional: Listar solo las respuestas lentas e incorrectas
if incorrect_count > 0:
    print("\nEntradas lentas e incorrectas:")
    for entry in incorrect_slow:
        print(f"ID: {entry['question_id']}, Tiempo: {entry['total_time']:.2f}s, Pregunta: {entry['question'][:50]}...") 