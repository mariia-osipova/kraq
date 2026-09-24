import json

def count_questions():
    with open('src/Benchmarks/qa_benchmark.json', 'r') as file:
        data = json.load(file)
        question_count = len(data)
        print(f"Total number of questions in the file: {question_count}")

if __name__ == "__main__":
    count_questions()