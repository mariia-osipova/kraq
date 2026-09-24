import json
import os
from collections import defaultdict

# Create documentos directory if it doesn't exist
if not os.path.exists('documentos'):
    os.makedirs('documentos')

# Read the JSON file
with open('processed_documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group documents by related questions
question_docs = defaultdict(list)
for doc_id, doc_info in data.items():
    for question in doc_info['related_questions']:
        question_docs[question].append({
            'title': doc_info['title'],
            'abstract': doc_info['abstract']
        })

# Create text files for each document
for question_num, documents in question_docs.items():
    for idx, doc in enumerate(documents):
        filename = f"{question_num}_{idx:02d}.txt"
        filepath = os.path.join('documentos', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc['title'])
            f.write('\n\n')
            f.write(doc['abstract'])