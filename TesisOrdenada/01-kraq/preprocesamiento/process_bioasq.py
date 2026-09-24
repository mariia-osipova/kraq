import json
from Bio import Entrez
from typing import Dict, List, Optional
import time

# Configure email for NCBI
Entrez.email = "sellaicompany@gmail.com"  # Replace with your email

class Question:
    def __init__(self, body: str, ideal_answer: List[str], exact_answer: Optional[List], question_type: str, index: int = None):
        self.body = body
        self.ideal_answer = ideal_answer
        self.exact_answer = exact_answer
        self.type = question_type
        self.index = index

class Document:
    def __init__(self, pubmed_id: str, title: str, abstract: str):
        self.pubmed_id = pubmed_id
        self.title = title
        self.abstract = abstract
        self.related_questions = set()  # Usamos set para evitar duplicados

def extract_pubmed_id(url: str) -> str:
    """Extract PubMed ID from URL."""
    return url.split('/')[-1]

def fetch_pubmed_document(pubmed_id: str) -> Optional[Document]:
    """Fetch document information from PubMed."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Cambiar el modo de retorno a XML
            handle = Entrez.efetch(
                db="pubmed",
                id=pubmed_id,
                rettype="xml",
                retmode="xml"
            )
            
            records = Entrez.read(handle)
            if not records['PubmedArticle']:
                print(f"    No data found for PubMed ID {pubmed_id}")
                return None
                
            article = records['PubmedArticle'][0]
            article_data = article['MedlineCitation']['Article']
            
            # Extraer título y abstract con manejo seguro
            title = article_data.get('ArticleTitle', '')
            
            abstract = ''
            if 'Abstract' in article_data:
                abstract_texts = article_data['Abstract'].get('AbstractText', [])
                if isinstance(abstract_texts, list):
                    abstract = ' '.join(str(text) for text in abstract_texts)
                else:
                    abstract = str(abstract_texts)
            
            return Document(pubmed_id, title, abstract)
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                time.sleep(retry_delay)
            else:
                print(f"    Error fetching PubMed ID {pubmed_id} after {max_retries} attempts: {str(e)}")
                return None

def process_training_file(file_path: str, questions_file: str, documents_file: str) -> tuple[List[Question], Dict[str, Document]]:
    """Process the training file and return questions and documents."""
    print(f"\nReading file: {file_path}")
    
    # Intentar cargar progreso existente
    questions = []
    documents = {}
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            questions = [Question(
                body=q['body'],
                ideal_answer=q['ideal_answer'],
                exact_answer=q['exact_answer'],
                question_type=q['type'],
                index=q.get('index')
            ) for q in questions_data]
        print(f"Loaded {len(questions)} previously processed questions")
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
            documents = {
                pid: Document(pid, doc['title'], doc['abstract'])
                for pid, doc in documents_data.items()
            }
        print(f"Loaded {len(documents)} previously processed documents")
    except FileNotFoundError:
        print("No previous progress found, starting from scratch")

    # Cargar archivo de entrada
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = len(data['questions'])
    print(f"Found {total_questions} questions to process")

    # Comenzar desde donde quedamos
    start_index = len(questions)
    
    for i, q in enumerate(data['questions'][start_index:], start_index + 1):
        print(f"\nProcessing question {i}/{total_questions}")
        print(f"Question: {q['body'][:100]}...")  # Print first 100 chars of question
        
        # Create Question object
        question = Question(
            body=q['body'],
            ideal_answer=q.get('ideal_answer', []),
            exact_answer=q.get('exact_answer'),
            question_type=q['type'],
            index=i
        )
        questions.append(question)

        # Fetch documents from PubMed
        print(f"Fetching {len(q['documents'])} documents for this question")
        for j, doc_url in enumerate(q['documents'], 1):
            pubmed_id = extract_pubmed_id(doc_url)
            
            if pubmed_id not in documents:
                print(f"  Fetching document {j}/{len(q['documents'])} (PubMed ID: {pubmed_id})")
                time.sleep(0.05)
                doc = fetch_pubmed_document(pubmed_id)
                if doc:
                    documents[pubmed_id] = doc
                    documents[pubmed_id].related_questions.add(i)  # Añadimos el índice de la pregunta
                    print("    ✓ Document fetched successfully")
                else:
                    print("    ✗ Failed to fetch document")
            else:
                documents[pubmed_id].related_questions.add(i)  # Añadimos la pregunta a un documento existente
                print(f"  Document {pubmed_id} already fetched, skipping")

        # Guardar progreso después de cada pregunta
        print("\nSaving current progress...")
        save_processed_data(questions, documents, questions_file, documents_file)
        print("Progress saved successfully")

    print(f"\nProcessing complete!")
    print(f"Total questions processed: {len(questions)}")
    print(f"Total unique documents fetched: {len(documents)}")
    return questions, documents

def save_processed_data(questions: List[Question], documents: Dict[str, Document], 
                       questions_file: str, documents_file: str):
    """Save processed data to files."""
    print(f"\nSaving processed data...")
    
    # Save questions
    print(f"Saving questions to {questions_file}")
    questions_data = [
        {
            'index': q.index,
            'body': q.body,
            'ideal_answer': q.ideal_answer,
            'exact_answer': q.exact_answer,
            'type': q.type
        }
        for q in questions
    ]
    
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    print("Questions saved successfully")

    # Save documents
    print(f"Saving documents to {documents_file}")
    documents_data = {
        pubmed_id: {
            'title': doc.title,
            'abstract': doc.abstract,
            'related_questions': list(doc.related_questions)
        }
        for pubmed_id, doc in documents.items()
    }
    
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, indent=2, ensure_ascii=False)
    print("Documents saved successfully")
    print("\nAll processing completed!")

def main():
    questions_file = 'processed_questions.json'
    documents_file = 'processed_documents.json'
    
    # Process the training file
    questions, documents = process_training_file(
        'training13b.json',
        questions_file,
        documents_file
    )
    
    # Final save (aunque ya debería estar todo guardado)
    save_processed_data(
        questions, 
        documents,
        questions_file,
        documents_file
    )

if __name__ == "__main__":
    main()