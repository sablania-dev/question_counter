import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_TOKEN"

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from collections import defaultdict
from operator import itemgetter
import numpy as np

# Load questions from a text file
loader = TextLoader("questions.txt", encoding="utf-8")
docs = loader.load()
questions = [doc.page_content for doc in docs]

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store questions in FAISS for similarity search
vector_db = FAISS.from_texts(questions, embedding_model)

def compute_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def cluster_questions(questions, vector_db, threshold=0.8):
    grouped_questions = defaultdict(list)
    question_embeddings = {q: embedding_model.embed_query(q) for q in questions}
    
    for idx, question in enumerate(questions):
        question_vector = question_embeddings[question]
        found_group = None
        
        for super_question, q_list in grouped_questions.items():
            super_vector = question_embeddings[super_question]
            similarity = compute_similarity(question_vector, super_vector)
            if similarity >= threshold:
                found_group = super_question
                break
        
        if found_group:
            grouped_questions[found_group].append(question)
        else:
            grouped_questions[question] = [question]
    
    return grouped_questions

# Perform clustering
clusters = cluster_questions(questions, vector_db)

# Initialize LLaMA 3.2-1B-Instruct
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={"temperature": 0.5}
)

def generate_super_question(question_list):
    prompt = PromptTemplate.from_template(
        "You are given the following similar questions:\n\n{questions}\n\n"
        "Create a single representative 'super question' that captures their meaning."
    )
    return llm.invoke(prompt.format(questions="\n".join(question_list)))

super_questions = []
for cluster, q_list in clusters.items():
    super_question = generate_super_question(q_list)
    super_questions.append((super_question, len(q_list)))

# Sort by frequency
super_questions_sorted = sorted(super_questions, key=itemgetter(1), reverse=True)

def answer_question(question):
    prompt = PromptTemplate.from_template(
        "Answer the following question concisely:\n\n{question}"
    )
    return llm.invoke(prompt.format(question=question))

# Generate answers
final_output = []
for super_question, count in super_questions_sorted:
    answer = answer_question(super_question)
    final_output.append({"super_question": super_question, "count": count, "answer": answer})

# Print results
for item in final_output:
    print(f"Q ({item['count']} times): {item['super_question']}\nA: {item['answer']}\n---")
