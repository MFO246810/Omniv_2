import os
import io
import sys
from chromadb import chromadb
import pymupdf
import json
from openai import OpenAI
import string
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

chroma_client = chromadb.PersistentClient(path="./VDB")
collections = chroma_client.list_collections()
existing_collection = None
for coll in collections:
    if coll == "Textbook":
        existing_collection = coll
        break

if existing_collection:
    print(existing_collection)
    collection = chroma_client.get_collection(name="Textbook")
else:
    collection = chroma_client.create_collection(name="Textbook")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

chunks = []
embedded_chunks = []
def chunker(pdf_path):
    doc = pymupdf.open(pdf_path)
    metadata = doc.metadata
    title = metadata.get("title", "untitled")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text = text.replace("\n", " ")
        chunks.append({'Data': text,'Title': title, 'page': page_num, 'filepath' : pdf_path})
        
def generate_embed(chunk):
    embedded = embedder.encode(chunk['Data'])
    embedded_chunk = {'Data': embedded, 'Document': chunk['Data'], 'page': chunk['page'], 'Title': chunk['Title']}
    return embedded_chunk

def Save_to_db(embedded_chunk):
    collection.add(
        documents=[embedded_chunk['Document']],  
        ids=[str(embedded_chunk['Title'] + " page: " + str(embedded_chunk['page']))],  
        embeddings=[embedded_chunk['Data']],
        metadatas=[{'Title': embedded_chunk['Title'], 'Page': embedded_chunk['page']}],
    )

def main(file_path):
    chunker(file_path)
    for chunk in chunks:
        Save_to_db(generate_embed(chunk))
      
directory_path = "TAMUhack Documentation\Textbooks"
for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    
    if os.path.isfile(file_path):
        main(file_path)
        os.remove(file_path)
        
        


    
