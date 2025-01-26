import os
import boto3
import speech_recognition as sr
from openai import OpenAI
import pygame
import chromadb
import time
import requests
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
#from phoenix.otel import register
#from opentelemetry import trace

#os.environ["OTEL_EXPORTER_OTLP_HEADERS"]=api_key='c0554391ead03552c62:55adf63'
#os.environ["PHOENIX_CLIENT_HEADERS"]=api_key='c0554391ead03552c62:55adf63'
#os.environ["PHOENIX_COLLECTOR_ENDPOINT"]='https://app.phoenix.arize.com'

#tracer_provider = register(
#  project_name="Tamuhack-project",
#  endpoint="https://app.phoenix.arize.com/v1/traces"
#)
#tracer = trace.get_tracer("phoenix")

load_dotenv()
pygame.init()
pygame.mixer.init()
metadata = []
is_recording = False
output_file_wav = "recorded_audio.wav"
chroma_client = chromadb.PersistentClient(path="./VDB")
collections = chroma_client.get_collection(name="Textbook")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

polly = boto3.client("polly",
    aws_access_key_id =os.getenv("Prolly_Access_Code"),
    aws_secret_access_key= os.getenv("Prolly_Secret_Code"),
    region_name="us-west-1"
)
Api_key = os.getenv('Open_Ai')
ai_bot = OpenAI(api_key=Api_key)



def play_audio(filename):
    pygame.mixer.music.load(f"{filename}.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

def Generate_audio(text):
    response = polly.synthesize_speech(
    Text=text,
    OutputFormat="mp3",
    VoiceId="Joanna"
    )

    with open("speech.mp3", "wb") as audio_file:
        audio_file.write(response["AudioStream"].read())

    play_audio("speech")

def record_audio():
    is_recording = True
    print("Recording started...")
    frequency = 44100  
    duration = 10
    
    recording = sd.rec(int(duration * frequency), samplerate=frequency, channels=1, dtype='int16')  
    time.sleep(duration)
    write(output_file_wav, frequency, recording)  
    print("Recording saved to", output_file_wav)

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped.")

def transcribe_audio(file_uri):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_uri) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print(text)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

def Gen_AI(text):
    documents = getting_doc_loc(text)
    metadata = documents["metadatas"]
    print(metadata)
    print(documents['documents'])
    return Response(documents['documents'], text)  
    
def process_query(user_query):
    query_summary = ai_bot.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Refine the query into a short string."},
            {"role": "user", "content": user_query}
        ]
    )
    message = query_summary.choices[0].message.content
    print(message)
    return embedder.encode(message)

def getting_doc_loc(user_query):
    results = collections.query(
        query_embeddings=[(process_query(user_query))],
        n_results=5
    )
    
    return{"documents": results["documents"],
        "metadatas": results["metadatas"],
        "ids": results["ids"]}     

def Response(document, user_query):
    query_summary = ai_bot.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"The user's query can be answered using the information provided. Answer the user's query using only the following information: {document}"},
            {"role": "user", "content": user_query}
        ]
    )
    message = query_summary.choices[0].message.content
    print(message)
    return message
    
def input_generation():
    record_audio()
    Generate_audio(Gen_AI(transcribe_audio(output_file_wav)))
    
input_generation()
