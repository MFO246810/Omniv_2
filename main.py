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
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


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

def clearTextInput():
    inputField.delete("1.0", tk.END)

def clearTextOutput():
    outputWindow.delete("1.0", tk.END)

def writeTextInput(string):
    inputField.insert("1.0", string)

def writeTextOutput(string):
    outputWindow.insert("1.0", string)

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
            {"role": "system", "content": f"You are a user manual. Tell the user the answer to their question or explain how to help them. Make it brief and informative. This is a demo of you, so keep the responses VERY brief and VERY information dense. Three sentences at most. The pages of your manual are as follows: {document}"},
            {"role": "user", "content": user_query}
        ]
    )
    message = query_summary.choices[0].message.content
    return message

def recordClick():
    record_audio()
    clearTextInput()
    writeTextInput(transcribe_audio(output_file_wav))

def searchClick():
    Data = transcribe_audio(output_file_wav)
    result = Gen_AI(Data)
    writeTextOutput(result)
    Generate_audio(result)

def openPageClick():
    hi = 2

def newDocClick():
    filePath = filedialog.askopenfilename(title="Select File",
                                          filetypes=[("PDF Files", "*.pdf"),
                                                     ("All Files", "*.*")])
#Setting up the root window

rootWindow = tk.Tk()
rootWindow.title("Fast Information Navigation through Documents")
rootWindow.geometry("600x400")
rootWindow.minsize(600, 400)
rootWindow.configure(bg="lightblue")

PADDING = 10

rootWindow.columnconfigure(0, weight = 1)
rootWindow.columnconfigure(1, weight = 0)
rootWindow.rowconfigure(0, weight = 0)
rootWindow.rowconfigure(1, weight = 1)

#Speech Input Field
inputField = tk.Text(rootWindow, wrap=tk.WORD, state = "normal", height = 5)
inputField.grid(row = 0, column = 0, sticky = "nsew", padx = PADDING, pady = PADDING)

# Add monospace text
omniSplashscreen = """

    ______       _____       _   _       _____      
   |  ____|     |_   _|     | \ | |     |  __ \     
   | |__          | |       |  \| |     | |  | |    
   |  __|         | |       | . ` |     | |  | |    
   | |       _   _| |_   _  | |\  |  _  | |__| |  _ 
   |_|      (_) |_____| (_) |_| \_| (_) |_____/  (_)
                                                  
                                                  
   Fast Information Navigation through Documents...
"""

# Insert the banner text into the text window
inputField.insert(tk.END, omniSplashscreen)

# Make sure the text is visible when the app starts
inputField.yview(tk.END)

#Output Window
outputWindow = tk.Text(rootWindow)
outputWindow.grid(row = 1, column = 0, columnspan = 2, sticky = "nsew", padx = PADDING, pady = PADDING)

#Button Frame
buttonFrame = ttk.Frame(rootWindow, width = 120)
buttonFrame.grid(row = 0, column = 1, sticky = "nsew")
buttonFrame.rowconfigure(0, weight=0)  # Top padding
buttonFrame.rowconfigure(1, weight=0)  # Record Button
buttonFrame.rowconfigure(2, weight=0)  # Search Button
buttonFrame.rowconfigure(3, weight=0)  # Open Page Button
buttonFrame.rowconfigure(4, weight=1)  # Push buttons to top (optional)

buttonFrame.configure(style="Button.TFrame")
style = ttk.Style()
style.configure("Button.TFrame", background="lightblue")

#Record Button
recordButton = tk.Button(
    buttonFrame,
    text="Record",
    relief="raised",
    borderwidth=1,
    highlightthickness = 0,
    width=15,  # Approx 100 pixels
    height=3,
    anchor="center",
    command = recordClick
)
recordButton.grid(row = 1, column = 0, padx = PADDING, pady = PADDING)

#Search Button
searchButton = tk.Button(
    buttonFrame,
    text="Search!",
    relief="raised",
    borderwidth=1,
    highlightthickness = 0,
    width=15,  # Approx 100 pixels
    height=3,
    anchor="center",
    command = searchClick
)
searchButton.grid(row = 2, column = 0)

#Open Page button
openPageButton = tk.Button(
    buttonFrame,
    text="Open Page",
    relief="raised",
    borderwidth=1,
    highlightthickness=0,
    width=15,  # Approx 100 pixels
    height=3,
    anchor="center",
    command = openPageClick
)
openPageButton.grid(row=3, column=0, pady = PADDING)

#New Document button
newDocumentButton = tk.Button(
    buttonFrame,
    text="Add New \n Document",
    relief="raised",
    borderwidth=1,
    highlightthickness=0,
    width=15,  # Approx 100 pixels
    height=3,
    anchor="center",
    command = newDocClick
)
newDocumentButton.grid(row=4, column=0)

rootWindow.mainloop()
