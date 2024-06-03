from mimetypes import guess_all_extensions, types_map
from itertools import chain

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, FinishReason

import tkinter as tk
import tkinter.filedialog as fd

import pickle
from pickle import dump, load

from dotenv import load_dotenv
import os


root = tk.Tk()
root.attributes("-topmost", True)
root.withdraw()

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt as prompt

project = os.environ.get("project")
location = os.environ.get("location")
credential_location = os.environ.get("credentials")
os.environ["GOOGLE_API_KEY"] = os.environ.get("key")



load_dotenv()
vertexai.init(project=project, location=location, credentials=credential_location)

## Clear the console and print the title
console = Console()
console.clear()
console.print(Markdown("# Parla con Gemini:"))
system_input = prompt.ask("Inserire la funzione che deve avere Gemini (Opzionale):\n")


# Setup for the model
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


system_instruction = [Part.from_text(system_input)] if system_input else None

model = GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings, system_instruction=system_instruction)
chat = model.start_chat(response_validation=False)

loaded_file = None

## Mimes and extensions available for the model
image_mime = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif"
    ]
audio_mime = [
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac"
]
video_mime = [
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp"
    ]
document_mime = [

    "text/plain",
    "text/html",
    "text/css",
    "text/javascript",
    "application/x-javascript",
    "text/x-typescript",
    "application/x-typescript",
    "text/csv",
    "text/markdown",
    "text/x-python",
    "application/x-python-code",
    "application/json",
    "text/xml",
    "application/rtf",
    "text/rtf",
    "application/pdf",
]
valid_mime = image_mime + audio_mime + video_mime + document_mime
extension_file_list = list(chain(*[guess_all_extensions(i) for i in valid_mime]))


def check_file_validation(file_name):
    global valid_mime
    global extension_file_list
    file_extension = get_file_extension(file_name)
    if file_extension in extension_file_list:
        return True
    else:
        return False

def get_file_extension(file_name):
    _, file_extension = os.path.splitext(file_name)
    return file_extension

def get_chat_response(prompt: list):
    global chat
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return Markdown("".join(text_response)) # type: ignore

def get_file():
    global extension_file_list
    global console
    filetypes = [["All Files", "*.*"]] + [[f"{i.upper().lstrip('.')}", f"*{i}"] for i in extension_file_list]
    print(filetypes)
    file_input = fd.askopenfilenames(filetypes=filetypes, title="Seleziona i file da inviare")
    data = {}
    
    for i in file_input:
        if not check_file_validation(i):
            console.print(f"{i} non ha un formato valido per il modello, perciò il file non verrà inviato")
            continue
        with open(i, "rb") as f:
            data[i] = f.read()
    parts = []
    
    for i in (data.keys()):
        console.print(f"Caricamento file {i}")
        j = data[i]
        url = Part.from_data(j, types_map[get_file_extension(i)])
        parts.append(url)
    if bool(data):
        textual_part = str(prompt.ask("Che messaggio vuoi allegare a Gemini?", default="Cosa dice questo documento?", show_default=True))
        return parts + [Part.from_text(textual_part)]
    
    return None

def save_chat():
    global chat
    global console
    global system_input
    global loaded_file
    if loaded_file:
        if os.path.exists(loaded_file):
            ask = prompt.ask(f"La conversazione è parte del file {loaded_file}, vuoi salvarla in un altro file?", default="No", show_default=True, choices=["Si", "No"], show_choices=True)
            if ask == "Si":
                console.print("Scegli dove salvare la conversazione")
                file_location = fd.asksaveasfilename(title="Salva la conversazione", defaultextension=".pkl",filetypes=[("Pickle", "*.pkl"), ("All Files", "*.*")])
            if ask == "No":
                file_location = loaded_file
        else:
            console.print("Scegli dove salvare la conversazione")
            file_location = fd.asksaveasfilename(title="Salva la conversazione", defaultextension=".pkl",filetypes=[("Pickle", "*.pkl"), ("All Files", "*.*")])
    else:
        console.print("Scegli dove salvare la conversazione")
        file_location = fd.asksaveasfilename(title="Salva la conversazione", defaultextension=".pkl",filetypes=[("Pickle", "*.pkl"), ("All Files", "*.*")])
    
    if not file_location:
        console.print("Conversazione non salvata, il nome del file non è stato inserito")
        return
    f = open(file_location, "wb")
    dump([chat.history, system_input], f)
    f.close()
    console.print(f"Conversazione salvata in {file_location}")

def load_chat():
    global chat
    global console
    global model
    global loaded_file
    console.print("Scegli la conversazione da caricare [.pkl]")
    file_location = fd.askopenfilename(title="Carica la conversazione", filetypes=[("Pickle", "*.pkl"), ("All Files", "*.*")])
    if not file_location:
        console.print("Conversazione non caricata, il nome del file non è stato inserito")
        return
    f = open(file_location, "rb")
    loaded = load(f)
    print(type(loaded))
    model = GenerativeModel('gemini-1.5-pro', safety_settings=safety_settings, system_instruction=loaded[1])
    chat = model.start_chat(response_validation=False, history=loaded[0])
    f.close()
    loaded_file = file_location
    console.print(f"Conversazione e istruzioni per gemini caricate da {file_location}")

def stop_chat():
    global chat
    global console
    
    save = prompt.ask("Vuoi salvare la chat?", default="No", show_default=True, choices=["Sì", "No"])
    if save == "Sì":
        save_chat()
    console.print("La conversazione è stata terminata")
    exit()

def stop_chat():
    global console
    try:
        answer = prompt.ask("Vuoi salvare la chat per poterla riaprire? (Sì/No)", default="No", show_default=True)
    except KeyboardInterrupt:
        console.print("Conversazione terminata")
        exit()
    if answer == "Sì":
        save_chat()
    console.print("Conversazione terminata")
    exit()


def print_help():
    global console
    global image_mime
    global audio_mime
    global video_mime
    global document_mime
    help_dictionary = {"/help":"Mostra questo messaggio di aiuto", "/file":"Invia un file tra quelli consentiti", "/exit":"Stoppa la conversazione", "/save":"Salva la conversazione", "/load":"Carica la conversazione"}
    for i,j in help_dictionary.items():
        console.print(f"{i} -> {j}")
    console.print("Le possibili estensioni sono:")
    
    console.print((create_table()))

def create_table():
    global image_mime
    global audio_mime
    global video_mime
    global document_mime
    table = ""
    nested_dictionary = {"Immagini":image_mime, "Audio":audio_mime, "Video":video_mime, "Documento":document_mime}
    nested_dictionary_sorted = {}
    for k in sorted(nested_dictionary, key=lambda k: len(nested_dictionary[k]), reverse=True):
        nested_dictionary_sorted[k] = nested_dictionary[k]
    table = ""
    for i in nested_dictionary_sorted.keys():
        table += f"|{i}"
    table += "\n|---|---|---|---|\n"
    longer_index = [*nested_dictionary_sorted.keys()][0]

    for i in nested_dictionary_sorted:
        nested_dictionary_sorted[i].extend(" " for _ in range(len(nested_dictionary_sorted[longer_index]) - len(nested_dictionary_sorted[i])))
    for i in range(len(nested_dictionary_sorted[longer_index])):
        for j in nested_dictionary_sorted:
            if {nested_dictionary_sorted[j][i]} != " ":
                table += f"| - {nested_dictionary_sorted[j][i]}"
            if {nested_dictionary_sorted[j][i]} == " ":
                table += f"|   "
        table += "\n"
    return Markdown(table)


    


try:
    input_chat = str(prompt.ask("Cosa vuoi chiedermi?", default="/help", show_default=True))
    while input_chat:
        if input_chat == "/file":
            message = get_file()
            if bool(message):
                response = get_chat_response(message)
                console.print(response)
        if input_chat == "/exit":
            stop_chat()
        if input_chat == "/help":
            print_help()
        if input_chat == "/save":
            save_chat()
        if input_chat == "/load":
            load_chat()
        elif input_chat not in ["/file", "/exit", "/help", "/save"]:
            response = get_chat_response(Part.from_text(input_chat))
            console.print(response)
        input_chat = str(prompt.ask("Cosa vuoi chiedermi?\n"))
except KeyboardInterrupt:
    stop_chat()

    

    
    



