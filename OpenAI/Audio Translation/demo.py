from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

#initialize open ai client

client = OpenAI(api_key = os.getenv("OPEN_AI_KEY"))  # Fixed variable name



#open the file with the recording

with open("file name.mp3", "rb") as audio: # rb -> is a read binary file

    translate = client.audio.translation.create(
        model = "whisper-1",
        file = audio
    )

print(translate.text)