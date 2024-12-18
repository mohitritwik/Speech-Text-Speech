# -*- coding: utf-8 -*-
import time
from src.capture_audio import capture_audio
from src.speech_to_text import speech_to_text
from src.text_to_speech import text_to_speech
import numpy as np
from translation_model import load_translation_model, translate_text  # Import translation functions

def main():
    # Initialize components
    ds = speech_to_text()  # Initialize the DeepSpeech engine
    translation_model, tokenizer = load_translation_model()  # Load the translation model

    while True:
        input("Press Enter to start listening: ")  # Wait for user input to start listening

        # Initialize variables to store audio data and track silence
        audio_data = np.array([], dtype=np.int16)
        silence_threshold = 1000  # Set a threshold for the volume of "silence"
        silence_duration_limit = 1.0  # Set a limit for how long silence can last (in seconds)
        last_spoken_time = time.time()  # Initialize the time of the last spoken word

        # Continuously capture audio until there is a period of silence longer than the limit
        while True:
            chunk = capture_audio()  # Capture a chunk of audio
            audio_data = np.concatenate((audio_data, chunk))  # Add the chunk to the array of audio data

            # If there is no audio in the chunk, break out of the loop
            if len(chunk) == 0:
                break

            volume = np.abs(chunk).mean()  # Calculate the volume of the chunk

            # If the volume is below the silence threshold, check how long it has been silent for
            if volume < silence_threshold:
                duration = time.time() - last_spoken_time  # Calculate the duration of the silence
                if duration > silence_duration_limit:
                    break

            # If the volume is above the silence threshold, update the time of the last spoken word
            else:
                last_spoken_time = time.time()

        # Transcribe audio data to text
        # text = ds.stt(audio_data)
        text='Hello I am from India.'
        print("You said:", text)

        # Translate the text
        model, tokenizer = load_translation_model()
        translated = translate_text(text, model, tokenizer)
        print("Translated text:", translated)
        # print translated.decode('utf-8')
        # Generate speech output from the translated text
        text_to_speech(translated)

# Ensure the main() function is only called if this script is run directly
if __name__ == "__main__":
    main()
