import speech_recognition as sr
import numpy as np

# Define constants
RATE = 16000
DURATION = 5  # Duration to capture in seconds

def capture_audio():
    """Captures audio using the SpeechRecognition library and returns the audio data as a numpy array in float32 format."""
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Use the default microphone as the source
    with sr.Microphone(sample_rate=RATE) as source:
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        print("Recording audio for {} seconds...".format(DURATION))
        # Capture the audio data from the microphone
        audio = recognizer.record(source, duration=DURATION)
    
    # Get the raw audio data as bytes
    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    
    # Convert to float32 and normalize between -1.0 and 1.0
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    return audio_data

# Example usage
if __name__ == "__main__":
    captured_audio = capture_audio()
    print("Captured audio data:", captured_audio)
