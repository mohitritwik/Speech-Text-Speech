import whisper

class speech_to_text:
    def __init__(self, model_size="base"):
        """
        Initializes the Whisper speech-to-text engine with the specified model size.
        Whisper provides multiple models ('tiny', 'base', 'small', 'medium', 'large') to choose from 
        based on the trade-off between speed and accuracy.
        """
        # Load the Whisper model. Choose between 'tiny', 'base', 'small', 'medium', 'large'
        self.model = whisper.load_model('base')

    def stt(self, audio_data):
        """Performs speech-to-text transcription on the specified audio file."""
        # Transcribe the audio file using Whisper
        result = self.model.transcribe(audio_data)
        return result['text']
