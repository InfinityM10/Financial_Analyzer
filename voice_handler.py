# voice_handler.py - Speech-to-Text and Text-to-Speech functionality

import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_available = True
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.tts_available = False
        
        # Initialize STT
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.stt_available = True
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            self.stt_available = False
        
        self.is_listening = False
        self.is_speaking = False
    
    def speak(self, text: str, callback: Optional[Callable] = None) -> bool:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            callback: Optional callback function to call when done
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.tts_available:
            logger.error("TTS not available")
            return False
        
        try:
            self.is_speaking = True
            
            def speak_text():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
                    if callback:
                        callback()
                except Exception as e:
                    logger.error(f"Error during speech: {e}")
                    self.is_speaking = False
            
            # Run TTS in separate thread to avoid blocking
            speech_thread = threading.Thread(target=speak_text)
            speech_thread.daemon = True
            speech_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in speak method: {e}")
            self.is_speaking = False
            return False
    
    def listen(self, timeout: int = 5, phrase_timeout: int = 1) -> Optional[str]:
        """
        Listen for speech and convert to text
        
        Args:
            timeout: Maximum time to wait for speech to start
            phrase_timeout: Maximum time to wait for phrase to complete
        
        Returns:
            str: Recognized text, or None if failed
        """
        if not self.stt_available:
            logger.error("STT not available")
            return None
        
        try:
            self.is_listening = True
            
            with self.microphone as source:
                logger.info("Listening for speech...")
                # Listen for the first phrase and extract it into audio data
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_timeout
                )
            
            self.is_listening = False
            
            # Recognize speech using Google Speech Recognition
            logger.info("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
            self.is_listening = False
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            self.is_listening = False
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech service: {e}")
            self.is_listening = False
            return None
        except Exception as e:
            logger.error(f"Error in listen method: {e}")
            self.is_listening = False
            return None
    
    def listen_continuously(self, callback: Callable[[str], None], stop_event: threading.Event):
        """
        Listen continuously for speech commands
        
        Args:
            callback: Function to call with recognized text
            stop_event: Threading event to stop listening
        """
        if not self.stt_available:
            logger.error("STT not available for continuous listening")
            return
        
        while not stop_event.is_set():
            try:
                text = self.listen(timeout=1, phrase_timeout=3)
                if text:
                    callback(text)
            except Exception as e:
                logger.error(f"Error in continuous listening: {e}")
                time.sleep(1)
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.tts_available and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def get_status(self) -> dict:
        """Get status of voice features"""
        return {
            "tts_available": self.tts_available,
            "stt_available": self.stt_available,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking
        }

# Global voice handler instance
voice_handler = VoiceHandler()

def test_voice_features():
    """Test function for voice features"""
    print("Testing Voice Features...")
    
    # Test TTS
    print("Testing Text-to-Speech...")
    if voice_handler.speak("Hello! I am your financial assistant. Voice features are now working."):
        print("✅ TTS working")
    else:
        print("❌ TTS failed")
    
    # Wait for TTS to complete
    time.sleep(3)
    
    # Test STT
    print("Testing Speech-to-Text...")
    print("Please say something (you have 5 seconds)...")
    
    text = voice_handler.listen(timeout=5, phrase_timeout=3)
    if text:
        print(f"✅ STT working - You said: {text}")
        voice_handler.speak(f"I heard you say: {text}")
    else:
        print("❌ STT failed or no speech detected")
    
    print("Voice feature testing complete!")

if __name__ == "__main__":
    test_voice_features()