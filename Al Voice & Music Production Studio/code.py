import streamlit as st
import os
import base64
import tempfile
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from gtts import gTTS
import librosa
import librosa.display
import soundfile as sf
from audiocraft.models import MusicGen, AudioGen
import matplotlib.pyplot as plt
import io
import pandas as pd
import time
import json
import logging
import threading
import queue
import re
from scipy.signal import butter, filtfilt
from scipy import signal
import concurrent.futures
import uuid
import shutil
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any, Union
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice_mixer")

# Constants
SAMPLE_RATE = 44100
OUTPUT_DIR = "output"
PRESETS_DIR = "presets"
PROJECTS_DIR = "projects"

# Ensure directories exist
for directory in [OUTPUT_DIR, PRESETS_DIR, PROJECTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Audio processing queue for background tasks
processing_queue = queue.Queue()

# ------------------- Model Loading -------------------

@st.cache_resource
def load_musicgen_model(model_size="small"):
    """Load and cache the MusicGen model."""
    try:
        logger.info(f"Loading MusicGen {model_size} model...")
        if model_size == "small":
            model = MusicGen.get_pretrained("facebook/musicgen-small")
        elif model_size == "medium":
            model = MusicGen.get_pretrained("facebook/musicgen-medium")
        elif model_size == "melody":
            model = MusicGen.get_pretrained("facebook/musicgen-melody")
        else:
            model = MusicGen.get_pretrained("facebook/musicgen-small")
        
        logger.info(f"MusicGen {model_size} model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading MusicGen model: {e}")
        st.error(f"Error loading MusicGen model: {str(e)}")
        return None

@st.cache_resource
def load_audiogen_model():
    """Load and cache the AudioGen model for sound effects."""
    try:
        logger.info("Loading AudioGen model...")
        model = AudioGen.get_pretrained("facebook/audiogen-medium")
        logger.info("AudioGen model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading AudioGen model: {e}")
        st.error(f"Error loading AudioGen model: {str(e)}")
        return None

# ------------------- Speech Generation -------------------

def generate_speech(text: str, lang: str = "en", voice_provider: str = "gtts", 
                    filename: str = "speech.wav") -> Tuple[Optional[str], Optional[np.ndarray], Optional[int]]:
    """Generate speech from text using various TTS providers."""
    try:
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        path = os.path.join(OUTPUT_DIR, filename)
        temp_speech = None
        
        if voice_provider == "gtts":
            # Create temp file with unique name
            temp_dir = tempfile.gettempdir()
            temp_filename = f"speech_temp_{uuid.uuid4().hex}.mp3"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # Generate speech
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(temp_path)
                
                # Wait a brief moment to ensure file is fully written
                time.sleep(0.1)
                
                # Load the audio
                waveform, sample_rate = torchaudio.load(temp_path)
                waveform_np = waveform.numpy()[0]
                
                # Resample to target sample rate if needed
                if sample_rate != SAMPLE_RATE:
                    waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
                    sample_rate = SAMPLE_RATE
                
                sf.write(path, waveform_np, sample_rate)
                logger.info(f"Speech generated successfully: {path}")
                return path, waveform_np, sample_rate
            
            finally:
                # Ensure temp file cleanup with proper error handling
                try:
                    if os.path.exists(temp_path):
                        # Try multiple times with small delays
                        for _ in range(3):
                            try:
                                os.unlink(temp_path)
                                break
                            except PermissionError:
                                time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")
        else:
            # Placeholder for additional TTS services
            st.warning(f"Voice provider {voice_provider} not implemented yet, falling back to gTTS")
            return generate_speech(text, lang, "gtts", filename)
            
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        st.error(f"Error generating speech: {str(e)}")
        return None, None, None

# ------------------- Audio Processing -------------------

def apply_audio_effects(waveform: np.ndarray, sample_rate: int, effects: Dict[str, Any]) -> np.ndarray:
    """Apply various audio effects to the waveform."""
    try:
        processed = waveform.copy()
        
        # Apply pitch shift if enabled
        if effects.get("pitch_shift", {}).get("enabled", False):
            pitch_factor = effects["pitch_shift"]["amount"]
            if pitch_factor != 0:
                if torch.cuda.is_available() and abs(pitch_factor) <= 4:
                    waveform_tensor = torch.tensor(processed).unsqueeze(0)
                    pitch_shift = T.PitchShift(sample_rate, pitch_factor)
                    processed = pitch_shift(waveform_tensor).squeeze(0).numpy()
                else:
                    processed = librosa.effects.pitch_shift(processed, sr=sample_rate, n_steps=pitch_factor)
        
        # Apply reverb if enabled
        if effects.get("reverb", {}).get("enabled", False):
            reverb_amount = effects["reverb"]["amount"]
            if reverb_amount > 0:
                reverb_time = 1 + (reverb_amount * 4)  # 1-5 seconds based on amount
                # Simple convolution reverb
                impulse_response = np.exp(-np.linspace(0, reverb_time, int(reverb_time * sample_rate)))
                processed = signal.convolve(processed, impulse_response, mode='full')[:len(processed)]
        
        # Apply EQ if enabled
        if effects.get("eq", {}).get("enabled", False):
            # Apply 3-band EQ
            low_gain = effects["eq"].get("low_gain", 0)
            mid_gain = effects["eq"].get("mid_gain", 0)
            high_gain = effects["eq"].get("high_gain", 0)
            
            if low_gain != 0 or mid_gain != 0 or high_gain != 0:
                # Low shelf filter
                if low_gain != 0:
                    b, a = butter(2, 300 / (sample_rate/2), 'lowpass')
                    low_band = filtfilt(b, a, processed)
                    processed = processed + (low_band * (low_gain / 10))
                
                # High shelf filter
                if high_gain != 0:
                    b, a = butter(2, 3000 / (sample_rate/2), 'highpass')
                    high_band = filtfilt(b, a, processed)
                    processed = processed + (high_band * (high_gain / 10))
                
                # Mid band is what remains
                if mid_gain != 0:
                    b1, a1 = butter(2, 300 / (sample_rate/2), 'highpass')
                    b2, a2 = butter(2, 3000 / (sample_rate/2), 'lowpass')
                    mid_band = filtfilt(b1, a1, processed)
                    mid_band = filtfilt(b2, a2, mid_band)
                    processed = processed + (mid_band * (mid_gain / 10))
                
                # Normalize after EQ
                processed = processed / (np.max(np.abs(processed)) + 1e-8)
        
        # Apply compression if enabled
        if effects.get("compression", {}).get("enabled", False):
            threshold = effects["compression"].get("threshold", -20)
            ratio = effects["compression"].get("ratio", 4)
            
            # Convert threshold from dB to amplitude
            threshold_amp = 10 ** (threshold / 20)
            
            # Simple compression
            mask = np.abs(processed) > threshold_amp
            processed[mask] = threshold_amp + (processed[mask] - threshold_amp) / ratio
        
        return processed
        
    except Exception as e:
        logger.error(f"Error applying audio effects: {e}")
        st.error(f"Error applying audio effects: {str(e)}")
        return waveform

def generate_audio_segment(model_type: str, model: Any, prompt: str, duration: float, 
                           filename: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[int]]:
    """Generate an audio segment using AI models."""
    try:
        path = os.path.join(OUTPUT_DIR, filename)
        
        # Set reasonable limits for duration
        max_duration = min(30.0, float(duration))
        
        # Use appropriate model
        if model_type == "music":
            model.set_generation_params(duration=int(max_duration))
            
            # Generate music with proper error handling
            with torch.no_grad():
                audio = model.generate([prompt])
            
            waveform = audio[0].cpu().numpy()[0]
            sample_rate = 32000  # MusicGen uses 32kHz
            
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
                sample_rate = SAMPLE_RATE
        
        elif model_type == "sfx":
            model.set_generation_params(duration=int(max_duration))
            
            with torch.no_grad():
                audio = model.generate([prompt])
            
            waveform = audio[0].cpu().numpy()[0]
            sample_rate = 16000  # AudioGen uses 16kHz
            
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
                sample_rate = SAMPLE_RATE
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        sf.write(path, waveform, sample_rate)
        logger.info(f"Audio generated successfully: {path}")
        return path, waveform, sample_rate
    
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        st.error(f"Error generating audio: {str(e)}")
        return None, None, None

def mix_audio_segments(segments: List[Dict[str, Any]], main_sample_rate: int, filename: str = "mix.wav") -> Optional[str]:
    """Mix multiple audio segments with volume control and effects."""
    try:
        if not segments:
            raise ValueError("No audio segments provided for mixing")
            
        # Find the longest segment to determine output length
        max_length = 0
        for segment in segments:
            if segment["waveform"] is None:
                continue
                
            length = len(segment["waveform"])
            if length > max_length:
                max_length = length
        
        if max_length == 0:
            raise ValueError("All segments are empty")
        
        # Create output mix (initialize with zeros)
        mixed_waveform = np.zeros(max_length)
        
        # Mix each segment with proper volume control
        for segment in segments:
            if segment["waveform"] is None or not segment["enabled"]:
                continue
            
            # Apply effects if any
            processed_waveform = apply_audio_effects(
                segment["waveform"], 
                main_sample_rate,
                segment.get("effects", {})
            )
            
            # Adjust segment length to match output
            seg_length = len(processed_waveform)
            if seg_length > max_length:
                processed_waveform = processed_waveform[:max_length]
            elif seg_length < max_length:
                # Handle loop option
                if segment.get("loop", False):
                    repeats = int(np.ceil(max_length / seg_length))
                    processed_waveform = np.tile(processed_waveform, repeats)[:max_length]
                else:
                    # Zero-pad if not looping
                    temp = np.zeros(max_length)
                    temp[:seg_length] = processed_waveform
                    processed_waveform = temp
            
            # Apply panning if needed
            pan = segment.get("pan", 0)  # -1.0 (left) to 1.0 (right)
            if pan != 0 and len(processed_waveform.shape) == 1:
                # Convert to stereo for panning
                left_gain = 1.0 - max(0, pan)
                right_gain = 1.0 + min(0, pan)
                stereo_waveform = np.vstack((processed_waveform * left_gain, processed_waveform * right_gain))
                processed_waveform = stereo_waveform
            
            # Apply volume
            volume = segment.get("volume", 1.0)
            processed_waveform = processed_waveform * volume
            
            # Add to mix
            mixed_waveform = mixed_waveform + processed_waveform
        
        # Apply master effects
        # TODO: Add master effects processing here
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_waveform))
        if max_val > 0.99:
            mixed_waveform = mixed_waveform / (max_val + 1e-8)
        
        # Apply soft clipping to prevent harsh distortion
        mixed_waveform = np.tanh(mixed_waveform)
        
        # Save the final mix
        path = os.path.join(OUTPUT_DIR, filename)
        sf.write(path, mixed_waveform, main_sample_rate)
        
        logger.info(f"Audio mixed successfully: {path}")
        return path
    
    except Exception as e:
        logger.error(f"Error mixing audio: {e}")
        st.error(f"Error mixing audio: {str(e)}")
        return None

# ------------------- Visualization -------------------

def create_waveform_plot(waveform: np.ndarray, sample_rate: int) -> Optional[str]:
    """Create a waveform visualization and return as base64 image."""
    try:
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(waveform)/sample_rate, len(waveform)), waveform)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode()
    
    except Exception as e:
        logger.error(f"Error creating waveform plot: {e}")
        return None

def create_spectrogram(waveform: np.ndarray, sample_rate: int) -> Optional[str]:
    """Create a spectrogram visualization and return as base64 image."""
    try:
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode()
    
    except Exception as e:
        logger.error(f"Error creating spectrogram: {e}")
        return None

# ------------------- Project Management -------------------

def save_project(project_data: Dict[str, Any], project_name: str) -> bool:
    """Save project data to file."""
    try:
        if not project_name:
            project_name = f"project_{int(time.time())}"
        
        # Create project directory
        project_dir = os.path.join(PROJECTS_DIR, re.sub(r'[^\w\-_]', '_', project_name))
        os.makedirs(project_dir, exist_ok=True)
        
        # Save audio files
        audio_paths = {}
        for segment_id, segment in project_data.get("segments", {}).items():
            if "path" in segment and os.path.exists(segment["path"]):
                filename = f"{segment_id}.wav"
                dest_path = os.path.join(project_dir, filename)
                shutil.copy(segment["path"], dest_path)
                audio_paths[segment_id] = filename
        
        # Prepare project JSON (exclude numpy arrays and other non-serializable data)
        json_data = {
            "name": project_name,
            "created": datetime.now().isoformat(),
            "segments": {},
            "settings": project_data.get("settings", {})
        }
        
        for segment_id, segment in project_data.get("segments", {}).items():
            json_data["segments"][segment_id] = {
                "type": segment.get("type", ""),
                "filename": audio_paths.get(segment_id, ""),
                "enabled": segment.get("enabled", True),
                "volume": segment.get("volume", 1.0),
                "pan": segment.get("pan", 0.0),
                "loop": segment.get("loop", False),
                "effects": segment.get("effects", {}),
                "metadata": {
                    k: v for k, v in segment.get("metadata", {}).items()
                    if k not in ["waveform", "path"]
                }
            }
        
        # Save project file
        with open(os.path.join(project_dir, "project.json"), 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Project saved: {project_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving project: {e}")
        st.error(f"Error saving project: {str(e)}")
        return False

def load_project(project_name: str) -> Optional[Dict[str, Any]]:
    """Load project data from file."""
    try:
        project_dir = os.path.join(PROJECTS_DIR, project_name)
        
        if not os.path.exists(project_dir):
            raise ValueError(f"Project not found: {project_name}")
        
        # Load project file
        with open(os.path.join(project_dir, "project.json"), 'r') as f:
            project_data = json.load(f)
        
        # Load audio files
        for segment_id, segment in project_data.get("segments", {}).items():
            if "filename" in segment and segment["filename"]:
                audio_path = os.path.join(project_dir, segment["filename"])
                if os.path.exists(audio_path):
                    waveform, sample_rate = sf.read(audio_path)
                    segment["path"] = audio_path
                    segment["waveform"] = waveform
                    segment["sample_rate"] = sample_rate
        
        logger.info(f"Project loaded: {project_name}")
        return project_data
    
    except Exception as e:
        logger.error(f"Error loading project: {e}")
        st.error(f"Error loading project: {str(e)}")
        return None

def list_projects() -> List[str]:
    """List available projects."""
    try:
        return [d for d in os.listdir(PROJECTS_DIR) 
                if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return []

# ------------------- Export Options -------------------

def export_audio(mix_path: str, format: str = "wav", quality: str = "high") -> Optional[str]:
    """Export audio to various formats with quality settings."""
    try:
        if not os.path.exists(mix_path):
            raise ValueError(f"Mix file not found: {mix_path}")
        
        # Load audio data
        waveform, sample_rate = sf.read(mix_path)
        
        # Determine output path
        export_path = os.path.splitext(mix_path)[0] + "." + format.lower()
        
        if format.lower() == "wav":
            sf.write(export_path, waveform, sample_rate, 
                     subtype='PCM_24' if quality == "high" else 'PCM_16')
        
        elif format.lower() == "mp3":
            # For MP3, we need to use a different approach
            # First save as WAV, then convert to MP3
            temp_path = os.path.splitext(mix_path)[0] + "_temp.wav"
            sf.write(temp_path, waveform, sample_rate)
            
            try:
                import subprocess
                bitrate = "320k" if quality == "high" else "192k" if quality == "medium" else "128k"
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_path, 
                    "-b:a", bitrate, export_path
                ], check=True)
                
                # Remove temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                # If ffmpeg fails, just copy the WAV file
                shutil.copy(temp_path, export_path)
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        elif format.lower() == "ogg":
            sf.write(export_path, waveform, sample_rate, format='OGG',
                     subtype='VORBIS')
        
        else:
            # Default to WAV
            sf.write(export_path, waveform, sample_rate)
        
        logger.info(f"Audio exported to {format}: {export_path}")
        return export_path
    
    except Exception as e:
        logger.error(f"Error exporting audio: {e}")
        st.error(f"Error exporting audio: {str(e)}")
        return None

# ------------------- File Management -------------------

def get_binary_file_downloader_html(file_path: str, file_label: str) -> Optional[str]:
    """Create a download link for a file."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        ext = os.path.splitext(file_path)[1][1:].lower()
        mime_type = "audio/wav"
        
        if ext == "mp3":
            mime_type = "audio/mpeg"
        elif ext == "ogg":
            mime_type = "audio/ogg"
        
        download_link = f'''
        <a href="data:{mime_type};base64,{b64}" 
           download="{os.path.basename(file_path)}" 
           class="download-button">
           Download {file_label}
        </a>
        '''
        return download_link
    
    except Exception as e:
        logger.error(f"Error creating download link: {e}")
        return None

# ------------------- UI Components -------------------
def show_audio_segment_controls(segment: Dict[str, Any], key_prefix: str) -> Dict[str, Any]:
    """Show controls for an audio segment."""
    updated_segment = segment.copy()
    
    # Common controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        updated_segment["enabled"] = st.checkbox("Enabled", segment.get("enabled", True), key=f"{key_prefix}_enabled")
    
    with col2:
        if segment.get("waveform") is not None:
            waveform_b64 = create_waveform_plot(segment["waveform"], segment.get("sample_rate", SAMPLE_RATE))
            if waveform_b64:
                st.image(f"data:image/png;base64,{waveform_b64}", use_column_width=True)
    
    with col3:
        st.audio(segment.get("path", ""), format="audio/wav")
    
    # Volume and Pan controls
    col1, col2 = st.columns(2)
    with col1:
        updated_segment["volume"] = st.slider(
            "Volume", 0.0, 2.0, segment.get("volume", 1.0), 0.05,
            key=f"{key_prefix}_volume"
        )
    
    with col2:
        updated_segment["pan"] = st.slider(
            "Pan (L/R)", -1.0, 1.0, segment.get("pan", 0.0), 0.1,
            key=f"{key_prefix}_pan",
            help="Pan audio left (-1.0) or right (1.0)"
        )
    
    # Type-specific controls
    if segment.get("type") == "music" or segment.get("type") == "sfx":
        if st.checkbox("Loop", segment.get("loop", False), key=f"{key_prefix}_loop"):
            updated_segment["loop"] = True
        else:
            updated_segment["loop"] = False
    
    # Effects
    st.markdown("#### Audio Effects")
    effects = segment.get("effects", {})
    
    # Instead of using an expander, just show the effects controls directly
    
    # Pitch shift
    st.markdown("**Pitch Shift**")
    pitch_shift = effects.get("pitch_shift", {"enabled": False, "amount": 0})
    col1, col2 = st.columns([1, 3])
    with col1:
        pitch_enabled = st.checkbox("Enable", pitch_shift.get("enabled", False), key=f"{key_prefix}_pitch_enabled")
    with col2:
        pitch_amount = st.slider(
            "Amount", -12, 12, pitch_shift.get("amount", 0), 1,
            key=f"{key_prefix}_pitch_amount",
            help="Shift pitch up or down (semitones)"
        )
    updated_segment.setdefault("effects", {})["pitch_shift"] = {
        "enabled": pitch_enabled,
        "amount": pitch_amount
    }
    
    # Reverb
    st.markdown("**Reverb**")
    reverb = effects.get("reverb", {"enabled": False, "amount": 0.3})
    col1, col2 = st.columns([1, 3])
    with col1:
        reverb_enabled = st.checkbox("Enable", reverb.get("enabled", False), key=f"{key_prefix}_reverb_enabled")
    with col2:
        reverb_amount = st.slider(
            "Amount", 0.0, 1.0, reverb.get("amount", 0.3), 0.05,
            key=f"{key_prefix}_reverb_amount",
            help="Add space/reverb to the sound"
        )
    updated_segment.setdefault("effects", {})["reverb"] = {
        "enabled": reverb_enabled,
        "amount": reverb_amount
    }
    
    # EQ
    st.markdown("**EQ**")
    eq = effects.get("eq", {"enabled": False, "low_gain": 0, "mid_gain": 0, "high_gain": 0})
    col1, col2 = st.columns([1, 3])
    with col1:
        eq_enabled = st.checkbox("Enable", eq.get("enabled", False), key=f"{key_prefix}_eq_enabled")
    with col2:
        if eq_enabled:
            low_gain = st.slider("Low", -10, 10, eq.get("low_gain", 0), 1, key=f"{key_prefix}_eq_low")
            mid_gain = st.slider("Mid", -10, 10, eq.get("mid_gain", 0), 1, key=f"{key_prefix}_eq_mid")
            high_gain = st.slider("High", -10, 10, eq.get("high_gain", 0), 1, key=f"{key_prefix}_eq_high")
        else:
            low_gain, mid_gain, high_gain = 0, 0, 0
    updated_segment.setdefault("effects", {})["eq"] = {
        "enabled": eq_enabled,
        "low_gain": low_gain,
        "mid_gain": mid_gain,
        "high_gain": high_gain
    }
    
    # Compression
    st.markdown("**Compression**")
    compression = effects.get("compression", {"enabled": False, "threshold": -20, "ratio": 4})
    col1, col2 = st.columns([1, 3])
    with col1:
        comp_enabled = st.checkbox("Enable", compression.get("enabled", False), key=f"{key_prefix}_comp_enabled")
    with col2:
        if comp_enabled:
            threshold = st.slider("Threshold", -60, 0, compression.get("threshold", -20), 1, key=f"{key_prefix}_comp_threshold")
            ratio = st.slider("Ratio", 1, 20, compression.get("ratio", 4), 1, key=f"{key_prefix}_comp_ratio")
        else:
            threshold, ratio = -20, 4
    updated_segment.setdefault("effects", {})["compression"] = {
        "enabled": comp_enabled,
        "threshold": threshold,
        "ratio": ratio
    }
    
    return updated_segment

# ------------------- Main Application -------------------

def main():
    st.set_page_config(
        page_title="AI Voice & Music Production Studio",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .download-button {
            display: inline-block;
            padding: 0.5em 1em;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 1em;
        }
        .stProgress .st-bo {
            background-color: #4CAF50;
        }
        div.block-container {padding-top: 1rem;}
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color:white;
            font-weight:bold;
        }
        h1, h2, h3 {
            margin-bottom: 0.5rem;
        }
        .audio-segment {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
        }
        .sidebar-content {
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "project_data" not in st.session_state:
        st.session_state.project_data = {
            "segments": {},
            "settings": {
                "sample_rate": SAMPLE_RATE,
            },
            "mix_path": None
        }
    
    if "current_project" not in st.session_state:
        st.session_state.current_project = ""
    
    # Sidebar for project management, loading models
    with st.sidebar:
        st.title("🎵 AI Voice & Music Mixer")
        
        # Project Management
        st.header("Project")
        
        project_action = st.radio("Project Action", 
                                ["New Project", "Save Project", "Load Project"],
                                horizontal=True)
        
        if project_action == "New Project":
            if st.button("Create New Project"):
                st.session_state.project_data = {
                    "segments": {},
                    "settings": {
                        "sample_rate": SAMPLE_RATE,
                    },
                    "mix_path": None
                }
                st.session_state.current_project = ""
                st.success("New project created")
        
        elif project_action == "Save Project":
            project_name = st.text_input("Project Name", 
                                        value=st.session_state.current_project or f"project_{int(time.time())}")
            if st.button("Save"):
                if save_project(st.session_state.project_data, project_name):
                    st.session_state.current_project = project_name
                    st.success(f"Project saved as: {project_name}")
        
        elif project_action == "Load Project":
            projects = list_projects()
            if projects:
                selected_project = st.selectbox("Select Project", projects)
                if st.button("Load"):
                    project_data = load_project(selected_project)
                    if project_data:
                        st.session_state.project_data = project_data
                        st.session_state.current_project = selected_project
                        st.success(f"Project loaded: {selected_project}")
            else:
                st.info("No saved projects found")
        
        # Model Loading
        st.header("AI Models")
        model_size = st.radio("MusicGen Model Size", 
                             ["small", "medium", "melody"], 
                             horizontal=True,
                             help="Larger models are more capable but slower")
        
        load_models = st.button("Load/Reload AI Models")
        
        if load_models or "musicgen_model" not in st.session_state:
            with st.spinner("Loading MusicGen model..."):
                st.session_state.musicgen_model = load_musicgen_model(model_size)
                
            with st.spinner("Loading AudioGen model..."):
                st.session_state.audiogen_model = load_audiogen_model()
            
            st.success("Models loaded successfully")
    
    # Main UI
    st.title("AI Voice & Music Production Studio")
    
    if st.session_state.current_project:
        st.subheader(f"Current Project: {st.session_state.current_project}")
    
    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["Add Content", "Mix & Edit", "Export", "Settings"])
    
    # Tab 1: Add Content
    with tab1:
        st.header("Generate Audio Content")
        
        content_type = st.radio("Content Type", 
                               ["Speech", "Music", "Sound Effects"],
                               horizontal=True)
        
        if content_type == "Speech":
            st.subheader("Text-to-Speech")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                speech_text = st.text_area("Text for Speech", height=150,
                                         placeholder="Enter the text you want to convert to speech...")
            with col2:
                language = st.selectbox("Language", ["en", "fr", "es", "de", "it", "ja"])
                voice_provider = st.selectbox("Voice Provider", ["gtts"])
            
            filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
            
            if st.button("Generate Speech"):
                with st.spinner("Generating speech..."):
                    speech_path, speech_waveform, speech_sr = generate_speech(
                        speech_text, language, voice_provider, filename
                    )
                    
                    if speech_path:
                        # Add to project data
                        segment_id = f"speech_{int(time.time())}"
                        st.session_state.project_data["segments"][segment_id] = {
                            "type": "speech",
                            "path": speech_path,
                            "waveform": speech_waveform,
                            "sample_rate": speech_sr,
                            "enabled": True,
                            "volume": 1.0,
                            "pan": 0.0,
                            "effects": {},
                            "metadata": {
                                "text": speech_text,
                                "language": language
                            }
                        }
                        
                        st.success("Speech generated and added to project")
                        st.audio(speech_path, format="audio/wav")
        
        elif content_type == "Music":
            st.subheader("AI Music Generation")
            
            if "musicgen_model" not in st.session_state:
                st.warning("MusicGen model not loaded. Please load models from the sidebar first.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    music_prompt = st.text_area("Music Description", height=150,
                                               placeholder="Describe the music you want to generate...")
                with col2:
                    music_duration = st.number_input("Duration (seconds)", 
                                                   min_value=1, max_value=30, value=5)
                
                filename = f"music_{uuid.uuid4().hex[:8]}.wav"
                
                if st.button("Generate Music"):
                    with st.spinner("Generating music... This may take a while"):
                        music_path, music_waveform, music_sr = generate_audio_segment(
                            "music",
                            st.session_state.musicgen_model,
                            music_prompt,
                            music_duration,
                            filename
                        )
                        
                        if music_path:
                            # Add to project data
                            segment_id = f"music_{int(time.time())}"
                            st.session_state.project_data["segments"][segment_id] = {
                                "type": "music",
                                "path": music_path,
                                "waveform": music_waveform,
                                "sample_rate": music_sr,
                                "enabled": True,
                                "volume": 1.0,
                                "pan": 0.0,
                                "loop": False,
                                "effects": {},
                                "metadata": {
                                    "prompt": music_prompt,
                                    "duration": music_duration
                                }
                            }
                            
                            st.success("Music generated and added to project")
                            st.audio(music_path, format="audio/wav")
        
        elif content_type == "Sound Effects":
            st.subheader("AI Sound Effects Generation")
            
            if "audiogen_model" not in st.session_state:
                st.warning("AudioGen model not loaded. Please load models from the sidebar first.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    sfx_prompt = st.text_area("Sound Effect Description", height=150,
                                            placeholder="Describe the sound effect you want to generate...")
                with col2:
                    sfx_duration = st.number_input("Duration (seconds)", 
                                                 min_value=1, max_value=10, value=3)
                
                filename = f"sfx_{uuid.uuid4().hex[:8]}.wav"
                
                if st.button("Generate Sound Effect"):
                    with st.spinner("Generating sound effect..."):
                        sfx_path, sfx_waveform, sfx_sr = generate_audio_segment(
                            "sfx",
                            st.session_state.audiogen_model,
                            sfx_prompt,
                            sfx_duration,
                            filename
                        )
                        
                        if sfx_path:
                            # Add to project data
                            segment_id = f"sfx_{int(time.time())}"
                            st.session_state.project_data["segments"][segment_id] = {
                                "type": "sfx",
                                "path": sfx_path,
                                "waveform": sfx_waveform,
                                "sample_rate": sfx_sr,
                                "enabled": True,
                                "volume": 1.0,
                                "pan": 0.0,
                                "loop": False,
                                "effects": {},
                                "metadata": {
                                    "prompt": sfx_prompt,
                                    "duration": sfx_duration
                                }
                            }
                            
                            st.success("Sound effect generated and added to project")
                            st.audio(sfx_path, format="audio/wav")
        
        # File upload option for existing audio
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            # Save the uploaded file
            file_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load audio data
            try:
                waveform, sample_rate = sf.read(file_path)
                
                # Add to project data
                segment_id = f"upload_{int(time.time())}"
                st.session_state.project_data["segments"][segment_id] = {
                    "type": "upload",
                    "path": file_path,
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "enabled": True,
                    "volume": 1.0,
                    "pan": 0.0,
                    "loop": False,
                    "effects": {},
                    "metadata": {
                        "filename": uploaded_file.name
                    }
                }
                
                st.success(f"File uploaded: {uploaded_file.name}")
                st.audio(file_path, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
    
    # Tab 2: Mix & Edit
    
    with tab2:
        st.header("Mix Audio Segments")
    
        segments = st.session_state.project_data.get("segments", {})
        if not segments:
            st.info("No audio segments added yet. Go to the 'Add Content' tab to create or upload audio.")
        else:
            # Display each segment with controls - FIX: Use containers instead of nested expanders
            for segment_id, segment in list(segments.items()):
                # Replace expander with a container and add visual separation
               st.markdown(f"### {segment.get('type', 'Audio').title()}: {segment_id}")
               segment_container = st.container()
                # Add visual separation with a light gray line
               st.markdown("---")
            
               with segment_container:
                st.session_state.project_data["segments"][segment_id] = show_audio_segment_controls(
                    segment, segment_id
                )
                
                # Delete button
                if st.button("Delete Segment", key=f"delete_{segment_id}"):
                    del st.session_state.project_data["segments"][segment_id]
                    st.experimental_rerun()
        
        # Mix button
        col1, col2 = st.columns([3, 1])
        with col1:
            mix_filename = st.text_input("Mix Filename", value="final_mix.wav")
        with col2:
            mix_button = st.button("Mix Audio", key="mix_audio")
            
            if mix_button:
                with st.spinner("Mixing audio..."):
                    segments_data = list(st.session_state.project_data["segments"].values())
                    mix_path = mix_audio_segments(
                        segments_data,
                        st.session_state.project_data["settings"]["sample_rate"],
                        mix_filename
                    )
                    
                    if mix_path:
                        st.session_state.project_data["mix_path"] = mix_path
                        st.success("Audio mixed successfully")
                        st.audio(mix_path, format="audio/wav")
                        
                        # Spectrogram visualization
                        waveform, sr = sf.read(mix_path)
                        st.subheader("Mix Visualization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            waveform_b64 = create_waveform_plot(waveform, sr)
                            if waveform_b64:
                                st.image(f"data:image/png;base64,{waveform_b64}", use_column_width=True)
                                st.caption("Waveform")
                        
                        with col2:
                            spectrogram_b64 = create_spectrogram(waveform, sr)
                            if spectrogram_b64:
                                st.image(f"data:image/png;base64,{spectrogram_b64}", use_column_width=True)
                                st.caption("Spectrogram")
    
    # Tab 3: Export
    with tab3:
        st.header("Export Audio")
        
        if not st.session_state.project_data.get("mix_path"):
            st.info("No mix available yet. Go to the 'Mix & Edit' tab to mix your audio segments.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_format = st.selectbox("Export Format", ["WAV", "MP3", "OGG"])
            
            with col2:
                quality = st.select_slider(
                    "Quality",
                    options=["Low", "Medium", "High"],
                    value="High"
                )
            
            with col3:
                if st.button("Export"):
                    with st.spinner(f"Exporting to {export_format}..."):
                        export_path = export_audio(
                            st.session_state.project_data["mix_path"],
                            export_format.lower(),
                            quality.lower()
                        )
                        
                        if export_path:
                            st.success(f"Audio exported to {export_format}")
                            st.audio(export_path, format=f"audio/{export_format.lower()}")
                            
                            # Download link
                            download_link = get_binary_file_downloader_html(
                                export_path, f"{export_format} File"
                            )
                            if download_link:
                                st.markdown(download_link, unsafe_allow_html=True)
    
    # Tab 4: Settings
    with tab4:
        st.header("Settings")
        
        # Sample rate settings
        sample_rate = st.selectbox(
            "Sample Rate",
            [22050, 44100, 48000],
            index=1,
            help="Higher sample rates provide better quality but larger file sizes"
        )
        
        st.session_state.project_data["settings"]["sample_rate"] = sample_rate
        
        # Additional settings
        st.subheader("Advanced Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            default_effects = st.checkbox(
                "Apply Default Effects to New Content", 
                value=st.session_state.project_data.get("settings", {}).get("default_effects", False),
                help="Automatically apply a set of default effects to newly created content"
            )
            st.session_state.project_data["settings"]["default_effects"] = default_effects
            
            auto_normalize = st.checkbox(
                "Auto-Normalize Mixes", 
                value=st.session_state.project_data.get("settings", {}).get("auto_normalize", True),
                help="Automatically normalize the final mix to prevent clipping"
            )
            st.session_state.project_data["settings"]["auto_normalize"] = auto_normalize
        
        with col2:
            temp_files_expiry = st.number_input(
                "Temporary Files Expiry (days)",
                min_value=1,
                max_value=30,
                value=st.session_state.project_data.get("settings", {}).get("temp_files_expiry", 7),
                help="Number of days to keep temporary files before cleanup"
            )
            st.session_state.project_data["settings"]["temp_files_expiry"] = temp_files_expiry
        
        # System performance
        st.subheader("System Performance")
        
        use_gpu = st.checkbox(
            "Use GPU (if available)", 
            value=st.session_state.project_data.get("settings", {}).get("use_gpu", torch.cuda.is_available()),
            help="Use GPU acceleration for faster processing if available"
        )
        st.session_state.project_data["settings"]["use_gpu"] = use_gpu
        
        processing_threads = st.slider(
            "Processing Threads", 
            min_value=1,
            max_value=max(8, os.cpu_count() or 4),
            value=st.session_state.project_data.get("settings", {}).get("processing_threads", max(2, (os.cpu_count() or 4) // 2)),
            help="Number of processing threads for audio operations"
        )
        st.session_state.project_data["settings"]["processing_threads"] = processing_threads
        
        # Clean temp files
        if st.button("Clean Temporary Files"):
            try:
                files_removed = 0
                current_time = time.time()
                for filename in os.listdir(OUTPUT_DIR):
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    # Skip non-files and project output files
                    if not os.path.isfile(file_path) or filename.startswith("final_mix"):
                        continue
                    
                    # Check if older than 7 days
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > (7 * 24 * 60 * 60):  # 7 days in seconds
                        os.remove(file_path)
                        files_removed += 1
                
                st.success(f"Cleaned up {files_removed} temporary files")
            except Exception as e:
                st.error(f"Error cleaning temporary files: {str(e)}")
        
        # App info
        st.subheader("About")
        st.markdown("""
        **AI Voice & Music Production Studio**
        
        This application allows you to generate and mix AI-powered speech, music, and sound effects.
        
        - Use the **Add Content** tab to generate or upload audio
        - Use the **Mix & Edit** tab to combine and process audio segments
        - Use the **Export** tab to save your final mix in various formats
        
        Built with Streamlit, Torchaudio, Librosa, AudioCraft, and more.
        """)

# ------------------- Background Task Worker -------------------

def background_worker():
    """Background worker to process audio tasks."""
    logger.info("Starting background worker thread")
    
    while True:
        try:
            task = processing_queue.get(timeout=1.0)
            if task is None:  # Sentinel to stop the thread
                break
                
            # Process task
            task_type = task.get("type")
            
            if task_type == "generate_audio":
                generate_audio_segment(
                    task["model_type"],
                    task["model"],
                    task["prompt"],
                    task["duration"],
                    task["filename"]
                )
            elif task_type == "mix_audio":
                mix_audio_segments(
                    task["segments"],
                    task["sample_rate"],
                    task["filename"]
                )
            
            # Mark task as done
            processing_queue.task_done()
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error in background worker: {e}")
    
    logger.info("Background worker thread stopped")

# ------------------- Entry Point -------------------

if __name__ == "__main__":
    # Start background worker thread
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        # Stop background worker
        processing_queue.put(None)
        worker_thread.join(timeout=5.0)
