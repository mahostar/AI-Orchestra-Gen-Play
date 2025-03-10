import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
import pygame
import threading
import tempfile
import os
import re
import time
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.effects import normalize, compress_dynamic_range
from mido import MidiFile, MidiTrack, Message, MetaMessage
import subprocess
import sys
import wave
import struct
import array
import math
from collections import deque
import random

# Import audio effects libraries if available
try:
    import sounddevice as sd
    import scipy.signal as signal
    from pedalboard import Pedalboard, Reverb, Compressor, Gain, Chorus, Delay, Limiter, LadderFilter, Phaser
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

class AudioEffects:
    """Class to handle audio effects processing"""
    
    def __init__(self):
        # Default effect parameters
        self.reverb_enabled = False
        self.reverb_amount = 0.3  # 0-1
        self.reverb_room_size = 0.5  # 0-1
        
        self.compression_enabled = False
        self.compression_threshold = -20  # dB
        self.compression_ratio = 4.0
        
        self.eq_enabled = False
        self.eq_low = 0.0  # -12 to 12 dB
        self.eq_mid = 0.0
        self.eq_high = 0.0
        
        self.chorus_enabled = False
        self.chorus_rate = 1.0
        self.chorus_depth = 0.25
        
        self.delay_enabled = False
        self.delay_time = 0.5
        self.delay_feedback = 0.3
        
        self.width_enabled = False
        self.width_amount = 50  # 0-100
        
        self.limiter_enabled = True
        self.gain = 0.0  # dB
        
        # Initialize pedalboard if available
        self.init_pedalboard()
    
    def init_pedalboard(self):
        """Initialize the audio effects chain using Pedalboard if available"""
        if PEDALBOARD_AVAILABLE:
            self.pedalboard = Pedalboard([
                Gain(self.gain),
                Chorus(rate_hz=self.chorus_rate, depth=self.chorus_depth, mix=0.0),
                Reverb(room_size=self.reverb_room_size, damping=0.5, wet_level=0.0, dry_level=1.0),
                Delay(delay_seconds=self.delay_time, feedback=self.delay_feedback, mix=0.0),
                Compressor(threshold_db=self.compression_threshold, ratio=self.compression_ratio, attack_ms=1.0, release_ms=100.0),
                Phaser(rate_hz=1.0, depth=0.5, feedback=0.0, mix=0.0),
                LadderFilter(cutoff_hz=1000, resonance=0.1, drive=1.0),
                Limiter(threshold_db=-1.5, release_ms=100)
            ])
        
    def update_pedalboard(self):
        """Update the pedalboard with current effect settings"""
        if not PEDALBOARD_AVAILABLE:
            return
        
        # Update effects based on enabled state and parameter values
        self.pedalboard = Pedalboard([])
        
        # Always add gain first
        self.pedalboard.append(Gain(self.gain))
        
        # Add enabled effects
        if self.chorus_enabled:
            self.pedalboard.append(Chorus(
                rate_hz=self.chorus_rate,
                depth=self.chorus_depth,
                mix=0.5
            ))
        
        if self.reverb_enabled:
            self.pedalboard.append(Reverb(
                room_size=self.reverb_room_size,
                damping=0.5,
                wet_level=self.reverb_amount,
                dry_level=1.0 - self.reverb_amount
            ))
        
        if self.delay_enabled:
            self.pedalboard.append(Delay(
                delay_seconds=self.delay_time,
                feedback=self.delay_feedback,
                mix=0.3
            ))
        
        if self.compression_enabled:
            self.pedalboard.append(Compressor(
                threshold_db=self.compression_threshold,
                ratio=self.compression_ratio,
                attack_ms=1.0,
                release_ms=100.0
            ))
        
        # Always add limiter at the end to prevent clipping
        if self.limiter_enabled:
            self.pedalboard.append(Limiter(threshold_db=-1.5, release_ms=100))
    
    def process_audio(self, audio_data, sample_rate=44100):
        """Process audio data through the effects chain"""
        if not PEDALBOARD_AVAILABLE:
            return audio_data
        
        # Ensure the pedalboard is up to date
        self.update_pedalboard()
        
        # Process the audio
        processed_audio = self.pedalboard.process(
            np.array(audio_data, dtype=np.float32),
            sample_rate=sample_rate
        )
        
        return processed_audio
    
    def process_wav_file(self, input_file, output_file):
        """Process a WAV file with the current effects chain"""
        if not PEDALBOARD_AVAILABLE:
            # Just copy the file if we can't process it
            with open(input_file, 'rb') as in_f:
                with open(output_file, 'wb') as out_f:
                    out_f.write(in_f.read())
            return
        
        # Ensure the pedalboard is up to date
        self.update_pedalboard()
        
        try:
            # Open the WAV file
            with wave.open(input_file, 'rb') as wav_file:
                # Get basic info
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read all frames
                frames = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 2:  # 16-bit audio
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit audio
                    dtype = np.int32
                else:
                    dtype = np.int8
                
                audio_data = np.frombuffer(frames, dtype=dtype)
                
                # Convert to float32 for processing
                float_data = audio_data.astype(np.float32) / (2**(8*sample_width-1))
                
                # Reshape if stereo
                if n_channels == 2:
                    float_data = float_data.reshape(-1, 2)
                
                # Process the audio
                processed_audio = self.pedalboard.process(float_data, sample_rate=sample_rate)
                
                # Convert back to original format
                processed_int = (processed_audio * (2**(8*sample_width-1))).astype(dtype)
                
                # Write the processed audio to the output file
                with wave.open(output_file, 'wb') as out_wav:
                    out_wav.setnchannels(n_channels)
                    out_wav.setsampwidth(sample_width)
                    out_wav.setframerate(sample_rate)
                    out_wav.writeframes(processed_int.tobytes())
        
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            # Fall back to copying the file
            with open(input_file, 'rb') as in_f:
                with open(output_file, 'wb') as out_f:
                    out_f.write(in_f.read())

class ABCMusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("ABC Orchestra Player")
        self.root.minsize(1000, 600)  # Set minimum window size
        self.root.configure(bg="#f0f0f0")
        
        # Create main container frame
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for content
        self.canvas = tk.Canvas(self.main_container, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack main elements
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel to scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Bind canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        
        # Create a temporary directory with random name inside project directory
        project_dir = "E:\\projects\\music GEN"
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
        self.temp_dir = os.path.join(project_dir, f"temp_{random_suffix}")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            print(f"Created temporary directory: {self.temp_dir}")
        
        # List of temporary files we create
        self.temp_files = [
            "temp.abc",           # ABC notation file
            "temp.mid",           # MIDI file
            "temp.wav",           # WAV file
            "export_temp.wav",    # Temporary WAV for export
            "temp_processed.wav", # Processed WAV file
            "export_temp.mid"     # Temporary MIDI for export
        ]
        
        # Initialize audio effects
        self.audio_effects = AudioEffects()
        
        # Dictionary of instruments and their corresponding MIDI program numbers
        self.instruments = {
            "Piano": 0,
            "Acoustic Grand Piano": 0,
            "Bright Acoustic Piano": 1,
            "Electric Grand Piano": 2,
            "Honky-tonk Piano": 3,
            "Electric Piano 1": 4,
            "Electric Piano 2": 5,
            "Harpsichord": 6,
            "Clavinet": 7,
            "Celesta": 8,
            "Glockenspiel": 9,
            "Music Box": 10,
            "Vibraphone": 11,
            "Marimba": 12,
            "Xylophone": 13,
            "Tubular Bells": 14,
            "Dulcimer": 15,
            "Drawbar Organ": 16,
            "Percussive Organ": 17,
            "Rock Organ": 18,
            "Church Organ": 19,
            "Reed Organ": 20,
            "Accordion": 21,
            "Harmonica": 22,
            "Tango Accordion": 23,
            "Acoustic Guitar (nylon)": 24,
            "Acoustic Guitar (steel)": 25,
            "Electric Guitar (jazz)": 26,
            "Electric Guitar (clean)": 27,
            "Electric Guitar (muted)": 28,
            "Overdriven Guitar": 29,
            "Distortion Guitar": 30,
            "Guitar Harmonics": 31,
            "Acoustic Bass": 32,
            "Electric Bass (finger)": 33,
            "Electric Bass (pick)": 34,
            "Fretless Bass": 35,
            "Slap Bass 1": 36,
            "Slap Bass 2": 37,
            "Synth Bass 1": 38,
            "Synth Bass 2": 39,
            "Violin": 40,
            "Viola": 41,
            "Cello": 42,
            "Contrabass": 43,
            "Tremolo Strings": 44,
            "Pizzicato Strings": 45,
            "Orchestral Harp": 46,
            "Timpani": 47,
            "String Ensemble 1": 48,
            "String Ensemble 2": 49,
            "Synth Strings 1": 50,
            "Synth Strings 2": 51,
            "Choir Aahs": 52,
            "Voice Oohs": 53,
            "Synth Voice": 54,
            "Orchestra Hit": 55,
            "Trumpet": 56,
            "Trombone": 57,
            "Tuba": 58,
            "Muted Trumpet": 59,
            "French Horn": 60,
            "Brass Section": 61,
            "Synth Brass 1": 62,
            "Synth Brass 2": 63,
            "Soprano Sax": 64,
            "Alto Sax": 65,
            "Tenor Sax": 66,
            "Baritone Sax": 67,
            "Oboe": 68,
            "English Horn": 69,
            "Bassoon": 70,
            "Clarinet": 71,
            "Piccolo": 72,
            "Flute": 73,
            "Recorder": 74,
            "Pan Flute": 75,
            "Blown Bottle": 76,
            "Shakuhachi": 77,
            "Whistle": 78,
            "Ocarina": 79,
            "Lead 1 (square)": 80,
            "Lead 2 (sawtooth)": 81,
            "Lead 3 (calliope)": 82,
            "Lead 4 (chiff)": 83,
            "Lead 5 (charang)": 84,
            "Lead 6 (voice)": 85,
            "Lead 7 (fifths)": 86,
            "Lead 8 (bass + lead)": 87,
            "Pad 1 (new age)": 88,
            "Pad 2 (warm)": 89,
            "Pad 3 (polysynth)": 90,
            "Pad 4 (choir)": 91,
            "Pad 5 (bowed)": 92,
            "Pad 6 (metallic)": 93,
            "Pad 7 (halo)": 94,
            "Pad 8 (sweep)": 95,
            "FX 1 (rain)": 96,
            "FX 2 (soundtrack)": 97,
            "FX 3 (crystal)": 98,
            "FX 4 (atmosphere)": 99,
            "FX 5 (brightness)": 100,
            "FX 6 (goblins)": 101,
            "FX 7 (echoes)": 102,
            "FX 8 (sci-fi)": 103,
            "Sitar": 104,
            "Banjo": 105,
            "Shamisen": 106,
            "Koto": 107,
            "Kalimba": 108,
            "Bagpipe": 109,
            "Fiddle": 110,
            "Shanai": 111,
            "Tinkle Bell": 112,
            "Agogo": 113,
            "Steel Drums": 114,
            "Woodblock": 115,
            "Taiko Drum": 116,
            "Melodic Tom": 117,
            "Synth Drum": 118,
            "Reverse Cymbal": 119,
            "Guitar Fret Noise": 120,
            "Breath Noise": 121,
            "Seashore": 122,
            "Bird Tweet": 123,
            "Telephone Ring": 124,
            "Helicopter": 125,
            "Applause": 126,
            "Gunshot": 127
        }
        
        # Instrument name mapping for automatic selection
        self.instrument_name_mapping = {
            "violin": "Violin",
            "viola": "Viola",
            "cello": "Cello",
            "violoncello": "Cello",
            "bass": "Contrabass",
            "contrabass": "Contrabass",
            "double bass": "Contrabass",
            "piano": "Piano",
            "flute": "Flute",
            "oboe": "Oboe",
            "clarinet": "Clarinet",
            "bassoon": "Bassoon",
            "trumpet": "Trumpet",
            "horn": "French Horn",
            "french horn": "French Horn",
            "trombone": "Trombone",
            "tuba": "Tuba",
            "timpani": "Timpani",
            "percussion": "Woodblock",
            "harp": "Orchestral Harp",
            "guitar": "Acoustic Guitar (nylon)",
            "organ": "Church Organ",
            "choir": "Choir Aahs",
            "voice": "Voice Oohs",
            "soprano": "Voice Oohs",
            "alto": "Voice Oohs",
            "tenor": "Voice Oohs",
            "bass voice": "Voice Oohs",
            "saxophone": "Tenor Sax",
            "sax": "Tenor Sax"
        }
        
        # Default voice-to-instrument mapping
        self.default_voice_instruments = {
            "1": "Piano",
            "2": "Acoustic Bass",
            "3": "Violin",
            "4": "Cello"
        }
        
        # The currently playing process
        self.current_process = None
        
        # Flag to check if music is playing
        self.is_playing = False
        
        # Voice combos dictionary
        self.voice_combos = {}
        
        # Voice frames dictionary for dynamic UI
        self.voice_frames = {}
        
        # Create the interface
        self.create_widgets()
        
        # Bind paste event to the ABC input text area
        self.abc_input.bind('<<Paste>>', self.on_paste)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        # Update the width of the frame to fill canvas
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.scrollable_frame, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create styles for different font sizes
        style = ttk.Style()
        style.configure('Title.TLabel', font=("Arial", 24, "bold"))
        style.configure('Header.TLabelframe.Label', font=("Arial", 12, "bold"))
        style.configure('Regular.TLabel', font=("Arial", 11))
        style.configure('Large.TButton', font=("Arial", 13), padding=(15, 10))
        style.configure('Regular.TCheckbutton', font=("Arial", 11))
        style.configure('Large.TCombobox', padding=(5, 8))
        
        # Create the editor contents directly in the main frame
        self.create_editor_content(main_frame)
        
        # Create bottom frame for effects and status bar
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add effects UI to bottom frame
        self.add_effects_ui(bottom_frame)
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Adjust window size after creating widgets
        self.root.update_idletasks()
        self.adjust_window_size()

    def create_editor_content(self, parent):
        """Create widgets for the main window"""
        # Title label with larger, bold font
        title_label = ttk.Label(parent, text="ABC Orchestra Player", style='Title.TLabel')
        title_label.pack(pady=15)
        
        # Top section: ABC input with bold title
        input_frame = ttk.LabelFrame(parent, text="ABC Notation Input", padding=10, style='Header.TLabelframe')
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ABC input text area with larger monospace font
        self.abc_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=80, height=15, 
                                                 font=("Consolas", 14))  # Changed to Consolas and larger size
        self.abc_input.pack(fill=tk.BOTH, expand=True)
        
        # Bind text change events
        self.abc_input.bind('<<Modified>>', self.on_text_change)
        self.abc_input.bind('<<Paste>>', self.on_paste)
        
        # Buttons frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Load file button
        load_button = ttk.Button(button_frame, text="Load ABC File", command=self.load_abc_file, style='Large.TButton')
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Save file button
        save_button = ttk.Button(button_frame, text="Save ABC to File", command=self.save_abc_file, style='Large.TButton')
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Play button
        self.play_button = ttk.Button(button_frame, text="▶ Play", command=self.play_abc, style='Large.TButton')
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        stop_button = ttk.Button(button_frame, text="■ Stop", command=self.stop_playing, style='Large.TButton')
        stop_button.pack(side=tk.LEFT, padx=5)
        
        # Export to MIDI button
        export_midi_button = ttk.Button(button_frame, text="Export to MIDI", command=self.export_to_midi, style='Large.TButton')
        export_midi_button.pack(side=tk.LEFT, padx=5)
        
        # Export to WAV button
        export_wav_button = ttk.Button(button_frame, text="Export to WAV", command=self.export_to_wav, style='Large.TButton')
        export_wav_button.pack(side=tk.LEFT, padx=5)
        
        # Voice to Instrument Mapping with bold title
        self.mapping_frame = ttk.LabelFrame(parent, text="Voice to Instrument Mapping", padding=10, style='Header.TLabelframe')
        self.mapping_frame.pack(fill=tk.BOTH, pady=5)
        
        # Create a container for voice frames that can be dynamically updated
        self.voices_container = ttk.Frame(self.mapping_frame)
        self.voices_container.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with default 4 voices
        self.update_voice_mapping(4)
    
    def create_editor_tab(self, parent):
        """This function is no longer needed - remove it entirely"""
        pass
    
    def update_effect(self, effect_name, enabled):
        """Update effect enabled state"""
        if effect_name == "reverb":
            self.audio_effects.reverb_enabled = enabled
        elif effect_name == "compression":
            self.audio_effects.compression_enabled = enabled
        elif effect_name == "delay":
            self.audio_effects.delay_enabled = enabled
        
        self.audio_effects.update_pedalboard()
        self.status_var.set(f"{effect_name.capitalize()} {'enabled' if enabled else 'disabled'}")
    
    def apply_preset(self, preset_name):
        """Apply an audio effects preset"""
        # Default state - all effects off
        self.audio_effects.reverb_enabled = False
        self.audio_effects.compression_enabled = False
        self.audio_effects.chorus_enabled = False
        self.audio_effects.delay_enabled = False
        
        # Apply preset settings
        if preset_name == "Concert Hall":
            self.audio_effects.reverb_enabled = True
            self.audio_effects.reverb_amount = 0.5
            self.audio_effects.reverb_room_size = 0.8
        elif preset_name == "Warm Studio":
            self.audio_effects.reverb_enabled = True
            self.audio_effects.reverb_amount = 0.2
            self.audio_effects.reverb_room_size = 0.4
            self.audio_effects.compression_enabled = True
            self.audio_effects.compression_threshold = -18
            self.audio_effects.compression_ratio = 3.0
        elif preset_name == "Bright & Clear":
            self.audio_effects.reverb_enabled = True
            self.audio_effects.reverb_amount = 0.15
            self.audio_effects.reverb_room_size = 0.3
        elif preset_name == "Orchestral":
            self.audio_effects.reverb_enabled = True
            self.audio_effects.reverb_amount = 0.4
            self.audio_effects.reverb_room_size = 0.7
            self.audio_effects.chorus_enabled = True
            self.audio_effects.chorus_rate = 0.5
            self.audio_effects.chorus_depth = 0.15
        elif preset_name == "Deep Bass":
            self.audio_effects.compression_enabled = True
            self.audio_effects.compression_threshold = -20
            self.audio_effects.compression_ratio = 4.0
        elif preset_name == "Vocal Enhancer":
            self.audio_effects.reverb_enabled = True
            self.audio_effects.reverb_amount = 0.25
            self.audio_effects.reverb_room_size = 0.4
            self.audio_effects.compression_enabled = True
            self.audio_effects.compression_threshold = -15
            self.audio_effects.compression_ratio = 2.5
        
        self.audio_effects.update_pedalboard()
        self.status_var.set(f"Applied '{preset_name}' preset")
    
    def update_voice_mapping(self, num_voices):
        """Update the voice mapping UI based on the number of voices"""
        # Clear existing voice frames
        for frame in self.voice_frames.values():
            frame.destroy()
        
        self.voice_frames = {}
        self.voice_combos = {}
        
        # Calculate layout
        columns = min(4, num_voices)  # Max 4 columns
        rows = (num_voices + columns - 1) // columns
        
        # Create new voice frames
        for i in range(1, num_voices + 1):
            row = (i - 1) // columns
            col = (i - 1) % columns
            
            voice_frame = ttk.Frame(self.voices_container)
            voice_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Voice label with larger font
            voice_label = ttk.Label(voice_frame, text=f"Voice {i}:", style='Regular.TLabel')
            voice_label.pack(anchor=tk.W)
            
            # Combo box for instrument selection with larger font
            voice_combo = ttk.Combobox(voice_frame, values=list(self.instruments.keys()), 
                                     state="readonly", font=("Arial", 13), style='Large.TCombobox')
            voice_combo.set(self.default_voice_instruments.get(str(i), "Piano"))
            voice_combo.pack(fill=tk.X, padx=5, pady=2)
            
            self.voice_combos[str(i)] = voice_combo
            self.voice_frames[str(i)] = voice_frame
        
        # Configure grid weights
        for i in range(rows):
            self.voices_container.grid_rowconfigure(i, weight=1)
        for i in range(columns):
            self.voices_container.grid_columnconfigure(i, weight=1)
        
        # Adjust window size after updating voice mapping
        self.root.update_idletasks()
        self.adjust_window_size()
    
    def on_text_change(self, event=None):
        """Handle text changes in the ABC input field"""
        if self.abc_input.edit_modified():  # Check if text was actually modified
            # Get current MIDI file if it exists
            temp_midi_file = os.path.join(self.temp_dir, "temp.mid")
            if os.path.exists(temp_midi_file):
                # Update instruments for the MIDI file
                self.update_midi_instruments(temp_midi_file)
            
            # Reset the modified flag
            self.abc_input.edit_modified(False)
    
    def on_paste(self, event):
        """Handle paste event to analyze ABC header and set instruments"""
        # Schedule this to run after the paste is complete
        self.root.after(10, self.analyze_abc_header)
    
    def analyze_abc_header(self):
        """Analyze the ABC header to determine voices and set appropriate instruments"""
        content = self.abc_input.get(1.0, tk.END)
        
        # Extract voice definitions
        voice_defs = []
        voice_pattern = re.compile(r'V:(\d+)\s+([^\n]*)')
        for match in voice_pattern.finditer(content):
            voice_num = match.group(1)
            voice_def = match.group(2)
            voice_defs.append((voice_num, voice_def))
        
        if not voice_defs:
            return
        
        # Count unique voice numbers
        voice_nums = set(vd[0] for vd in voice_defs)
        num_voices = len(voice_nums)
        
        # Update UI if number of voices changed
        if num_voices != len(self.voice_combos):
            self.update_voice_mapping(num_voices)
        
        # Set instruments based on voice definitions
        for voice_num, voice_def in voice_defs:
            if voice_num in self.voice_combos:
                # Extract instrument name from voice definition
                name_match = re.search(r'nm="([^"]*)"', voice_def)
                if name_match:
                    instrument_name = name_match.group(1).lower()
                    
                    # Find the best matching instrument
                    selected_instrument = None
                    for key in self.instrument_name_mapping:
                        if key in instrument_name:
                            selected_instrument = self.instrument_name_mapping[key]
                            break
                    
                    # Set the instrument in the combo box
                    if selected_instrument and selected_instrument in self.instruments:
                        self.voice_combos[voice_num].set(selected_instrument)
                    else:
                        # Default based on clef or position
                        if "bass" in voice_def.lower():
                            self.voice_combos[voice_num].set("Contrabass")
                        elif "alto" in voice_def.lower():
                            self.voice_combos[voice_num].set("Viola")
                        elif "treble" in voice_def.lower():
                            if int(voice_num) % 2 == 0:  # Even voices
                                self.voice_combos[voice_num].set("Violin")
                            else:  # Odd voices
                                self.voice_combos[voice_num].set("Flute")
        
        self.status_var.set(f"Detected {num_voices} voices in ABC notation")
    
    def load_abc_file(self):
        file_path = filedialog.askopenfilename(
            title="Select ABC file",
            filetypes=[("ABC files", "*.abc"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.abc_input.delete(1.0, tk.END)
                    self.abc_input.insert(tk.END, content)
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                # Analyze the header after loading
                self.analyze_abc_header()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")
    
    def save_abc_file(self):
        file_path = filedialog.asksaveasfilename(
            title="Save ABC file",
            defaultextension=".abc",
            filetypes=[("ABC files", "*.abc"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                content = self.abc_input.get(1.0, tk.END)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
    
    def abc_to_midi(self, abc_content, output_file, update_instruments=True):
        """Convert ABC notation to MIDI file"""
        if not self.check_abc2midi_exists():
            messagebox.showerror("Error", "abc2midi not found. Please install abcMIDI package.")
            return False
        
        # Create a temporary ABC file
        temp_abc_file = os.path.join(self.temp_dir, "temp.abc")
        
        try:
            # Write ABC content to temporary file
            with open(temp_abc_file, 'w', encoding='utf-8') as f:
                f.write(abc_content)
            
            # Run abc2midi to convert ABC to MIDI
            process = subprocess.Popen(
                ["abc2midi", temp_abc_file, "-o", output_file],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"abc2midi error: {stderr}")
                messagebox.showerror("Conversion Error", f"Failed to convert ABC to MIDI:\n{stderr}")
                return False
            
            return True
        except Exception as e:
            print(f"Error in abc_to_midi: {str(e)}")
            messagebox.showerror("Error", f"Failed to convert ABC to MIDI: {str(e)}")
            return False
    
    def check_abc2midi_exists(self):
        """Check if abc2midi is installed and in PATH"""
        try:
            # Try to run abc2midi -h to check if it exists
            subprocess.run(['abc2midi', '-h'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           check=False)
            return True
        except:
            return False
    
    def update_midi_instruments(self, midi_file):
        """Update the MIDI file with the selected instruments"""
        try:
            # Load the MIDI file
            mid = MidiFile(midi_file)
            
            # Create a new MIDI file
            new_mid = MidiFile()
            new_mid.ticks_per_beat = mid.ticks_per_beat
            
            # Process each track
            for i, track in enumerate(mid.tracks):
                # Create a new track
                new_track = MidiTrack()
                
                # Voice number (1-based)
                voice_num = str(i)
                
                # Get the selected instrument for this voice
                instrument_name = "Piano"  # Default
                if voice_num in self.voice_combos:
                    instrument_name = self.voice_combos[voice_num].get()
                
                # Get the MIDI program number for the instrument
                program_num = self.instruments.get(instrument_name, 0)
                
                # Add program change message at the beginning
                new_track.append(Message('program_change', program=program_num, time=0, channel=i % 16))
                
                # Copy all other messages
                for msg in track:
                    # Skip existing program change messages
                    if msg.type == 'program_change':
                        continue
                    new_track.append(msg)
                
                # Add the track to the new MIDI file
                new_mid.tracks.append(new_track)
            
            # Save the new MIDI file
            new_mid.save(midi_file)
            return True
        except Exception as e:
            print(f"Error updating MIDI instruments: {str(e)}")
            return False
    
    def play_abc(self):
        """Play the ABC notation using abc2midi and pygame"""
        if self.is_playing:
            self.pause_resume()
            return
        
        # Get ABC content from the text area
        abc_content = self.abc_input.get(1.0, tk.END)
        if not abc_content.strip():
            messagebox.showinfo("Info", "Please enter ABC notation first.")
            return
        
        # Create a thread to handle conversion and playback
        self.play_thread = threading.Thread(target=self.convert_and_play, args=(abc_content,))
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def convert_and_play(self, abc_content):
        """Convert ABC to MIDI and play it with effects (runs in a separate thread)"""
        self.status_var.set("Converting ABC to MIDI...")
        
        # Create a temporary MIDI file
        temp_midi_file = os.path.join(self.temp_dir, "temp.mid")
        
        # Convert ABC to MIDI
        if not self.abc_to_midi(abc_content, temp_midi_file):
            self.status_var.set("Conversion failed.")
            return
        
        # If pedalboard is available, convert MIDI to WAV first for effects processing
        if PEDALBOARD_AVAILABLE:
            temp_wav_file = os.path.join(self.temp_dir, "temp.wav")
            processed_wav_file = os.path.join(self.temp_dir, "temp_processed.wav")
            
            # Convert MIDI to WAV using FluidSynth if available
            try:
                # Get FluidSynth path (either in PATH or in current directory)
                fluidsynth_path = self.get_fluidsynth_path()
                
                if fluidsynth_path:
                    soundfont = self.get_soundfont_path()
                    if soundfont:
                        self.status_var.set("Rendering high-quality audio...")
                        print(f"Using FluidSynth with soundfont: {soundfont}")
                        
                        # Use FluidSynth to get high-quality audio
                        try:
                            subprocess.run(
                                [fluidsynth_path, "-ni", soundfont, temp_midi_file, "-F", temp_wav_file, "-r", "44100"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True
                            )
                            
                            # Apply effects
                            self.status_var.set("Applying studio effects...")
                            self.audio_effects.process_wav_file(temp_wav_file, processed_wav_file)
                            
                            # Load and play the processed audio
                            try:
                                pygame.mixer.music.load(processed_wav_file)
                                pygame.mixer.music.play()
                                
                                self.is_playing = True
                                self.play_button.config(text="⏸ Pause")
                                self.status_var.set("Playing studio-quality audio...")
                                
                                # Wait for the music to finish
                                while pygame.mixer.music.get_busy() and self.is_playing:
                                    time.sleep(0.1)
                                
                                # If stopped naturally (not by user)
                                if not pygame.mixer.music.get_busy() and self.is_playing:
                                    self.is_playing = False
                                    self.play_button.config(text="▶ Play")
                                    self.status_var.set("Finished playing.")
                                
                                return
                            except Exception as e:
                                print(f"Error playing processed audio: {str(e)}")
                                # Fall back to regular MIDI playback
                        except subprocess.CalledProcessError as e:
                            print("Error running FluidSynth:")
                            print(f"Command returned code {e.returncode}")
                            print(f"Output: {e.stdout}")
                            print(f"Error: {e.stderr}")
                            print("Falling back to regular MIDI playback...")
                        except Exception as e:
                            print(f"Unexpected error running FluidSynth: {str(e)}")
                    else:
                        print("No SoundFont file found for FluidSynth.")
                        print("For high-quality audio, download a SoundFont (.sf2) file.")
                        print("Falling back to regular MIDI playback...")
                else:
                    print("FluidSynth not found in PATH or common locations.")
                    print("For high-quality audio, please install FluidSynth.")
                    print("Falling back to regular MIDI playback...")
            except Exception as e:
                print(f"Error in high-quality rendering: {str(e)}")
                print("Falling back to regular MIDI playback...")
        
        # Load and play the MIDI file (fallback method)
        try:
            pygame.mixer.music.load(temp_midi_file)
            pygame.mixer.music.play()
            
            self.is_playing = True
            self.play_button.config(text="⏸ Pause")
            self.status_var.set("Playing MIDI (basic mode)...")
            
            # Wait for the music to finish
            while pygame.mixer.music.get_busy() and self.is_playing:
                time.sleep(0.1)
            
            # If stopped naturally (not by user)
            if not pygame.mixer.music.get_busy() and self.is_playing:
                self.is_playing = False
                self.play_button.config(text="▶ Play")
                self.status_var.set("Finished playing.")
        
        except Exception as e:
            self.is_playing = False
            self.play_button.config(text="▶ Play")
            self.status_var.set(f"Error playing: {str(e)}")
            print(f"Error playing MIDI: {str(e)}")
    
    def check_fluidsynth(self):
        """Check if FluidSynth is installed"""
        try:
            result = subprocess.run(["fluidsynth", "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           text=True,
                           check=False)
            if result.returncode == 0:
                return True
            else:
                print(f"FluidSynth check failed with return code: {result.returncode}")
                print(f"Error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("FluidSynth not found in PATH. Please add FluidSynth bin directory to your PATH.")
            return False
        except Exception as e:
            print(f"Error checking FluidSynth: {str(e)}")
            return False
    
    def get_fluidsynth_path(self):
        """Get the path to the FluidSynth executable, checking project directory first"""
        # Explicitly use the project directory on E: drive
        project_dir = "E:\\projects\\music GEN"
        project_fluidsynth = os.path.join(project_dir, "important_to_run", "FluidSynth", "bin", "fluidsynth.exe")
        
        if os.path.exists(project_fluidsynth):
            print(f"Found FluidSynth in project directory: {project_fluidsynth}")
            return project_fluidsynth
        
        # Then try the PATH
        try:
            result = subprocess.run(["fluidsynth", "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
            if result.returncode == 0:
                return "fluidsynth"  # It's in the PATH
        except FileNotFoundError:
            pass  # Not in PATH, continue checking
        
        # Check current directory
        current_dir_exe = os.path.join(os.getcwd(), "fluidsynth.exe")
        if os.path.exists(current_dir_exe):
            return current_dir_exe
        
        # Check common installation locations
        common_locations = [
            os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), "FluidSynth", "bin", "fluidsynth.exe"),
            os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), "FluidSynth", "bin", "fluidsynth.exe"),
            os.path.join(os.path.expanduser("~"), "FluidSynth", "bin", "fluidsynth.exe"),
            "C:\\FluidSynth\\bin\\fluidsynth.exe",
            os.path.join(os.getcwd(), "fluidsynth.exe")
        ]
        
        for path in common_locations:
            if os.path.exists(path):
                return path
        
        return None
        
    def get_soundfont_path(self):
        """Get path to a soundfont file, checking project directory first"""
        # Explicitly use the project directory on E: drive
        project_dir = "E:\\projects\\music GEN"
        project_soundfonts_dir = os.path.join(project_dir, "soundfonts")
        
        if os.path.exists(project_soundfonts_dir):
            # Look for .sf2 files in the project soundfonts directory
            for file in os.listdir(project_soundfonts_dir):
                if file.endswith(".sf2"):
                    sf_path = os.path.join(project_soundfonts_dir, file)
                    print(f"Found SoundFont in project directory: {sf_path}")
                    return sf_path
        
        # Check common locations for soundfonts
        common_soundfonts = [
            # User-specific location
            os.path.join(os.path.expanduser("~"), "soundfonts", "FluidR3_GM.sf2"),
            # Windows Program Files
            "C:\\Program Files\\FluidSynth\\share\\soundfonts\\FluidR3_GM.sf2",
            # Linux system locations
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            # macOS location
            "/Library/Audio/Sounds/Banks/FluidR3_GM.sf2",
            # Project directory
            os.path.join(os.getcwd(), "FluidR3_GM.sf2")
        ]
        
        # Check if any of the common soundfonts exist
        for sf_path in common_soundfonts:
            if os.path.exists(sf_path):
                print(f"Found SoundFont: {sf_path}")
                return sf_path
        
        # If no soundfont found, try to find one in the current directory
        sf2_files = []
        for file in os.listdir(os.getcwd()):
            if file.endswith(".sf2"):
                sf2_files.append(file)
                sf_path = os.path.join(os.getcwd(), file)
                print(f"Found SoundFont in current directory: {sf_path}")
                return sf_path
        
        # Print helpful message if no soundfont is found
        print("No SoundFont (.sf2) files found. Please download a SoundFont file.")
        print("Checked the following locations:")
        print(f"  - {project_soundfonts_dir} (project directory)")
        for path in common_soundfonts:
            print(f"  - {path}")
        if not sf2_files:
            print("No .sf2 files found in the current directory.")
        
        return None
    
    def diagnose_fluidsynth(self):
        """Run a comprehensive diagnosis of FluidSynth setup"""
        print("\n=== FluidSynth Diagnostic Report ===")
        
        # Get project directory paths (explicit path)
        project_dir = "E:\\projects\\music GEN"
        project_fluidsynth = os.path.join(project_dir, "important_to_run", "FluidSynth", "bin", "fluidsynth.exe")
        project_soundfonts_dir = os.path.join(project_dir, "soundfonts")
        
        # Check if FluidSynth executable is available
        print("\n1. Checking FluidSynth executable:")
        fluidsynth_exists = False
        fluidsynth_path = self.get_fluidsynth_path()
        
        if fluidsynth_path:
            print(f"✓ FluidSynth found: {fluidsynth_path}")
            # Try to get version info
            try:
                if fluidsynth_path == "fluidsynth":
                    # It's in PATH
                    result = subprocess.run(["fluidsynth", "--version"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          text=True,
                                          check=False)
                else:
                    # Use the full path
                    result = subprocess.run([fluidsynth_path, "--version"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          text=True,
                                          check=False)
                
                if result.returncode == 0:
                    print(f"  Version: {result.stdout.strip()}")
                    fluidsynth_exists = True
            except Exception as e:
                print(f"  Warning: Found FluidSynth but couldn't get version info: {str(e)}")
                fluidsynth_exists = True  # Still mark as found
        else:
            print("✗ FluidSynth executable not found")
            
            # Check project directory specifically
            if os.path.exists(project_fluidsynth):
                print(f"  Warning: FluidSynth exists at {project_fluidsynth} but couldn't be accessed")
            else:
                print(f"  Project directory checked: {os.path.dirname(project_fluidsynth)}")
                print("  FluidSynth not found in project directory")
            
            # Check if in current directory
            if os.path.exists(os.path.join(os.getcwd(), "fluidsynth.exe")):
                print("  Note: fluidsynth.exe was found in your current directory but couldn't be accessed")
        
        # Check for SoundFont files
        print("\n2. Checking for SoundFont files:")
        soundfont_found = False
        
        # First check project directory
        if os.path.exists(project_soundfonts_dir):
            sf2_files = [f for f in os.listdir(project_soundfonts_dir) if f.endswith('.sf2')]
            if sf2_files:
                print(f"✓ Found SoundFont(s) in project directory {project_soundfonts_dir}:")
                for sf in sf2_files:
                    print(f"  - {sf}")
                soundfont_found = True
            else:
                print(f"  Checked project directory {project_soundfonts_dir}: No .sf2 files found")
        else:
            print(f"  Project soundfonts directory does not exist: {project_soundfonts_dir}")
            print("  Consider creating this directory and adding SoundFont files")
        
        # Check other standard locations
        soundfont_paths = [
            os.path.join(os.path.expanduser("~"), "soundfonts"),
            "C:\\Program Files\\FluidSynth\\share\\soundfonts",
            "/usr/share/sounds/sf2",
            "/usr/share/soundfonts",
            "/Library/Audio/Sounds/Banks",
            os.getcwd(),
        ]
        
        for path in soundfont_paths:
            if os.path.exists(path):
                sf2_files = [f for f in os.listdir(path) if f.endswith('.sf2')]
                if sf2_files:
                    print(f"✓ Found SoundFont(s) in {path}:")
                    for sf in sf2_files:
                        print(f"  - {sf}")
                    soundfont_found = True
                else:
                    print(f"  Checked {path}: No .sf2 files found")
            else:
                print(f"  Checked {path}: Directory does not exist")
        
        if not soundfont_found:
            print("✗ No SoundFont (.sf2) files found in any location")
            print("  Please download a SoundFont file (FluidR3_GM.sf2 recommended)")
            print("  From: https://musical-artifacts.com/artifacts/28")
        
        # Check PATH environment variable
        print("\n3. Checking PATH environment variable:")
        path_env = os.environ.get('PATH', '')
        path_dirs = path_env.split(os.pathsep)
        
        fluidsynth_in_path = False
        for directory in path_dirs:
            if os.path.exists(directory):
                if os.path.exists(os.path.join(directory, 'fluidsynth.exe')) or \
                   os.path.exists(os.path.join(directory, 'fluidsynth')):
                    print(f"✓ FluidSynth found in PATH: {directory}")
                    fluidsynth_in_path = True
        
        if not fluidsynth_in_path:
            print("✗ FluidSynth not found in any PATH directory")
            
        # Summary
        print("\n=== Summary ===")
        if fluidsynth_exists and soundfont_found:
            print("✓ FluidSynth appears to be correctly set up!")
        else:
            print("✗ FluidSynth setup is incomplete. Please fix the issues above.")
            
            if not fluidsynth_exists:
                print("\nTo fix FluidSynth executable issues:")
                print(f"1. Create directory: {os.path.dirname(project_fluidsynth)}")
                print("2. Extract the FluidSynth ZIP to this directory")
                print("   OR add the FluidSynth bin directory to your PATH environment variable")
            
            if not soundfont_found:
                print("\nTo fix SoundFont issues:")
                print("1. Download FluidR3_GM.sf2 from https://musical-artifacts.com/artifacts/28")
                print(f"2. Create directory: {project_soundfonts_dir}")
                print("3. Place the .sf2 file in that directory")
        
        print("\n===============================")
        return fluidsynth_exists and soundfont_found
    
    def pause_resume(self):
        """Pause or resume playback"""
        if not self.is_playing:
            return
        
        if pygame.mixer.music.get_busy():
            # Pause
            pygame.mixer.music.pause()
            self.play_button.config(text="▶ Resume")
            self.status_var.set("Paused")
        else:
            # Resume
            pygame.mixer.music.unpause()
            self.play_button.config(text="⏸ Pause")
            self.status_var.set("Playing...")
    
    def stop_playing(self):
        """Stop playback"""
        if self.is_playing:
            pygame.mixer.music.stop()
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        self.status_var.set("Stopped")
    
    def export_to_midi(self):
        """Export ABC notation to a MIDI file"""
        # Get ABC content
        abc_content = self.abc_input.get(1.0, tk.END)
        if not abc_content.strip():
            messagebox.showinfo("Information", "Please enter ABC notation first.")
            return
        
        # Ask for output file location
        file_path = filedialog.asksaveasfilename(
            title="Save MIDI file",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Before conversion, analyze the header to ensure voice mapping is up to date
        self.analyze_abc_header()
        
        # Convert to MIDI
        self.status_var.set("Exporting to MIDI...")
        if self.abc_to_midi(abc_content, file_path):
            self.status_var.set(f"Exported to: {os.path.basename(file_path)}")
        else:
            self.status_var.set("Export failed.")
    
    def export_to_wav(self):
        """Export ABC notation to a WAV file with effects applied"""
        # Get ABC content
        abc_content = self.abc_input.get(1.0, tk.END)
        if not abc_content.strip():
            messagebox.showinfo("Information", "Please enter ABC notation first.")
            return
        
        # Ask for output file location
        file_path = filedialog.asksaveasfilename(
            title="Save WAV file",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Before conversion, analyze the header to ensure voice mapping is up to date
        self.analyze_abc_header()
        
        # First convert to MIDI
        self.status_var.set("Converting to MIDI...")
        temp_midi_file = os.path.join(self.temp_dir, "export_temp.mid")
        
        if not self.abc_to_midi(abc_content, temp_midi_file):
            self.status_var.set("Conversion to MIDI failed.")
            return
        
        # Now convert MIDI to WAV with effects
        self.status_var.set("Rendering audio with effects...")
        
        # Try to use FluidSynth for high-quality rendering
        success = False
        
        # Get FluidSynth path (either in PATH or in current directory)
        fluidsynth_path = self.get_fluidsynth_path()
        
        if fluidsynth_path:
            soundfont = self.get_soundfont_path()
            if soundfont:
                try:
                    # Create a temporary WAV file
                    temp_wav_file = os.path.join(self.temp_dir, "export_temp.wav")
                    
                    # Use FluidSynth to render MIDI to WAV
                    subprocess.run(
                        [fluidsynth_path, "-ni", soundfont, temp_midi_file, "-F", temp_wav_file, "-r", "44100"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    # Apply audio effects if pedalboard is available
                    if PEDALBOARD_AVAILABLE:
                        self.status_var.set("Applying studio effects...")
                        self.audio_effects.process_wav_file(temp_wav_file, file_path)
                    else:
                        # If pedalboard is not available, just copy the file
                        with open(temp_wav_file, 'rb') as in_f:
                            with open(file_path, 'wb') as out_f:
                                out_f.write(in_f.read())
                    
                    success = True
                    self.status_var.set(f"Exported to: {os.path.basename(file_path)}")
                    messagebox.showinfo("Export Complete", f"Successfully exported to {os.path.basename(file_path)} with effects.")
                
                except Exception as e:
                    print(f"Error exporting to WAV: {str(e)}")
                    self.status_var.set("Export to WAV failed.")
                    messagebox.showerror("Export Error", f"Failed to export to WAV: {str(e)}")
        
        if not success:
            self.status_var.set("High-quality export failed. Please install FluidSynth and a SoundFont.")
            messagebox.showwarning(
                "Export Warning", 
                "Could not create high-quality WAV export. Please install FluidSynth and a SoundFont file."
            )
    
    def cleanup(self):
        """Clean up temporary files"""
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
            return
            
        # Stop any playing audio and release pygame resources
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except:
            pass
            
        try:
            if os.path.exists(self.temp_dir):
                # First try to remove all known temp files
                for temp_file in self.temp_files:
                    file_path = os.path.join(self.temp_dir, temp_file)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Removed temp file: {temp_file}")
                    except Exception as e:
                        print(f"Error removing {temp_file}: {str(e)}")
                
                # Then clean up any remaining files in the directory
                for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            print(f"Removed additional file: {file}")
                        except Exception as e:
                            print(f"Error removing file {file}: {str(e)}")
                
                # Finally remove the temp directory itself
                try:
                    os.rmdir(self.temp_dir)
                    print(f"Removed temp directory: {self.temp_dir}")
                except Exception as e:
                    print(f"Error removing temp directory: {str(e)}")
                
                # Clear the temp_dir attribute
                self.temp_dir = None
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def add_effects_ui(self, parent):
        """Add a simple audio effects UI to the main window"""
        # Create effects frame with bold title
        effects_frame = ttk.LabelFrame(parent, text="Studio Effects", padding=10, style='Header.TLabelframe')
        effects_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create effects controls
        controls_frame = ttk.Frame(effects_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Reverb
        reverb_enabled = tk.BooleanVar(value=self.audio_effects.reverb_enabled)
        reverb_check = ttk.Checkbutton(
            controls_frame, text="Reverb", 
            variable=reverb_enabled,
            command=lambda: self.toggle_effect("reverb", reverb_enabled.get()),
            style='Regular.TCheckbutton'
        )
        reverb_check.grid(row=0, column=0, padx=10)
        
        # Compression
        comp_enabled = tk.BooleanVar(value=self.audio_effects.compression_enabled)
        comp_check = ttk.Checkbutton(
            controls_frame, text="Compression", 
            variable=comp_enabled,
            command=lambda: self.toggle_effect("compression", comp_enabled.get()),
            style='Regular.TCheckbutton'
        )
        comp_check.grid(row=0, column=1, padx=10)
        
        # Chorus
        chorus_enabled = tk.BooleanVar(value=self.audio_effects.chorus_enabled)
        chorus_check = ttk.Checkbutton(
            controls_frame, text="Chorus", 
            variable=chorus_enabled,
            command=lambda: self.toggle_effect("chorus", chorus_enabled.get()),
            style='Regular.TCheckbutton'
        )
        chorus_check.grid(row=0, column=2, padx=10)
        
        # Presets dropdown
        ttk.Label(controls_frame, text="Preset:", style='Regular.TLabel').grid(row=0, column=3, padx=5)
        presets = ["Default", "Concert Hall", "Warm Studio", "Bright & Clear", 
                 "Orchestral", "Deep Bass", "Vocal Enhancer"]
        preset_combo = ttk.Combobox(controls_frame, values=presets, width=15, font=("Arial", 13), style='Large.TCombobox')
        preset_combo.set("Default")
        preset_combo.grid(row=0, column=4, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(e.widget.get()))
        
        # Add diagnostic button
        diagnose_button = ttk.Button(
            controls_frame, 
            text="Diagnose FluidSynth",
            command=self.diagnose_fluidsynth,
            style='Large.TButton'
        )
        diagnose_button.grid(row=0, column=5, padx=10)
        
        # If the pedalboard is not available, show a note
        if not PEDALBOARD_AVAILABLE:
            note_label = ttk.Label(
                controls_frame, 
                text="Install 'pedalboard' package for studio-quality effects",
                foreground="blue",
                style='Regular.TLabel'
            )
            note_label.grid(row=1, column=0, columnspan=5, pady=5)
    
    def toggle_effect(self, effect, enabled):
        """Toggle an audio effect on/off"""
        if effect == "reverb":
            self.audio_effects.reverb_enabled = enabled
            if enabled:
                self.audio_effects.reverb_amount = 0.3
                self.audio_effects.reverb_room_size = 0.5
        elif effect == "compression":
            self.audio_effects.compression_enabled = enabled
            if enabled:
                self.audio_effects.compression_threshold = -20
                self.audio_effects.compression_ratio = 4.0
        elif effect == "chorus":
            self.audio_effects.chorus_enabled = enabled
            if enabled:
                self.audio_effects.chorus_rate = 1.0
                self.audio_effects.chorus_depth = 0.25
        
        self.audio_effects.update_pedalboard()
        self.status_var.set(f"{effect.capitalize()} {'enabled' if enabled else 'disabled'}")

    def adjust_window_size(self):
        """Adjust window size to show all elements without scrolling if possible"""
        # Update the scroll region
        self.canvas.update_idletasks()
        
        # Get the required height for all content
        scroll_region = self.canvas.bbox("all")
        if not scroll_region:
            return
            
        content_height = scroll_region[3] - scroll_region[1]
        
        # Get current window dimensions
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate the total height needed (including the bottom frame)
        bottom_frame_height = 0
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame) and widget.winfo_y() > 0:
                bottom_frame_height += widget.winfo_reqheight()
        
        total_height_needed = content_height + bottom_frame_height + 50  # Add padding
        
        # Calculate new dimensions (don't exceed 90% of screen size)
        new_height = min(total_height_needed, screen_height * 0.9)
        new_width = min(max(current_width, 1000), screen_width * 0.9)  # Keep minimum width of 1000
        
        # Set new size
        self.root.geometry(f"{int(new_width)}x{int(new_height)}")
        
        # Update the canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    # Check for ABC2MIDI
    try:
        subprocess.run(['abc2midi', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        missing.append("abcMIDI")
    
    # Check for pygame
    try:
        import pygame
    except:
        missing.append("pygame")
    
    # Check for numpy
    try:
        import numpy
    except:
        missing.append("numpy")
    
    # Check for pydub
    try:
        import pydub
    except:
        missing.append("pydub")
    
    # Check for mido
    try:
        import mido
    except:
        missing.append("mido")
    
    # Check for advanced audio processing libraries (not required but recommended)
    try:
        import pedalboard
    except:
        missing.append("pedalboard (optional for studio effects)")
    
    try:
        import sounddevice
    except:
        missing.append("sounddevice (optional for studio effects)")
    
    try:
        import scipy
    except:
        missing.append("scipy (optional for studio effects)")
    
    # Check for FluidSynth (first in project directory, then in PATH)
    fluidsynth_found = False
    
    # Check project directory first - use explicit path
    project_dir = "E:\\projects\\music GEN"
    project_fluidsynth = os.path.join(project_dir, "important_to_run", "FluidSynth", "bin", "fluidsynth.exe")
    
    if os.path.exists(project_fluidsynth):
        print(f"Found FluidSynth in project directory: {project_fluidsynth}")
        fluidsynth_found = True
    else:
        # Then check PATH
        try:
            result = subprocess.run(['fluidsynth', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                fluidsynth_found = True
        except:
            # Check common installation locations
            common_locations = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), "FluidSynth", "bin", "fluidsynth.exe"),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), "FluidSynth", "bin", "fluidsynth.exe"),
                os.path.join(os.path.expanduser("~"), "FluidSynth", "bin", "fluidsynth.exe"),
                "C:\\FluidSynth\\bin\\fluidsynth.exe",
                os.path.join(os.getcwd(), "fluidsynth.exe")
            ]
            
            for path in common_locations:
                if os.path.exists(path):
                    fluidsynth_found = True
                    break
    
    if not fluidsynth_found:
        missing.append("FluidSynth (optional for high-quality rendering)")
    
    return missing

def install_instructions():
    """Return instructions for installing missing dependencies"""
    instructions = """
    To install required dependencies:
    
    1. ABC2MIDI (abcMIDI package):
       - Windows: Download from https://ifdo.ca/~seymour/runabc/abcMIDI-win.html
       - Mac/Linux: Use package manager or compile from source
    
    2. Python packages:
       pip install pygame numpy pydub mido
    
    3. Optional (for studio-quality audio effects):
       pip install pedalboard sounddevice scipy
    
    4. FluidSynth (for high-quality audio rendering):
       - Windows: Download from https://github.com/FluidSynth/fluidsynth/releases
       - Mac: brew install fluid-synth
       - Linux: apt-get install fluidsynth
    
    5. SoundFont file (for FluidSynth):
       - Download free GM soundfont from https://musical-artifacts.com/artifacts/28
       - Place in a directory named 'soundfonts' in your home folder
    """
    return instructions

def main():
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Filter out optional dependencies for the warning message
    critical_deps = [dep for dep in missing_deps if "optional" not in dep]
    
    if missing_deps:
        print("Missing dependencies:", ", ".join(missing_deps))
        print(install_instructions())
        
        if "abcMIDI" in " ".join(missing_deps):
            print("\nWARNING: abcMIDI (abc2midi) is required for this application to work.")
            print("The application will start but won't be able to play music without abc2midi.")
        
        # Only show warning dialog for critical dependencies
        if critical_deps:
            messagebox.showwarning(
                "Missing Dependencies",
                "Some critical dependencies are missing:\n" + 
                ", ".join(critical_deps) + 
                "\n\nSee console for installation instructions."
            )
    
    # Start the application
    root = tk.Tk()
    app = ABCMusicPlayer(root)
    
    # Run FluidSynth diagnostics if it's missing
    if "FluidSynth" in " ".join(missing_deps):
        print("\nRunning FluidSynth diagnostic to help with setup...")
        app.diagnose_fluidsynth()
    
    # Clean up on exit
    def on_closing():
        app.stop_playing()
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()