import tkinter as tk
from tkinter import ttk, scrolledtext
import re

class ABCCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("ABC Code Corrector")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input section
        input_label = ttk.Label(main_frame, text="Input ABC Code:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(main_frame, height=12, width=80, wrap=tk.WORD)
        self.input_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Correct button
        self.correct_button = ttk.Button(main_frame, text="Correct ABC Code", command=self.correct_abc_code)
        self.correct_button.grid(row=2, column=0, pady=(0, 10))
        
        # Output section
        output_label = ttk.Label(main_frame, text="Corrected ABC Code:")
        output_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=12, width=80, wrap=tk.WORD)
        self.output_text.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Copy button
        self.copy_button = ttk.Button(main_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.grid(row=5, column=0, pady=(0, 10))

    def correct_abc_code(self):
        # Get input text
        abc_code = self.input_text.get("1.0", tk.END).strip()
        
        # Apply corrections
        corrected_code = self.apply_corrections(abc_code)
        
        # Update output text
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", corrected_code)

    def extract_components(self, code):
        """Extract headers, voices, and music from ABC code"""
        headers = []
        voices = []
        music = []
        
        lines = code.split('\n')
        current_section = 'headers'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if current_section == 'headers':
                if line.startswith('V:'):
                    current_section = 'voices'
                    voices.append(line)
                elif line.startswith(('%', 'L:', 'Q:', 'M:', 'K:', '%%')):
                    headers.append(line)
                elif line.startswith('[r:'):
                    current_section = 'music'
                    music.append(line)
            elif current_section == 'voices':
                if line.startswith('V:'):
                    voices.append(line)
                elif line.startswith('[r:'):
                    current_section = 'music'
                    music.append(line)
            elif current_section == 'music' and line.startswith('[r:'):
                music.append(line)
                
        return headers, voices, music

    def apply_corrections(self, code):
        if not code:
            return ""
            
        # Split into sections by %Classical marker
        sections = re.split(r'(?=%Classical\n)', code)
        
        # Use first section for headers and voices
        main_headers, main_voices, main_music = self.extract_components(sections[0])
        
        # Collect all music content, keeping only the most complete version
        all_music = []
        max_measures = 0
        best_music = []
        
        for section in sections:
            _, _, music = self.extract_components(section)
            if music:
                # Count measures to determine most complete section
                measure_count = len([line for line in music if line.startswith('[r:')])
                if measure_count > max_measures:
                    max_measures = measure_count
                    best_music = music
        
        # Process music lines
        corrected_music = []
        seen_measures = set()
        
        for line in best_music:
            if not line.startswith('[r:'):
                continue
                
            # Extract measure number
            measure_num = re.search(r'\[r:(\d+)/', line)
            if measure_num:
                measure_id = measure_num.group(1)
                if measure_id in seen_measures:
                    continue
                seen_measures.add(measure_id)
            
            # Fix syntax
            line = re.sub(r'\|\s*\[V:', '| [V:', line)  # Fix spacing before voice
            line = re.sub(r'\[V:', ' [V:', line)        # Ensure space before voice
            line = re.sub(r'\s+', ' ', line.strip())    # Normalize spacing
            
            # Ensure bar line at end
            if not line.endswith('|'):
                line += '|'
                
            # Fix bracket mismatches
            open_brackets = line.count('[')
            close_brackets = line.count(']')
            if open_brackets > close_brackets:
                line += ']' * (open_brackets - close_brackets)
            elif close_brackets > open_brackets:
                line = line[:-(close_brackets - open_brackets)]
                
            corrected_music.append(line)
        
        # Assemble final output
        final_code = []
        final_code.extend(main_headers)
        final_code.append('')  # Empty line
        final_code.extend(main_voices)
        final_code.append('')  # Empty line
        final_code.extend(corrected_music)
        
        return '\n'.join(final_code)

    def copy_to_clipboard(self):
        corrected_code = self.output_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(corrected_code)
        self.root.update()

def main():
    root = tk.Tk()
    app = ABCCorrector(root)
    root.mainloop()

if __name__ == "__main__":
    main()