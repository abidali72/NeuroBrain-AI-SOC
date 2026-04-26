"""
🧠 Neuro Brain — Desktop Application
A graphical user interface (GUI) to interact with your trained AI models.

Run this script to launch the app:
    python desktop_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Load the quick predictor from our model manager
try:
    from src.model_manager import QuickPredictor
except ImportError:
    print("❌ Error: Could not import QuickPredictor. Make sure you are running this from the 'brain' directory.")
    exit(1)


class NeuroBrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Neuro Brain")
        self.root.geometry("650x500")
        
        # Modern dark theme colors
        self.bg_color = '#1a1a2e'
        self.panel_color = '#16213e'
        self.accent_color = '#e94560'
        self.text_color = '#ffffff'
        self.success_color = '#00ff88'
        self.warning_color = '#ffb347'
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize the AI models
        self.predictor = None
        self._show_loading_screen()
        
        # Use simple 'after' to allow window to render before loading heavy models
        self.root.after(100, self._init_models)

    def _show_loading_screen(self):
        """Show a temporary screen while models load."""
        self.loading_frame = tk.Frame(self.root, bg=self.bg_color)
        self.loading_frame.pack(fill='both', expand=True)
        
        tk.Label(self.loading_frame, text="🧠 Neuro Brain", 
                 font=('Arial', 28, 'bold'), bg=self.bg_color, fg=self.accent_color).pack(pady=(150, 10))
        
        tk.Label(self.loading_frame, text="Loading AI Models into Memory...\nThis may take a few seconds.", 
                 font=('Arial', 12), bg=self.bg_color, fg=self.text_color).pack()

    def _init_models(self):
        """Load the PyTorch/Transformer models in the background."""
        try:
            self.predictor = QuickPredictor()
            # Destory loading screen and build main UI
            self.loading_frame.destroy()
            self._build_main_ui()
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                                 f"Failed to load models. Did you train them first?\n\nError: {str(e)}")
            self.root.destroy()

    def _build_main_ui(self):
        """Construct the main tabbed interface."""
        # Title Header
        title_frame = tk.Frame(self.root, bg=self.bg_color)
        title_frame.pack(fill='x', pady=15)
        
        tk.Label(title_frame, text="🧠 Neuro Brain", 
                 font=('Arial', 24, 'bold'), bg=self.bg_color, fg=self.accent_color).pack()
        tk.Label(title_frame, text="AI Interaction Console | v1.0", 
                 font=('Arial', 10), bg=self.bg_color, fg=self.text_color).pack()

        # Custom styling for Tabs
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', padding=[20, 5], font=('Arial', 12, 'bold'), 
                        background=self.panel_color, foreground=self.text_color)
        style.map('TNotebook.Tab', background=[('selected', self.accent_color)])

        # Create Notebook (Tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Add specific task tabs
        self._build_number_tab(notebook)
        self._build_text_tab(notebook)

    def _build_number_tab(self, notebook):
        """Build the House Price Predictor UI."""
        frame = tk.Frame(notebook, bg=self.panel_color)
        notebook.add(frame, text=' 🔢 Price Predictor ')

        # Instructions
        tk.Label(frame, text="Predict house prices based on specifications", 
                 font=('Arial', 14, 'bold'), bg=self.panel_color, fg=self.text_color).pack(pady=(20, 10))
        
        # Input Form Grid
        form_frame = tk.Frame(frame, bg=self.panel_color)
        form_frame.pack(pady=10)

        # Labels and Entries
        fields = [
            ("Square Feet:", "2500"),
            ("Bedrooms:", "4"),
            ("Age (years):", "15"),
            ("Distance to City (miles):", "5.5")
        ]
        
        self.num_entries = {}
        for row, (label_text, default_val) in enumerate(fields):
            tk.Label(form_frame, text=label_text, font=('Arial', 12), 
                     bg=self.panel_color, fg=self.text_color, anchor='e').grid(row=row, column=0, padx=10, pady=5, sticky='e')
            
            entry = tk.Entry(form_frame, font=('Arial', 12), width=15, bg='#ffffff', fg='#000000')
            entry.insert(0, default_val)
            entry.grid(row=row, column=1, padx=10, pady=5, sticky='w')
            self.num_entries[label_text] = entry

        # Predict Button
        tk.Button(frame, text="CALCULATE PRICE", font=('Arial', 12, 'bold'), 
                  bg=self.accent_color, fg=self.text_color, activebackground='#d83a54',
                  activeforeground='white', borderwidth=0, padx=20, pady=10,
                  command=self._predict_number).pack(pady=20)

        # Result display
        self.num_result = tk.Label(frame, text="", font=('Arial', 24, 'bold'), 
                                   bg=self.panel_color, fg=self.success_color)
        self.num_result.pack()

    def _build_text_tab(self, notebook):
        """Build the Text Sentiment Analysis UI."""
        frame = tk.Frame(notebook, bg=self.panel_color)
        notebook.add(frame, text=' 📝 Text Analyzer ')

        tk.Label(frame, text="Detect the emotion/sentiment of any text block", 
                 font=('Arial', 14, 'bold'), bg=self.panel_color, fg=self.text_color).pack(pady=(20, 10))

        # Text Area
        self.text_input = tk.Text(frame, font=('Arial', 12), height=6, width=50, 
                                  bg='#ffffff', fg='#000000', wrap='word')
        self.text_input.insert('1.0', "Type a product review, tweet, or message here to see if the AI thinks it's positive or negative!")
        self.text_input.pack(padx=20, pady=10)

        # Analyze Button
        tk.Button(frame, text="ANALYZE SENTIMENT", font=('Arial', 12, 'bold'), 
                  bg=self.accent_color, fg=self.text_color, activebackground='#d83a54',
                  activeforeground='white', borderwidth=0, padx=20, pady=10,
                  command=self._analyze_text).pack(pady=10)

        # Result display panel
        result_frame = tk.Frame(frame, bg=self.panel_color)
        result_frame.pack(fill='x', pady=10)
        
        self.text_emoji = tk.Label(result_frame, text="", font=('Arial', 36), bg=self.panel_color)
        self.text_emoji.pack(side='left', padx=(50, 10))
        
        info_frame = tk.Frame(result_frame, bg=self.panel_color)
        info_frame.pack(side='left', anchor='w')
        
        self.text_label = tk.Label(info_frame, text="", font=('Arial', 18, 'bold'), 
                                   bg=self.panel_color, fg=self.text_color)
        self.text_label.pack(anchor='w')
        
        self.text_conf = tk.Label(info_frame, text="", font=('Arial', 12), 
                                  bg=self.panel_color, fg='#aaaaaa')
        self.text_conf.pack(anchor='w')

    def _predict_number(self):
        """Handle the number prediction button click."""
        try:
            # Extract and convert inputs
            features = [
                float(self.num_entries["Square Feet:"].get()),
                float(self.num_entries["Bedrooms:"].get()),
                float(self.num_entries["Age (years):"].get()),
                float(self.num_entries["Distance to City (miles):"].get())
            ]
            
            # Predict
            result = self.predictor.predict_number(features)
            
            # Format and display
            self.num_result.config(text=f"Estimated Value: ${result:,.0f}", fg=self.success_color)
            
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter valid numbers in all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _analyze_text(self):
        """Handle the sentiment analysis button click."""
        try:
            text = self.text_input.get('1.0', 'end').strip()
            if not text:
                return
                
            # Predict
            result = self.predictor.predict_text(text)
            
            # Update UI
            label = result['label']
            conf = result['confidence']
            
            if label == 'POSITIVE':
                self.text_emoji.config(text="😊")
                self.text_label.config(text="POSITIVE", fg=self.success_color)
            else:
                self.text_emoji.config(text="😞")
                self.text_label.config(text="NEGATIVE", fg=self.warning_color)
                
            self.text_conf.config(text=f"AI Confidence: {conf:.1%}")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroBrainApp(root)
    root.mainloop()
