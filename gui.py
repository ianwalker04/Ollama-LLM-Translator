import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from typing import List
from typing import Optional
import requests
import json
import threading
import subprocess
import re
import time
from sacrebleu import sentence_bleu as bleu


def is_cjk(text: str) -> Optional[str]:
    # Detects if language is in Chinese, Japanese, Korean, or none
    if re.search("[\u4e00-\u9FFF]", text):
        return "ZH"
    if re.search("[\u3040-\u30ff]", text):
        return "JA"
    if re.search("[\uac00-\ud7a3]", text):
        return "KO"
    return None


def get_models() -> List[str]:
    # Get a list of all currently installed Ollama models
    model_list_process = subprocess.check_output("ollama list", shell=True, encoding="utf-8")
    model_names = re.findall(r'^(\S+)', model_list_process, re.MULTILINE)
    return model_names[1:]


def show_prompting_tips() -> None:
    # Display a messagebox containing prompting tips
    prompting_tips = open("prompting_tips.txt", "r", encoding="utf-8")
    messagebox.showinfo(title="Prompting Tips", message=prompting_tips.read())
    prompting_tips.close()


def download_transcript() -> None:
    # Download a TXT file of the transcript
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    transcript = open("transcript.txt", "r", encoding="utf-8")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(transcript.read())
    transcript.close()


class GUI:

    def __init__(self):
        # Server variables
        self.ollama_process = None
        self.start_ollama_server()
        with open("transcript.txt", "w") as transcript:
            transcript.write("Transcript" + "\n\n")

        # Main GUI elements
        self.root = tk.Tk()
        self.root.geometry("1100x650")
        self.root.title("Ollama LLM Translator")

        tk.Label(self.root, text="Ollama LLM Translator", font=("Times New Roman", 18)).pack(padx=20, pady=10)

        # Model/BLEU/Prompting/model parameter GUI elements
        self.parameter_frame = tk.Frame(self.root)
        self.parameter_frame.pack()

        self.option_value = tk.StringVar(self.root)
        self.option_value.set("Select Model")
        self.option_menu = tk.OptionMenu(self.parameter_frame, self.option_value, *get_models())
        self.option_menu.pack(padx=1, pady=1, side="left")
        self.option_menu.config(indicatoron=False, font=("Times New Roman", 8))

        self.import_bleu_button = tk.Button(self.parameter_frame, text="Import References (BLEU)", font=("Times New "
                                                                                                         "Roman", 8),
                                            command=self.import_bleu)
        self.import_bleu_button.pack(padx=1, pady=1, side="left")
        self.clear_bleu_button = tk.Button(self.parameter_frame, text="Clear References (BLEU)",
                                           font=("Times New Roman", 8),
                                           command=self.clear_bleu)
        self.clear_bleu_button.pack(padx=1, pady=1, side="left")
        self.prompting_tips_button = tk.Button(self.parameter_frame, text="Show Prompting Tips",
                                               font=("Times New Roman", 8),
                                               command=show_prompting_tips)
        self.prompting_tips_button.pack(padx=1, pady=1, side="left")

        tk.Label(self.parameter_frame, text="Temperature - Top-K - Top-P", font=("Times New Roman", 12)).pack()
        self.temp_input = tk.Entry(self.parameter_frame)
        self.temp_input.pack(padx=10, pady=10, side="left")
        self.temp_input.insert("0", "0.8")
        self.top_k_input = tk.Entry(self.parameter_frame)
        self.top_k_input.pack(padx=10, pady=10, side="left")
        self.top_k_input.insert("0", "40")
        self.top_p_input = tk.Entry(self.parameter_frame)
        self.top_p_input.pack(padx=10, pady=10, side="left")
        self.top_p_input.insert("0", "0.9")

        # Text box/translate button GUI elements
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        tk.Label(self.main_frame, text="Input:", font=("Times New Roman", 12)).pack()
        self.input_box = tk.Text(self.main_frame, height=5, font=("Times New Roman", 16))
        self.input_box.pack(padx=10, pady=10)

        tk.Label(self.main_frame, text="Prompt:", font=("Times New Roman", 12)).pack()
        self.prompt_box = tk.Text(self.main_frame, height=1, font=("Times New Roman", 16))
        self.prompt_box.pack(padx=10, pady=10)

        self.translate_button = tk.Button(self.main_frame, text="Translate", font=("Times New Roman", 12),
                                          command=self.translate)
        self.translate_button.pack(padx=10, pady=10)

        tk.Label(self.main_frame, text="Output:", font=("Times New Roman", 12)).pack()
        self.output_box = tk.Text(self.main_frame, height=5, font=("Times New Roman", 16))
        self.output_box.pack(padx=10, pady=10)

        # Time/BLEU label GUI elements
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack()

        self.time_label = tk.Label(self.stats_frame, text="Translation Time (Seconds): N/A",
                                   font=("Times New Roman", 10))
        self.time_label.pack(side="left")
        self.start_time = 0
        self.end_time = 0

        self.bleu_label = tk.Label(self.stats_frame, text="BLEU Score: N/A", font=("Times New Roman", 12))
        self.bleu_label.pack(side="bottom")

        self.recalculate_bleu_button = tk.Button(self.stats_frame, text="Recalculate BLEU",
                                                 font=("Times New Roman", 10),
                                                 command=self.calculate_bleu)
        self.recalculate_bleu_button.pack(side="right", padx=10)
        self.download_transcript_button = tk.Button(self.stats_frame, text="Download TXT Transcript",
                                                    font=("Times New Roman", 10),
                                                    command=download_transcript)
        self.download_transcript_button.pack(padx=1, pady=1)

        # API variables
        self.context_window = []
        self.references = []
        self.url = "http://localhost:11434/api/generate"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.conversation_id = None

        self.root.mainloop()

    def import_bleu(self) -> None:
        # Ask user to import a .txt file and add it to a list of reference texts
        file_path = filedialog.askopenfilename(filetypes=[("TXT Files", "*.txt")])
        file = open(file_path, "r", encoding="utf-8")
        self.references.append(file.read())
        file.close()
        messagebox.showinfo(title="Files Imported", message="Text files successfully imported.")

    def clear_bleu(self) -> None:
        # Clear the list of reference texts
        self.references = []
        messagebox.showinfo(title="Files Cleared", message="Text files successfully cleared.")

    def start_ollama_server(self) -> None:
        # Start the Ollama server
        try:
            self.ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)  # Give the server time to start up
            print("Ollama server started successfully.")
        except Exception as e:
            print(f"Failed to start Ollama server: {e}")

    def translate(self) -> None:
        # Disable GUI elements and begin the API call
        self.output_box.delete("1.0", tk.END)
        [widget.config(state=tk.DISABLED) for widget in self.parameter_frame.winfo_children()]
        [widget.config(state=tk.DISABLED) for widget in self.main_frame.winfo_children()]
        [widget.config(state=tk.DISABLED) for widget in self.stats_frame.winfo_children()]
        self.recalculate_bleu_button.config(state=tk.DISABLED)
        self.start_time = time.time()  # Begin translation timer
        threading.Thread(target=self.api_call, daemon=True).start()

    def display_bleu(self, score: float) -> None:
        # Update BLEU score label with appropriate score and color
        score = score / 100
        self.bleu_label.config(text="BLEU Score: " + str(score))
        if 0 <= score < 0.1:
            self.bleu_label.config(fg="red4")
        elif 0.1 <= score < 0.19:
            self.bleu_label.config(fg="red2")
        elif 0.2 <= score < 0.29:
            self.bleu_label.config(fg="orange")
        elif 0.3 <= score < 0.39:
            self.bleu_label.config(fg="lime green")
        elif 0.4 <= score < 0.49:
            self.bleu_label.config(fg="green2")
        else:
            self.bleu_label.config(fg="green4")
        with open("transcript.txt", "a") as transcript:
            transcript.write("BLEU: " + str(score) + "\n\n")
        transcript.close()

    def calculate_bleu(self) -> None:
        # If there is text and references, calculate the BLEU score with the appropriate tokenizer
        text = self.output_box.get("1.0", tk.END)
        if text and self.references:
            if is_cjk(text) == "ZH":
                self.display_bleu(bleu(text, self.references, tokenize="zh").score)
            elif is_cjk(text) == "JA":
                self.display_bleu(bleu(text, self.references, tokenize="ja-mecab").score)
            elif is_cjk(text) == "KO":
                self.display_bleu(bleu(text, self.references, tokenize="ko-mecab").score)
            else:
                self.display_bleu(bleu(text, self.references).score)
        else:
            self.bleu_label.config(text="BLEU Score: N/A", fg="black")
            with open("transcript.txt", "a") as transcript:
                transcript.write("BLEU: N/A" + "\n\n")
            transcript.close()

    def api_call(self) -> None:
        # Perform API call and display result in the output box
        data = {"model": self.option_value.get(),
                "prompt": self.prompt_box.get("1.0", tk.END) + self.input_box.get("1.0", tk.END),
                "stream": False,
                "context": self.context_window,
                "system": "You are an expert translator who translates text that the user gives you into a language "
                          "of their choosing."
                          "When translating text, DO NOT add any additional comments, explanations, "
                          "or pronunciations. This includes quotes explanations of the translations, parentheticals, "
                          "etc. Just give the translated text."
                          "Unless prompted otherwise, do not give multiple translations for a text. Just give one "
                          "translation.",
                "options": {"temperature": float(self.temp_input.get()), "top_k": int(self.top_k_input.get()),
                            "top_p": float(self.top_p_input.get())}}
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            output = json.loads(response.text)["response"]
            output = output.replace("```", "")
            self.context_window.extend(json.loads(response.text)["context"])
            self.output_box.config(state=tk.NORMAL)
            self.output_box.insert("1.0", output)
            self.end_time = time.time()  # End translation timer
            self.time_label.config(text="Translation Time (Seconds): " + str(self.end_time - self.start_time))
            with open("transcript.txt", "a", encoding="utf-8") as transcript:
                transcript.write(("Output: " + output + "\n" + "Model: " + self.option_value.get() + "\n" +
                                  "Input: " + self.input_box.get("1.0", tk.END) +
                                  "Prompt: " + self.prompt_box.get("1.0", tk.END) + "Time (Seconds): " +
                                  str(self.end_time - self.start_time) + "\n"))
            transcript.close()
            self.calculate_bleu()
        else:
            messagebox.showinfo(title="API Error",
                                message="API Error: " + str(response.status_code) + ". " + response.text)
        [widget.config(state=tk.NORMAL) for widget in self.parameter_frame.winfo_children()]
        [widget.config(state=tk.NORMAL) for widget in self.main_frame.winfo_children()]
        [widget.config(state=tk.NORMAL) for widget in self.stats_frame.winfo_children()]
        self.recalculate_bleu_button.config(state=tk.NORMAL)
