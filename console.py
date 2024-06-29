import subprocess
import time
import requests
import json


def main():
    context_window = []
    ollama_process = None
    session = requests.Session()
    conversation_id = None
    try:
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give the server time to run
        print("Ollama server started successfully.")
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {"model": "phi3:mini", "prompt": input("\nEnter prompt:\n"), "stream": False, "context": context_window,
            "system": "You are Phi 3, a general-use large language model."}
    while True:
        if conversation_id:
            data["conversation_id"] = conversation_id
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            code = json.loads(response.text)["response"]
            code = code.replace("```", "")
            context_window.extend(json.loads(response.text)["context"])
            print(code)
        else:
            print("API Error:", response.status_code, response.text)
        data["prompt"] = input("\nEnter prompt:\n")


if __name__ == '__main__':
    main()
