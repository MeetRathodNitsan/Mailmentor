import requests
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, prompt: str, model: str = "llama2:latest") -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30  # Increased timeout to 120 seconds
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.Timeout:
            print("Request timed out. The server took too long to respond.")
            return None
        except requests.ConnectionError:
            print("Connection error. Make sure Ollama server is running.")
            return None
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            return None

    def summarize(self, text: str, model: str = "llama2:latest") -> Optional[str]:
        prompt = f"Summarize this text in one clear, concise sentence: {text}"
        return self.generate(prompt, model)

    def generate_response(self, subject: str, content: str, model: str = "llama2:latest") -> Optional[str]:
        prompt = f"Generate a professional one-sentence response to this email:\nSubject: {subject}\nContent: {content}"
        return self.generate(prompt, model)