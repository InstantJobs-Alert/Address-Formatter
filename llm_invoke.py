import sys, os, subprocess, time
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("matplotlib")
install("ollama")

import ollama


class LLMInvoke:
    def __init__(self, model: str = None):
        self.selected_model = model

    def set_model(self, model: str):
        self.selected_model = model

    def invoke(self, input: str) -> str:
        if not self.selected_model:
            return "No model selected."
        try:
            response = ollama.chat(
                model=self.selected_model,
                messages=[
                    {
                        "role": "user",
                        "temperature": 0,
                        "repeat_penalty": 6,
                        "content": input,
                        "num_ctx": 4096,
                    },
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            return str(e)