import json
import requests

def generate_response_streaming(prompt: str):
    """
    Streams the response from Ollama chunk by chunk.
    """
    print("\n\n--- STARTING STREAMED RESPONSE ---")
    payload = {
        "model": "gemma4:cloud",
        "prompt": prompt,
        "stream": True # This is the crucial setting for streaming!
    }

    try:
        # We use stream=True with requests.post to process content chunk by chunk
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=payload, 
            stream=True
        )
        response.raise_for_status()

        full_response = ""
        # Iterate over the response content line by line (chunk by chunk)
        for chunk in response.iter_lines():
            if chunk:
                # Ollama streams JSON objects separated by newlines
                try:
                    data = json.loads(chunk.decode('utf-8'))
                except json.JSONDecodeError:
                    continue # Skip invalid chunks

                content = data.get("response")
                if content:
                    print(content, end="", flush=True) # Print immediately to the console
                    full_response += content
        
        print("\n\n--- STREAMING COMPLETE ---")
        return full_response

    except requests.exceptions.ConnectionError:
        print("🚨 ERROR: Cannot connect to Ollama.")


# Example Usage
if __name__ == "__main__":
    user_prompt = "i'm running gemma4 locally via ollama. how do i interface it programmatically? show me an example in python"
    generate_response_streaming(user_prompt)
