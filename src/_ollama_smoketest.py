import json
import ollama

print("import_ok")
resp = ollama.chat(
    model="phi3:mini",
    messages=[{"role": "user", "content": "Return ONLY this JSON object with key a and value 1."}],
    format="json",
    options={"temperature": 0},
)
print(type(resp))
content = resp.get("message", {}).get("content", "")
print("content:", content)
print("parsed:", json.loads(content))
