import os
import json
from openai import OpenAI
import tiktoken

# === CONFIG ===
MODEL = "gpt-4o"  # or "o1" if you have access
CONTEXT_LIMIT = 16000
COMPRESS_THRESHOLD = 0.8  # compress at 80%
SESSION_FILE = "chat_session.json"
API_KEY = "YOUR_API_KEY"

# === SETUP ===
client = OpenAI(api_key=API_KEY)
encoding = tiktoken.encoding_for_model(MODEL)

def count_tokens(messages):
    text = "".join([m["content"] for m in messages])
    return len(encoding.encode(text))

# === COMPRESSION PROMPT ===
COMPRESS_PROMPT = """
You are now acting as a "Context Compressor."
Your job is to compress the conversation so far into the smallest possible token count while preserving:
1. All technical instructions and constraints
2. All key facts, variables, data structures, or decisions
3. Relevant assumptions or reasoning steps that future answers will rely on
4. Current goals and sub-goals of the project

Output format:
- Summary of current objectives
- Condensed history (chronological bullet points, shortest phrasing possible)
- Key facts and variables (short JSON if possible)
- Open issues / pending questions

Make the compression self-contained so it can fully replace the previous conversation in context.
"""

# === FILE HELPERS ===
def save_session(messages):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    print("[Session saved]")

def load_session():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)
        print("[Previous session loaded]")
        return messages
    else:
        print("[New session started]")
        return [{"role": "system", "content": "You are ChatGPT helping with development projects."}]

# === CONTEXT COMPRESSION ===
def compress_context(messages):
    compression_request = messages + [{"role": "user", "content": COMPRESS_PROMPT}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=compression_request,
        max_tokens=800,
    )
    summary = response.choices[0].message.content
    print("\n[Context compressed and saved!]\n")
    new_context = [{"role": "system", "content": "Compressed summary of previous chat:\n" + summary}]
    save_session(new_context)
    return new_context

# === MAIN CHAT LOOP ===
def chat_loop():
    messages = load_session()
    print("Auto-Compress Chat (Persistent) — type 'exit' to quit, 'save' to save manually.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            save_session(messages)
            print("Goodbye!")
            break
        elif user_input.lower() == "save":
            save_session(messages)
            continue

        messages.append({"role": "user", "content": user_input})
        total_tokens = count_tokens(messages)

        # Auto compress if close to context limit
        if total_tokens > CONTEXT_LIMIT * COMPRESS_THRESHOLD:
            messages = compress_context(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1500,
        )

        reply = response.choices[0].message.content
        print(f"Assistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

        # Save progress each turn
        save_session(messages)

if __name__ == "__main__":
    chat_loop()
