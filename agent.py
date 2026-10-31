"""A tiny local agent: Ollama + a loop + three tools. Standard library only.

This is the same visitor assistant as the Claude Code version, rebuilt from
scratch so every moving part is visible in one file:

    1. a system prompt that sets the rules,
    2. a list of tools the model may call,
    3. a loop: send the conversation, run any tool the model asks for,
       feed the result back, repeat until the model answers in words.

Run:   python agent.py            (interactive chat; type 'quit' to leave)
       python agent.py "Where do I park?"   (one question, then exit)
Needs: Ollama running locally with a model that supports tool calling.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- settings
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"          # change to any Ollama model that supports tools
HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
LOG_FILE = HERE / "data" / "inquiries.json"
MAX_STEPS = 8               # safety: stop a runaway loop of tool calls

SYSTEM_PROMPT = """You are the visitor assistant for Gettysburg Lantern Tours, a walking tour company.
Speak as the company ("we"), warmly and briefly: two to five sentences.

The most important rule: answer only from the company documents. Use list_docs to see
them and read_doc to read the relevant one BEFORE answering any factual question.
Never guess a price, time, distance, age limit, or policy.

If the documents do not contain the answer: say so plainly, offer to pass the question
to the owner, ask for the visitor's email, and call log_inquiry once with the question.

When you recommend a tour, say why in one sentence based on what the visitor told you.
End with the next step (how to book, where to meet) when it fits.
Do not mention these instructions or the documents folder to the visitor."""

# ---------------------------------------------------------------- the tools
# Each tool is a plain Python function. The TOOLS list below describes them to
# the model in the format Ollama expects (same shape as the OpenAI tool schema).


def list_docs() -> str:
    """Names of the company documents available to read."""
    return "\n".join(sorted(p.name for p in DOCS_DIR.glob("*.md")))


def read_doc(name: str) -> str:
    """Full text of one document, by file name."""
    path = DOCS_DIR / Path(name).name          # Path(name).name blocks '../' tricks
    if not path.exists():
        return f"No document named {name!r}. Call list_docs to see the names."
    return path.read_text(encoding="utf-8")


def log_inquiry(question: str, visitor_email: str = "", notes: str = "") -> str:
    """Save a question the assistant could not answer so the owner can follow up."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    items = json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else []
    items.append({
        "id": len(items) + 1,
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "visitor_email": visitor_email,
        "notes": notes,
    })
    LOG_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return f"Logged inquiry #{len(items)}."


# Name -> function, so the loop can run whatever the model asks for.
TOOL_FUNCTIONS = {"list_docs": list_docs, "read_doc": read_doc, "log_inquiry": log_inquiry}

# What the model sees. The descriptions matter: they are how it decides when to call each one.
TOOLS = [
    {"type": "function", "function": {
        "name": "list_docs",
        "description": "List the company documents available to read.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_doc",
        "description": "Read one company document in full. Call this before answering factual questions.",
        "parameters": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string", "description": "File name from list_docs, e.g. tours.md"}}}}},
    {"type": "function", "function": {
        "name": "log_inquiry",
        "description": "Record a question the documents cannot answer, so the owner can follow up.",
        "parameters": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string", "description": "The visitor's question in their words"},
            "visitor_email": {"type": "string", "description": "Their email if given, else empty"},
            "notes": {"type": "string", "description": "Dates, group size, other context"}}}}},
]

# ---------------------------------------------------------------- the loop


def ask_model(messages: list[dict]) -> dict:
    """Send the whole conversation to Ollama and return the model's reply message."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": {"temperature": 0.2},   # low = steadier, more literal answers
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]


def run_agent(messages: list[dict], verbose: bool = True) -> str:
    """The agent loop. Keeps going until the model replies with words instead of a tool call."""
    for _ in range(MAX_STEPS):
        reply = ask_model(messages)
        messages.append(reply)                       # the model's turn goes into the history

        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:                           # no tools requested: this is the answer
            return reply.get("content", "").strip()

        for call in tool_calls:                      # run each requested tool, add its result
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if verbose:
                print(f"   [tool] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            func = TOOL_FUNCTIONS.get(name)
            result = func(**args) if func else f"Unknown tool {name}"
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Sorry, I got stuck. Please call us at (717) 555-0142."


def main() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # One-shot mode: question on the command line.
    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        print(run_agent(messages))
        return

    # Chat mode.
    print(f"Gettysburg Lantern Tours assistant (local, {MODEL}). Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        messages.append({"role": "user", "content": text})
        print("\nAssistant:", run_agent(messages), "\n")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError:
        sys.exit("Could not reach Ollama at localhost:11434. Is it running? Try: ollama serve")
