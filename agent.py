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

The most important rule: answer only from the company documents. Call read_doc on the
relevant document BEFORE answering any factual question. Read more than one if the
question touches more than one topic. Never guess a price, time, distance, age limit,
or policy.

The documents and what each covers:
- tours.md: every tour with price, length, walking, best-for, schedule; which tour to pick
- faq.md: booking, group size limits, discounts, parking, dogs, rain, how scary
- policies.md: refunds, cancellations, late arrival, accessibility, gift cards
- about.md: address, phone, season, who we are

If the documents do not contain the answer: say so plainly, offer to pass the question
to the owner, ask for the visitor's email if you don't have it, and call the log_inquiry
tool in that same turn. Calling the tool is the only way the owner finds out. Never say
"I'll log this" or "I'll pass this along" without actually calling log_inquiry.

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


# Phrases that mean "I'm passing this to the owner". If the model says one of these
# without having called log_inquiry, the loop sends it back once to actually do it.
PROMISE_WORDS = ("log", "pass", "forward", "noted", "follow up", "follow-up", "owner will", "reach out")


def promised_without_logging(text: str, logged: bool) -> bool:
    """True if the reply promises a follow-up but no log_inquiry call happened this turn."""
    lowered = text.lower()
    return not logged and any(word in lowered for word in PROMISE_WORDS)


def run_agent(messages: list[dict], verbose: bool = True) -> str:
    """The agent loop. Keeps going until the model replies with words instead of a tool call."""
    read_any = False        # did the model read at least one document this turn?
    logged = False          # did the model call log_inquiry during this turn?
    nudges_left = 2         # at most two reminders per turn, so the loop always ends
    for _ in range(MAX_STEPS):
        reply = ask_model(messages)
        messages.append(reply)                       # the model's turn goes into the history

        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:                           # no tools requested: this is the answer
            text = reply.get("content", "").strip()

            # Check 1: answered without reading anything. Small models skip this step and
            # then invent facts, so send the question back once with a reminder.
            if not read_any and nudges_left:
                nudges_left -= 1
                if verbose:
                    print("   [check] answered without reading any document; asking it to read first")
                messages.append({"role": "user", "content":
                    "(system check) You answered without calling read_doc. Read the relevant "
                    "document(s) first, then answer again using only what they say."})
                continue

            # Check 2: promised to pass the question along but never called log_inquiry.
            if promised_without_logging(text, logged) and nudges_left:
                nudges_left -= 1
                if verbose:
                    print("   [check] promised a follow-up without calling log_inquiry; asking again")
                messages.append({"role": "user", "content":
                    "(system check) You said you would pass this along, but you did not call "
                    "log_inquiry. Call it now with the visitor's question, email, and details, "
                    "then repeat your answer."})
                continue
            return text

        for call in tool_calls:                      # run each requested tool, add its result
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if verbose:
                print(f"   [tool] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            func = TOOL_FUNCTIONS.get(name)
            result = func(**args) if func else f"Unknown tool {name}"
            if name == "log_inquiry":
                logged = True
            if name == "read_doc":
                read_any = True
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
    import urllib.error
    try:
        main()
    except urllib.error.URLError:
        sys.exit("Could not reach Ollama at localhost:11434. Is it running? Try: ollama serve")
