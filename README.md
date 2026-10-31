# Tour Assistant, local edition

The same visitor assistant as [tour-assistant](https://github.com/nandras19/tour-assistant),
rebuilt as one Python file running against a local model through Ollama.
No cloud, no API key, no packages to install. The point is to show how an agent
loop works when nothing is hidden behind a framework.

## What an agent loop is

An agent is a model in a loop with tools. Each pass through the loop:

1. Send the whole conversation to the model, along with a list of tools it may call.
2. If the model replies with words, that's the answer. Stop.
3. If it replies with a tool call, run that function, add the result to the
   conversation, and go back to step 1.

That's the entire idea. In `agent.py` it is the `run_agent` function, about
twenty lines. Everything else is the system prompt, three tool functions, and
the plumbing that talks to Ollama.

```
user question
     │
     ▼
┌─────────────┐   tool call   ┌──────────────┐
│    model     │ ───────────▶ │  run tool    │
│  (Ollama)    │ ◀─────────── │  (Python)    │
└─────────────┘   result      └──────────────┘
     │
     │ plain-text reply
     ▼
   answer
```

## The three tools

| Tool | What it does | Why the agent needs it |
| --- | --- | --- |
| `list_docs` | names the files in `docs/` | so it knows what it can read |
| `read_doc` | returns one file's text | grounding: answers come from here, not memory |
| `log_inquiry` | appends to `data/inquiries.json` | turns "I don't know" into a lead for the owner |

A tool is just a Python function plus a short JSON description of its name,
purpose, and arguments. The description is what the model reads to decide when
to call it, so the wording matters as much as the code.

## Run it

Requirements: Python 3.11+, [Ollama](https://ollama.com), and a model that
supports tool calling.

```bash
ollama pull qwen3:8b
```

```bash
python agent.py
```

Or one question at a time:

```bash
python agent.py "We have a 6-year-old and my mom uses a cane. Which tour?"
```

To use a different model, change `MODEL` at the top of `agent.py`.

## What's different from the Claude Code version

| | Claude Code version | This version |
| --- | --- | --- |
| Model | Claude, hosted | 8B open model on your own GPU |
| Instructions | `CLAUDE.md` + skills | one system prompt string |
| Tools | MCP server, separate process | plain functions in the same file |
| Loop | inside Claude Code | written by hand in `run_agent` |
| Quality | high; follows multi-step skills reliably | good on direct questions; weaker on long multi-step tasks |
| Cost | API usage | electricity |

The hosted version is what I'd deploy for a client. This version is what I'd
show someone who asked "but how does it actually work?"

## What I learned

- The loop is small. The hard parts are the prompt, the tool descriptions, and
  testing with realistic conversations.
- Small local models follow a grounding rule well when the rule is stated once,
  plainly, and the tool to satisfy it (`read_doc`) is obvious.
- Keeping the whole conversation in a list and resending it every turn is how
  the model "remembers". Nothing is stored on the model side.
