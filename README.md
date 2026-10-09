# agent-reliability-eval

**Research question: Can AI agents reliably complete multi-step research tasks?**

This repo compares three system designs on the same cybersecurity research questions:

1. **Basic RAG**: retrieve relevant records, then answer in one LLM call
2. **Agent + tools**: an LLM that calls tools in a loop
3. **Agent + verification**: the same agent, plus a step that checks each claim against its cited source before answering

Each is scored on **accuracy, citation correctness, cost and latency**, followed by an analysis of how and why they fail.

All three systems get their data from [cyber-intel-mcp](https://github.com/Jody2905/cyber-intel-mcp), an MCP server over a frozen snapshot of CVE, CISA KEV and MITRE ATT&CK data. Freezing the data means every gold answer can be checked and every result reproduced.

## Status

- [x] Tool layer: [cyber-intel-mcp](https://github.com/Jody2905/cyber-intel-mcp)
- [x] Question schema and 15 seed questions, all validated
- [ ] Expand to 100–200 questions
- [ ] System 1: basic RAG
- [ ] System 2: agent + tools
- [ ] System 3: agent + verification
- [ ] Scoring pipeline
- [ ] Experiments, failure analysis, write-up

## Dataset

`dataset/seed_questions.jsonl` holds 15 questions across four categories:

| Category | Count | Tests |
|---|---|---|
| single_hop | 5 | Looking up one fact |
| multi_hop | 4 | Combining records, date arithmetic, search then lookup |
| comparison | 4 | Counting and ranking across many records |
| unanswerable | 2 | Saying "not available" instead of hallucinating |

Several questions contain deliberate traps where an answer from a model's memory disagrees with the data. For example, most sources give Zerologon (CVE-2020-1472) a CVSS of 10.0, but the CVE record in the snapshot scores it 5.5.

**Gold answers are computed from the snapshot by code** (`dataset/build_seed.py`), never typed by hand. **Every question is validated** (`dataset/validate.py`) by a scripted reference solution that must reach the gold answer using only the MCP tools. See `dataset/SCHEMA.md` for the field definitions and scoring rules.

## Setup

```bash
git clone https://github.com/Jody2905/agent-reliability-eval.git
cd agent-reliability-eval
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e .
python dataset/validate.py
```

This installs `cyber-intel-mcp` straight from GitHub. The expected output is `15/15 questions valid`.
