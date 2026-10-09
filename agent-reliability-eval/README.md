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
- [x] System 1: basic RAG
- [x] Scoring pipeline and experiment runner
- [ ] System 2: agent + tools
- [ ] System 3: agent + verification
- [ ] Expand to 100–200 questions
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
pip install -e ".[dev]"
python dataset/validate.py     # expect: 15/15 questions valid
pytest                         # offline tests, no API key needed
```

This installs `cyber-intel-mcp` straight from GitHub.

### Add your API key

Copy `.env.example` to `.env` and paste your Anthropic API key after `ANTHROPIC_API_KEY=`. `.env` is git-ignored, so the key never reaches GitHub.

## Running experiments

```bash
python -m reliability.run --system rag --fake          # dry run: fake model, no key, no cost
python -m reliability.run --system rag                 # Claude Haiku 5.5
python -m reliability.run --system rag --ids q001 q009 # just some questions
python -m reliability.scoring results/<file>.jsonl     # re-score a saved run
```

Each run saves every question's answer, citations, raw model reply, tool calls, tokens, cost and time to `results/`, then prints a report.

### What gets measured

| Metric | How |
|---|---|
| Accuracy | Deterministic match against the gold answer (rules in `dataset/SCHEMA.md`), overall and per category |
| Partial credit | Fraction of parts correct in multi-part answers |
| Hallucination rate | Unanswerable questions where the system gave an answer anyway |
| Wrong abstention rate | Answerable questions where the system said "not available" |
| Citation precision / recall | Cited ids that are gold sources / gold sources that were cited |
| Fabricated citations | Cited ids that do not exist in the data at all |
| Cost | Tokens × published price per model (`reliability/llm.py`) |
| Latency | Wall-clock seconds per question, p50 and p95 |

**Same model, same prompt contract, same tools for every system**, so differences come from the architecture. Claude 5.5 models don't accept a temperature setting, so outputs vary between runs. Run each configuration several times and report the spread.

## Systems

- **`rag`**: BM25-searches the corpus with the question, fetches the top 8 records in full, and makes one model call. No tools, no loop.
