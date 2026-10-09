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
- [x] System 2: agent + tools
- [x] System 3: agent + verification
- [x] Expand to 150 questions (23 templates + 15 seed questions)
- [ ] First real runs
- [ ] Experiments, failure analysis, write-up

## Dataset

`dataset/questions.jsonl` holds **150 questions**:

| Category | Easy | Medium | Hard | Total | Tests |
|---|---|---|---|---|---|
| single_hop | 40 | | | 40 | Looking up one fact |
| multi_hop | | 27 | 13 | 40 | Combining records, date arithmetic, finding a CVE by name and then looking it up |
| comparison | | 26 | 14 | 40 | Counting and ranking across many records |
| unanswerable | | 21 | 9 | 30 | Saying "not available" instead of hallucinating |

The first 15 are hand-written seed questions (`dataset/build_seed.py`). The other 135 come from **23 templates** (`dataset/templates.py`). Each template is one kind of question, filled in from the data with a fixed random seed, so rebuilding always gives the identical file.

Several questions are deliberate traps where a model's memory disagrees with the data:
- Zerologon (CVE-2020-1472) is 10.0 in most sources, but its CVE record scores it 5.5.
- CVE-2021-3449 is widely listed as 5.9, but its CVE record carries no score at all.
- Some unanswerable questions have a false premise, such as asking when a CVE that is not in KEV was added to KEV.

**Every gold answer is computed from the data by code, never typed by hand. Every question is validated** by a reference solution that must reach the gold answer using only the MCP tools. Planted wrong answers are caught. See `dataset/SCHEMA.md` for field definitions and scoring rules.

```bash
python dataset/build.py      # regenerate questions.jsonl (identical output every time)
python dataset/validate.py   # prove all 150 through the MCP tools
```

## Setup

```bash
git clone https://github.com/Jody2905/agent-reliability-eval.git
cd agent-reliability-eval
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python dataset/validate.py     # expect: 150/150 questions valid
pytest                         # offline tests, no API key needed
```

This installs `cyber-intel-mcp` straight from GitHub.

### Add your API key

Copy `.env.example` to `.env` and paste your Anthropic API key after `ANTHROPIC_API_KEY=`. `.env` is git-ignored, so the key never reaches GitHub.

## Running experiments

```bash
python -m reliability.run --system rag --fake          # dry run: fake model, no key, no cost
python -m reliability.run --system rag                 # Claude Haiku 5.5
python -m reliability.run --system agent
python -m reliability.run --system agent_verify
python -m reliability.run --system rag --ids q001 q009 # just some questions
python -m reliability.run --system rag --questions dataset/seed_questions.jsonl  # the 15 seed questions only
python -m reliability.scoring results/<file>.jsonl     # re-score a saved run
```

**Estimated cost with Claude Haiku 5.5:** about $1–2 for one pass of all three systems over the 150 questions, and roughly $5 for the recommended 3 repeats. This is estimated from prompt sizes; agent runs vary.

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

All three use the same model, the same answer format and citation rules, and the same MCP tools.

| System | How it works | Model calls |
|---|---|---|
| `rag` | Code BM25-searches the corpus with the question, fetches the top 8 records in full, and makes one model call. No loop. | 1 |
| `agent` | The model gets the five MCP tools and decides what to call, looping until it answers or uses its budget of 8 tool-call rounds. Then tools are switched off and it must answer. | 2+ |
| `agent_verify` | The `agent`, plus two checks on its answer. **Mechanical** (code): valid JSON, citations present, every cited id exists. **Verifier** (a separate model call with no tools): do the cited records actually support every part of the answer? Problems go back to the agent, which revises with its tools (up to 2 revisions). | 3+ |

Every run saves the agent's full transcript (tool calls, results and reasoning), and for `agent_verify` every draft answer and the verifier's objections. This is the raw material for the failure analysis.

**Design notes**

- The verifier only sees the records the agent cited, so it checks *support*, not truth. It catches answers from memory and answers backed by the wrong record. It cannot catch a well-supported answer to the wrong question.
- Abstentions are not verified, since a verifier can't confirm that something is absent from the data. So verification can reduce hallucinations but not wrong abstentions.
- All three systems are tested offline with scripted fake models (`tests/test_agents.py`), including a verifier catching the classic Zerologon "10.0 from memory" error.
