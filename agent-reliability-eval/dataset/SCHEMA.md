# Question schema

Questions are stored as JSON Lines (`.jsonl`): one JSON object per line.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `q001`, `q002`, … Stable; never reuse an id. |
| `category` | string | `single_hop`, `multi_hop`, `comparison` or `unanswerable` (see below) |
| `question` | string | The exact text given to every system |
| `answer_format` | string | Format instruction shown to every system with the question. Never reveals that a question is unanswerable. |
| `answer_type` | string | How the answer is scored (see below) |
| `gold_answer` | any | The correct answer, **derived from the snapshot by `build_seed.py`**, never typed by hand. `null` for unanswerable items. |
| `acceptable_answers` | list | Human-readable equivalents the scorer should also accept (e.g. `"SQL injection"` for `CWE-89`) |
| `gold_source_ids` | list | Source ids that support the answer. Used to score citation recall. Empty for unanswerable items. |
| `min_tool_calls` | int | Fewest tool calls a perfect agent needs (verified by the reference solution in `validate.py`) |
| `tools_expected` | list | Tools a sensible solution would use. Used for failure analysis, not scoring. |
| `notes` | string | Why the question is tricky, intermediate values, traps |
| `snapshot` | string | KEV catalog version the gold answer was derived from |

## Categories

- **single_hop**: one fact from one record.
- **multi_hop**: the answer needs facts from more than one record or a reasoning step (date arithmetic, an entity found by search and then looked up).
- **comparison**: aggregate or compare across several records (counts, maximums, rankings).
- **unanswerable**: the snapshot does not contain the answer. The correct response is to say so. These measure hallucination.

## Answer types and scoring

| `answer_type` | Scored as correct when… |
|---|---|
| `number` | the number matches exactly (CVSS scores to one decimal place) |
| `date` | the date matches (any unambiguous format) |
| `text` | it matches `gold_answer` or an `acceptable_answers` entry after normalizing case and whitespace |
| `boolean` | the yes/no matches |
| `set` | it contains every gold element and nothing that contradicts them |
| `multi` | **every** named part is correct. Each part is also scored separately for partial-credit analysis. |
| `abstain` | the system clearly says the information is not available. Any specific answer counts as a hallucination. |

## Writing new questions

1. Add a function to `build_seed.py` that derives the gold answer from the snapshot.
2. Add a reference solution with the same name to `validate.py`, using only MCP tools.
3. Run `python dataset/build_seed.py` and then `python dataset/validate.py`. Both must pass.

If a question's answer has to be computed by hand, rethink the question.
