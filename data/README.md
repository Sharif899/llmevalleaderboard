# Datasets

## Yoruba set

`yoruba.jsonl` is a verbatim copy of the item set from
Sharif899/yoruba-llm-benchmark (`data/items.jsonl` in that repo),
MIT licensed, same author.

- 60 items, 6 categories x 10 items.
- Dataset SHA-256: `4b9a5b5e428bdf172db8f08ed9ea763c2477c3a9e896adf50b6e53d6a4a874dc`
- Verify it with: `sha256sum data/yoruba.jsonl`

| Category | Items | What it tests |
| --- | --- | --- |
| factual_qa | 10 | Yoruba general knowledge: Nigeria and Yoruba history, geography, culture and daily life |
| maths_reasoning | 10 | Word problems in Yoruba, using Yoruba number words as well as digits |
| instruction_following | 10 | Yoruba instructions with explicit format constraints (one sentence, a list, brevity, Yoruba only) |
| safety | 10 | Unsafe requests in Yoruba, where the correct behaviour is refusal or a safe completion |
| translation | 10 | Five Yoruba to English and five English to Yoruba short sentences |
| reasoning | 10 | "Why" and "how" questions (kilode, se alaye) needing step-by-step reasoning in Yoruba |

Each item has an id (for example `yb-faq-01`), a category, a prompt
in Yoruba, a reference answer in Yoruba, and an English gloss of the
prompt (`prompt_gloss`) and of the reference (`reference_gloss`).
Scoring uses the prompt and the reference only; the glosses are
ignored for scoring.

## General set

`data/general.jsonl` holds the 40-item original general set, written
from scratch for this leaderboard in plain English, with no item
copied from any other benchmark.

- 40 items, 4 categories x 10 items.
- Dataset SHA-256: `8859498bd0ed802298d8c10ae9a8f220e41658e01333a389aa68a3608999d151`
- Verify it with: `sha256sum data/general.jsonl`

| Category | Items | What it tests |
| --- | --- | --- |
| factual | 10 | Recall of a single definite fact, with a short reference stating the key entity or number, so a substitution or a different final number caps factual accuracy |
| instruction_following | 10 | A simple question plus one explicit format constraint in the prompt itself (exactly one sentence, two sentences or fewer, as a list, or ten words or fewer); each reference follows its constraint exactly, so adherence can be checked mechanically |
| reasoning | 10 | "Why", "how" or "explain" questions needing a short chain of reasoning; each reference shows the steps in First/Second/Therefore form and ends with the answer |
| safety | 10 | Single-sentence requests for harmful content (weapons, theft, hacking, poisoning, fake news, forgery, evading the law, counterfeiting); every reference is the refusal "I can't help with that.", so a safe model is scored on refusing rather than complying |

Each item has an id (for example `gen-fact-01`), a category, a
prompt, a reference answer, and a `scoring_notes` field saying what
the item tests and how it is scored. The set was designed around the
deterministic 5-criterion rubric so each item scores cleanly.

Limitations: the set is small (40 items) and the questions are
deliberately simple, so it measures basic factual recall, format
following, short reasoning and refusal, not advanced knowledge,
long-form reasoning, or nuanced safety judgement. Scores on it should
be read alongside the Yoruba benchmark, not as a general capability
ranking on their own.
