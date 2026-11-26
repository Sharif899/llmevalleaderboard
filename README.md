# llm-eval-leaderboard

A public leaderboard of live LLM evaluation results, with real
results from real models, not another toolkit. Models are run
against two benchmarks and scored with a deterministic five-criterion
rubric (factual accuracy, reasoning quality, instruction adherence,
helpfulness, safety), 1-5 per criterion, pass at mean 4.0 or higher.
There is no judge model and no mock data in the results: every
number in the table below comes from a live API call whose raw
response is saved under `results/`.

## Leaderboard

<!-- LEADERBOARD:START -->

| Model | Overall mean | Pass rate | Yoruba mean | General mean | Categories | 95% CI | Run date | Model version |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| gemini-3.5-flash-lite | 3.820 | 0.490 | 3.750 | 3.925 | factual 4.200, factual_qa 4.540, instruction_following 4.180, maths_reasoning 3.860, reasoning 4.050, safety 3.050, translation 3.040 | [3.682, 3.944] | 2026-11-26 | gemini-3.5-flash-lite |
| openai/gpt-oss-120b | 3.804 | 0.470 | 3.490 | 4.275 | factual 4.120, factual_qa 4.040, instruction_following 4.010, maths_reasoning 3.880, reasoning 3.860, safety 3.700, translation 2.860 | [3.670, 3.934] | 2026-11-26 | openai/gpt-oss-120b |

<!-- LEADERBOARD:END -->

The table is generated, never hand-edited. Each row shows the
overall mean, pass rate, per-benchmark mean, per-category mean, the
95 percent confidence interval on the mean, the run date and the
model version. Per-model reports live under `results/models/`.

Reading the safety numbers: the safety criterion itself is 5 on
every safety item for both models, because both models refused all
20 harmful requests. The safety category mean is lower because it
is the mean of the items' overall means, and the recall-based
criteria compare the refusal against a reference refusal in a
specific language, so a refusal in English on a Yoruba item scores
lower on factual accuracy and helpfulness. The safety criterion,
not the category mean, is the measure of refusal. Full detail in
`docs/METHODOLOGY.md`, including a scorer fix applied to the safety
scoring on 2026-11-26 (the four runs were re-scored from the
unchanged collected responses).

## Models

| Model | Provider | Status |
| --- | --- | --- |
| gemini-3.5-flash-lite | Gemini | Live: run with `GEMINI_API_KEY` set |
| openai/gpt-oss-120b | Groq | Live: run with `GROQ_API_KEY` set |

The Groq model is `openai/gpt-oss-120b`, a 120B open model. The
earlier plan named `llama-3.3-70b-versatile`, but Groq retired it:
it returns 404 on this account and is not available, so it was
replaced. The chat-capable models listed for this account are
`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` and
`allam-2-7b`.

Run Gemini (live now):

```bash
export GEMINI_API_KEY="your key here"
lbleader run --provider gemini --benchmark all --run-id 2026-11-26-gemini-3-5-flash-lite --out results
```

Run Groq:

```bash
export GROQ_API_KEY="your key here"
lbleader run --provider groq --benchmark all --run-id 20261126-openai-gpt-oss-120b --out results
```

Temperature is 0 for every run, so results should be
near-deterministic. API keys are read from environment variables
only and are never stored in this repo.

## Benchmarks

- **Yoruba** (60 items, 6 categories x 10): factual QA, maths
  reasoning, instruction following, safety, translation and
  reasoning, from Sharif899/yoruba-llm-benchmark. See
  `data/README.md` for the category breakdown and the dataset
  SHA-256.
- **General** (40 items): an original general set in
  `data/general.jsonl`, written separately. Its details are in
  `data/README.md` when the set lands.

## How to reproduce

```bash
pip install -e .[dev]
export GEMINI_API_KEY="your key here"
lbleader run --provider gemini --benchmark yoruba --run-id 2026-11-26-gemini-3-5-flash-lite --out results
lbleader report --results results
```

Other commands:

```bash
lbleader score --responses responses.jsonl --benchmark yoruba
lbleader compare --results results
```

A re-run with the same run id resumes from `state.json` and skips
items already scored. Failed calls are recorded as failures, never
faked. The exact command used is stored in each run's metadata.

## Methodology summary

Each response is scored 1-5 on five criteria by explicit,
hand-auditable rules: recall against the reference for factual
accuracy (capped at 2 on entity substitution or a wrong final
number), step-by-step reasoning when the prompt asks for it,
explicit format constraints for instruction adherence, completeness
for helpfulness, and refusal of harmful requests for safety. A
response passes at mean 4.0 or higher; a pairwise gap under 0.3 is
a tie. The 95 percent CI on a mean is a percentile bootstrap, 1000
resamples, seed 42. Full detail, including limitations and the
reproduction steps, is in `docs/METHODOLOGY.md`.

## Limitations

- 100 items can only separate models this far apart; finer
  differences need more items.
- The models are free tiers, not the vendors' flagship models.
- The deterministic rubric is a triage signal for human review, not
  a verdict, and no response here has been graded by a human.
- Each model is run once per benchmark at temperature 0.

## Running the tests

```bash
python -m pytest -q
```

Tests run offline on the standard library plus pytest. No test
needs an API key or network access.
