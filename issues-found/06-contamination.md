# 06: Contamination

- **Experiment:** PEFT methods head-to-head (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Contamination: AG News is a classic public dataset likely in pretraining; zero-shot is 73.3% (depressed by the multi-token Sci/Tech scoring), so contamination less of an issue for relative comparisons.

## Status
Open, documented. Nothing in the results was changed because of it. The [writeup](../writeup.md) states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
