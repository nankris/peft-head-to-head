# 09: Headline "accuracy per GPU-minute" is a derived ratio whose ranking QLoRA≈LoRA>Full>DoR...

- **Experiment:** PEFT methods head-to-head (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Headline "accuracy per GPU-minute" is a derived ratio whose ranking QLoRA≈LoRA>Full>DoRA is driven by training time, not accuracy; title "QLoRA gave best accuracy per GPU-minute" is honestly downgraded to a tie in writeup.

## Status
Open, documented. Nothing in the results was changed because of it. The [writeup](../writeup.md) states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
