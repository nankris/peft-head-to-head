# 09 (101): Headline "accuracy per GPU-minute" is a derived ratio whose ranking QLoRA≈LoRA>Full>DoR...

- **Experiment:** 101 (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `101-peft-head-to-head.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Headline "accuracy per GPU-minute" is a derived ratio whose ranking QLoRA≈LoRA>Full>DoRA is driven by training time, not accuracy; title "QLoRA gave best accuracy per GPU-minute" is honestly downgraded to a tie in writeup.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
