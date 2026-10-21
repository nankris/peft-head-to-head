# 05 (101): Wall-clock/efficiency metric spans 4 VMs, and seed-1236 runs used expandable_segments

- **Experiment:** 101 (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `101-peft-head-to-head.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Wall-clock/efficiency metric spans 4 VMs, and seed-1236 runs used expandable_segments; per-arm times vary <3% so OK, but DoRA's 2.6× slowdown is PEFT-implementation-specific (unmerged DoRA).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
