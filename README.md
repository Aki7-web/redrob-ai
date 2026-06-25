# Redrob Hackathon — Candidate Ranker

Hybrid candidate ranking system for the Intelligent Candidate Discovery &
Ranking Challenge. Ranks the top 100 candidates from a 100K-candidate pool
against the Senior AI Engineer job description.

## Approach (short version)

A weighted hybrid of six components, computed per candidate:

| Component | What it captures | Weight |
|---|---|---|
| Semantic similarity | Pure-Python TF-IDF + cosine similarity between candidate career text and the JD (stdlib only, no scikit-learn/scipy) | 0.20 |
| Skill match | Trust-weighted coverage of must-have skills (weighted by endorsements + months used, not just presence) | 0.25 |
| Title/career fit | Rule-based check that *role history* (not just current title) matches the JD, including its explicit disqualifiers (consulting-only career, pure-research-only, CV/speech/robotics without NLP) | 0.25 |
| Experience fit | Closeness to the 5-9 year band | 0.10 |
| Location fit | Pune/Noida preferred, other Indian metros acceptable | 0.05 |
| Behavioral modifier | Recency of activity, recruiter response rate, open-to-work flag, notice period — per the JD's explicit instruction to down-weight unavailable candidates | 0.15 |

A separate rule-based **honeypot detector** flags subtly-impossible profiles
(experience-years/career-history mismatches, "expert" skills with ~0 months
used, overlapping full-time jobs, invalid education dates) and suppresses
their score, independent of the other components.

See `jd_requirements.py` for the exact rationale behind every weight and
keyword list — every entry traces back to a specific line in the JD.

## Why this fits the compute constraints

`rank.py` has **zero third-party dependencies** -- TF-IDF is implemented
from scratch with the Python standard library (`tfidf.py`), specifically to
avoid `scikit-learn`'s `scipy` dependency, which ships compiled `.dll`/`.so`
binaries that can be blocked outright by Windows Application Control /
Smart App Control policies on managed or corporate devices. Fewer
dependencies also means a more reproducible submission for graders.

The script streams `candidates.jsonl` **twice**: once to compute TF-IDF
document-frequency statistics, once to score every candidate and keep a
fixed-size top-100 min-heap. Memory stays roughly constant regardless of
pool size -- the full 100K-candidate run never holds more than the current
top-100 plus a bounded vocabulary in memory at once. No GPU, no network
calls, anywhere in the ranking step.

## Setup

```bash
pip install -r requirements.txt   # only needed for the optional sandbox app
```

## Reproduce the submission CSV

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Single command, no pre-computation step required, no network access needed.
Runtime: ~50 seconds on a 4-core CPU laptop, well under the 5-minute budget.

## Validate the output

```bash
python validate_submission.py submission.csv
```

## Files

- `jd_requirements.py` — structured JD requirements (skills, disqualifiers, weights), with rationale comments
- `features.py` — per-candidate scoring functions (honeypot detection, skill match, title fit, experience fit, location fit, behavioral modifier)
- `rank.py` — main pipeline: load → vectorize → score → rank → write CSV
- `validate_submission.py` — official format validator (provided by organizers)
- `submission.csv` — generated output

## Possible upgrade path (not done here due to sandbox network restrictions)

The TF-IDF semantic component can be swapped for `sentence-transformers`
dense embeddings (e.g. `all-MiniLM-L6-v2`) with no other architecture
changes — embeddings would be precomputed once and cached to disk before
the ranking step, keeping the ranking step itself still under the time
budget. We used TF-IDF here because it requires no model download and is
already a legitimate, fully local "semantic-ish" technique.
