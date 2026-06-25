"""
rank.py -- produces the top-100 ranked submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design notes:

1. Compute constraints (5 min, CPU-only, no network/GPU): the only
   "global" pass needed is computing document frequency for TF-IDF (idf
   weights). Everything else -- similarity to the JD, skill/title/
   behavioral scoring -- is computed per-candidate, independently, while
   streaming the file. No per-candidate API calls, no GPU.

2. Memory: rather than loading all 100K candidates into memory and then
   sorting, we stream the file twice (once to build idf stats, once to
   score) and keep only a fixed-size top-N heap in memory at any time.
   This keeps memory roughly constant regardless of dataset size --
   the same pattern large-scale ranking systems use to avoid materializing
   a full scored dataset just to keep the top results.
"""

import argparse
import csv
import heapq
import json
import sys
import time

import tfidf
import jd_requirements as req
import features as feat


def iter_candidates(path):
    """Stream candidates.jsonl one record at a time, instead of loading
    the whole file into a list. Generators like this are how you process
    files larger than RAM in Python -- each `yield` hands back exactly one
    record and waits, rather than building the full list up front."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_candidate(candidate, semantic_sim):
    signals = candidate.get("redrob_signals", {})

    is_honeypot, honeypot_flags = feat.detect_honeypot(candidate)

    skill_score, matched_skills, skill_reason = feat.skill_match_score(candidate)
    title_score, title_notes = feat.title_career_fit_score(candidate)
    exp_score = feat.experience_fit_score(candidate["profile"].get("years_of_experience", 0))
    loc_score = feat.location_fit_score(candidate)
    behav_score, behav_notes = feat.behavioral_modifier(signals)

    w = req.WEIGHTS
    composite = (
        w["semantic_similarity"] * semantic_sim
        + w["skill_match"] * skill_score
        + w["title_career_fit"] * title_score
        + w["experience_fit"] * exp_score
        + w["location_fit"] * loc_score
        + w["behavioral_modifier"] * behav_score
    )

    if is_honeypot:
        composite *= 0.05  # near-zero, but not hard-deleted -- keeps debugging easy

    detail = {
        "semantic_sim": round(float(semantic_sim), 3),
        "skill_score": round(skill_score, 3),
        "title_score": round(title_score, 3),
        "exp_score": round(exp_score, 3),
        "loc_score": round(loc_score, 3),
        "behav_score": round(behav_score, 3),
        "matched_skills": matched_skills,
        "title_notes": title_notes,
        "behav_notes": behav_notes,
        "is_honeypot": is_honeypot,
        "honeypot_flags": honeypot_flags,
    }
    return composite, detail


def build_reasoning(candidate, detail):
    p = candidate["profile"]
    bits = [f"{p.get('current_title','')} with {p.get('years_of_experience','?')} yrs"]
    if detail["matched_skills"]:
        bits.append(f"core skills: {', '.join(detail['matched_skills'][:3])}")
    if detail["title_notes"]:
        bits.append(detail["title_notes"][0])
    if detail["behav_notes"]:
        bits.append(detail["behav_notes"][0])
    loc = p.get("location", "")
    if loc:
        bits.append(f"based in {loc}")
    text = "; ".join(bits)
    return text[:300]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()

    # Pass 1: stream the file once just to build idf statistics.
    # build_idf only ever keeps a bounded (max_vocab-sized) dict in memory,
    # not anything per-document.
    texts_pass1 = (feat._text_blob(c) for c in iter_candidates(args.candidates))
    idf, n_docs = tfidf.build_idf(texts_pass1, max_vocab=30000)
    query_weights, query_norm = tfidf.tfidf_vector(req.JD_TEXT, idf)
    print(f"Pass 1 (idf over {n_docs} docs) done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Pass 2: stream the file again, score each candidate, keep only a
    # fixed-size min-heap of the current top-N. heapq.heappush/heappop
    # keep the heap's SMALLEST element accessible in O(log n); when the
    # heap exceeds top_n we pop the smallest, so only the top_n largest
    # composite scores ever survive -- without ever sorting all 100K.
    t1 = time.time()
    heap = []  # entries: (composite, candidate_id, reasoning, is_honeypot)
    counter = 0
    for candidate in iter_candidates(args.candidates):
        sim = tfidf.cosine_sim_to_query(feat._text_blob(candidate), idf, query_weights, query_norm)
        composite, detail = score_candidate(candidate, sim)
        reasoning = build_reasoning(candidate, detail)
        entry = (composite, candidate["candidate_id"], reasoning, detail["is_honeypot"])

        if len(heap) < args.top_n:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
        counter += 1

    print(f"Pass 2 (scored {counter} candidates) done in {time.time()-t1:.1f}s", file=sys.stderr)

    # heap currently holds the top_n entries in arbitrary heap order;
    # sort descending by score, with candidate_id ascending as the
    # deterministic tiebreak (required by submission_spec section 3).
    top = sorted(heap, key=lambda e: (-e[0], e[1]))

    raw_scores = [e[0] for e in top]
    max_s, min_s = max(raw_scores), min(raw_scores)
    span = (max_s - min_s) or 1.0

    rows = []
    for composite, cid, reasoning, is_honeypot in top:
        norm_score = round(0.4 + 0.59 * (composite - min_s) / span, 4)  # 0.4-0.99 band
        rows.append((cid, norm_score, reasoning, is_honeypot))

    # Re-sort using the FINAL ROUNDED score so ties resolve identically
    # to how the validator will read the file.
    rows.sort(key=lambda r: (-r[1], r[0]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, norm_score, reasoning, is_honeypot) in enumerate(rows, start=1):
            writer.writerow([cid, rank, f"{norm_score:.4f}", reasoning])

    n_honeypots_in_top = sum(1 for r in rows if r[3])
    print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    print(f"Honeypots in top {args.top_n}: {n_honeypots_in_top}", file=sys.stderr)
    print(f"Total time: {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
