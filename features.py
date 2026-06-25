"""
Feature extraction for each candidate, given the structured JD requirements.

Each function below returns a score in [0, 1] (or a flag), and a short
plain-English reason string. Keeping the "why" attached to every number is
what lets us build honest, non-hallucinated reasoning text later, and is
also just good engineering practice: a score with no explanation is a
black box you can't debug.
"""

import datetime
import jd_requirements as req


def _text_blob(candidate):
    """Concatenate the parts of a candidate profile that carry semantic
    meaning about *what they actually did*, for TF-IDF comparison against
    the JD. We deliberately weight career history descriptions heavily --
    that's where real signal lives, vs. the skills list which is easiest
    to game."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts)


def detect_honeypot(candidate):
    """Rule-based detector for 'subtly impossible' profiles, per the
    redrob_signals_doc warning. We count independent red flags rather than
    relying on a single rule, since the doc says these are *subtle* --
    a single weird data point could be noise, but two or more together
    is a strong signal of a constructed trap.

    Concept: this is a simple ensemble of heuristics, the same idea behind
    fraud-detection systems -- no single rule catches everything, but
    several weak rules together catch a lot.
    """
    flags = []
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    # Flag 1: claimed total experience wildly mismatches career history sum
    total_months_claimed = profile.get("years_of_experience", 0) * 12
    total_months_listed = sum(job.get("duration_months", 0) for job in history)
    if total_months_claimed > 0 and total_months_listed > 0:
        ratio = total_months_listed / total_months_claimed
        if ratio < 0.4 or ratio > 2.5:
            flags.append("experience_years_mismatch")

    # Flag 2: "expert" proficiency claimed with ~0 months actually using the skill
    zero_duration_expert_skills = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 0) <= 1
    ]
    if len(zero_duration_expert_skills) >= 3:
        flags.append("expert_skills_with_no_time_used")

    # Flag 3: overlapping full-time roles that can't both be true
    # (two roles marked is_current=True, or start dates overlapping by
    # more than a couple of months across different companies)
    current_roles = [j for j in history if j.get("is_current")]
    if len(current_roles) > 1:
        flags.append("multiple_current_jobs")

    # Flag 4: education end_year before start_year, or finishing before born
    # (we don't have birth year, so just check internal consistency)
    for edu in education:
        if edu.get("end_year") and edu.get("start_year") and edu["end_year"] < edu["start_year"]:
            flags.append("education_dates_invalid")
            break

    # Flag 5: career history date ordering broken (a later-listed job
    # starts before an earlier-listed job ends, for non-overlapping orgs,
    # repeatedly)
    parsed = []
    for job in history:
        try:
            start = datetime.date.fromisoformat(job["start_date"])
            parsed.append((start, job))
        except Exception:
            pass
    parsed.sort(key=lambda x: x[0])
    overlap_count = 0
    for i in range(len(parsed) - 1):
        end_i = parsed[i][1].get("end_date")
        if end_i:
            try:
                end_date = datetime.date.fromisoformat(end_i)
                if end_date > parsed[i + 1][0]:
                    overlap_count += 1
            except Exception:
                pass
    if overlap_count >= 2:
        flags.append("overlapping_employment_history")

    return len(flags) >= 2, flags


def skill_match_score(candidate):
    """Trust-weighted coverage of MUST_HAVE_SKILLS.

    Concept: a skill tag by itself is weak evidence -- anyone can list
    'RAG'. We weight each matched skill by endorsements and duration_months,
    so a skill someone has *actually used* for a while counts more than a
    skill with 0 endorsements and 0 duration. This single multiplier is
    the main defense against keyword stuffing at the skills level.
    """
    must = set(s.lower() for s in req.MUST_HAVE_SKILLS)
    nice = set(s.lower() for s in req.NICE_TO_HAVE_SKILLS)
    skills = candidate.get("skills", [])

    matched_must = []
    trust_total = 0.0
    for s in skills:
        name = s.get("name", "").lower()
        hit = any(m in name or name in m for m in must)
        nice_hit = any(n in name or name in n for n in nice)
        if hit or nice_hit:
            duration = s.get("duration_months", 0)
            endorsements = s.get("endorsements", 0)
            proficiency_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(
                s.get("proficiency", "beginner"), 0.3
            )
            trust = proficiency_weight * min(1.0, duration / 24) * min(1.0, 0.3 + endorsements / 10)
            weight = 1.0 if hit else 0.4
            trust_total += trust * weight
            if hit:
                matched_must.append(s["name"])

    # normalize: cap denominator so having a handful of strong matches
    # can already reach a high score (we don't expect anyone to have all
    # ~20 must-have skills -- the JD explicitly says it doesn't expect
    # many close matches in the pool)
    score = min(1.0, trust_total / 4.0)
    reason = f"{len(matched_must)} core skill(s) with real usage history" if matched_must else "no verified core skills"
    return score, matched_must, reason


def title_career_fit_score(candidate):
    """Does the *role history*, not just current title, look like the kind
    of work this JD describes? Also applies the JD's explicit disqualifiers.

    Concept: this is the rule-based half of the hybrid system, encoding
    domain knowledge a pure embedding model would miss (e.g. "consulting-only
    career" is a structural pattern, not a semantic-similarity pattern).
    """
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    current_title = profile.get("current_title", "").lower()
    titles = [j.get("title", "").lower() for j in history]
    companies = [j.get("company", "").lower() for j in history]
    descriptions = " ".join(j.get("description", "").lower() for j in history)

    score = 0.0
    notes = []

    legit_hits = [t for t in req.LEGITIMATE_TITLE_KEYWORDS if any(t in title for title in [current_title] + titles)]
    non_tech_hits = [t for t in req.NON_TECHNICAL_TITLE_KEYWORDS if t in current_title]

    if non_tech_hits and not legit_hits:
        score = 0.05
        notes.append(f"current title '{profile.get('current_title')}' is non-technical")
        return score, notes

    if legit_hits:
        score += 0.5
        notes.append("technical title history")

    # production-deployment language vs pure-research language
    prod_terms = ["deployed", "production", "shipped", "real users", "scale", "users", "latency"]
    research_terms = ["research lab", "academic", "thesis", "publication", "phd research"]
    if any(t in descriptions for t in prod_terms):
        score += 0.3
        notes.append("evidence of production deployment")
    if any(t in descriptions for t in research_terms) and not any(t in descriptions for t in prod_terms):
        score -= 0.3
        notes.append("research-only signal with no production evidence (JD disqualifier)")

    # consulting-only disqualifier: ALL companies are consulting firms
    if companies and all(any(c in comp for c in req.CONSULTING_FIRMS) for comp in companies):
        score -= 0.4
        notes.append("entire career at consulting firms only (JD disqualifier)")

    # CV/speech/robotics without NLP/IR exposure
    domain_text = descriptions + " " + " ".join(s.get("name", "").lower() for s in candidate.get("skills", []))
    if any(d in domain_text for d in req.NON_NLP_DOMAINS) and not any(
        k in domain_text for k in ["nlp", "retrieval", "search", "ranking", "embedding"]
    ):
        score -= 0.2
        notes.append("CV/speech/robotics background without NLP/IR exposure")

    return max(0.0, min(1.0, score)), notes


def experience_fit_score(years):
    lo, hi = req.EXPERIENCE_BAND
    if lo <= years <= hi:
        return 1.0
    distance = min(abs(years - lo), abs(years - hi))
    return max(0.0, 1.0 - distance / 8.0)


def location_fit_score(candidate):
    loc = candidate["profile"].get("location", "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    if any(p in loc for p in req.PREFERRED_LOCATIONS_TIER1):
        return 1.0
    if any(p in loc for p in req.PREFERRED_LOCATIONS_TIER2):
        return 0.75
    if candidate["profile"].get("country", "").lower() == "india" and willing:
        return 0.5
    if willing:
        return 0.3
    return 0.1


def behavioral_modifier(signals):
    """Combine platform-activity signals into a single multiplier-like
    score in [0,1]. Concept: this models the JD's own instruction --
    'a perfect-on-paper candidate who hasn't logged in for 6 months... is
    not actually available. Down-weight them appropriately.' We treat this
    as a *modifier*, not a primary score, matching how the JD frames it
    (a secondary adjustment, not the main basis for fit).
    """
    if not signals:
        return 0.3, ["no behavioral signals"]

    notes = []
    score = 0.0
    weight_sum = 0.0

    def add(value, weight, note=None):
        nonlocal score, weight_sum
        score += value * weight
        weight_sum += weight
        if note:
            notes.append(note)

    # recency of activity
    try:
        last_active = datetime.date.fromisoformat(signals["last_active_date"])
        days_inactive = (datetime.date(2026, 6, 25) - last_active).days
        recency = max(0.0, 1.0 - days_inactive / 180)
        add(recency, 0.3)
        if days_inactive > 150:
            notes.append(f"inactive for {days_inactive} days")
    except Exception:
        add(0.3, 0.3)

    add(signals.get("recruiter_response_rate", 0.3), 0.25)
    add(1.0 if signals.get("open_to_work_flag") else 0.4, 0.2)
    add(signals.get("interview_completion_rate", 0.5), 0.15)

    notice = signals.get("notice_period_days", 60)
    notice_score = 1.0 if notice <= req.IDEAL_NOTICE_PERIOD_DAYS else max(0.2, 1.0 - (notice - 30) / 150)
    add(notice_score, 0.1)
    if notice > 60:
        notes.append(f"long notice period ({notice} days)")

    final = score / weight_sum if weight_sum else 0.3
    if signals.get("recruiter_response_rate", 0) > 0.7:
        notes.append("highly responsive to recruiters")
    return final, notes
