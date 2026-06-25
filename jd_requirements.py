"""
Structured encoding of job_description.docx for the Redrob hackathon.

This file is the output of *reading the JD carefully*, not an algorithm.
Every list below traces back to a specific sentence in the JD. Keeping this
as an explicit, editable config (rather than baking it into rank.py) means
you can defend every weight in the Stage 5 interview by pointing at this file.
"""

# Full JD text used for the TF-IDF "semantic" similarity component.
# (Trimmed to the substantive parts; boilerplate headers removed.)
JD_TEXT = """
Senior AI Engineer, Founding Team, Redrob AI, Series A AI-native talent
intelligence platform. Own the intelligence layer of the product: ranking,
retrieval, and matching systems. Production experience with embeddings-based
retrieval systems such as sentence-transformers, OpenAI embeddings, BGE, E5,
deployed to real users, handling embedding drift, index refresh, retrieval
quality regression in production. Production experience with vector databases
or hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Strong Python and code quality. Hands-on
experience designing evaluation frameworks for ranking systems: NDCG, MRR,
MAP, offline to online correlation, AB test interpretation. LLM fine-tuning
LoRA QLoRA PEFT is a plus. Learning to rank models XGBoost or neural is a
plus. Background in distributed systems or large scale inference
optimization is a plus. Has shipped at least one end to end ranking, search,
or recommendation system to real users at meaningful scale. Strong opinions
about retrieval hybrid versus dense, evaluation offline versus online, LLM
integration fine tune versus prompt, defended with reference to systems
actually built. Six to eight years total experience, four to five years in
applied ML AI roles at product companies, not pure services. Located in or
willing to relocate to Noida or Pune, India.
"""

# Skills that map directly to "things you absolutely need"
MUST_HAVE_SKILLS = [
    "embeddings", "retrieval", "vector database", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "sentence-transformers", "bge", "e5",
    "hybrid search", "ranking", "recommendation system",
    "evaluation framework", "ndcg", "mrr", "map", "a/b testing",
    "python",
]

# Skills explicitly called out as "nice to have, not a blocker"
NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "learning to rank", "xgboost",
    "distributed systems", "inference optimization", "open source",
    "hr-tech", "recruiting tech",
]

# Title words suggesting a *real* AI/ML/Search/Data role (used to sanity
# check that "skills" aren't just decorative keyword stuffing on an
# unrelated title -- this is the JD's "Marketing Manager with AI keywords"
# trap, made explicit).
LEGITIMATE_TITLE_KEYWORDS = [
    "ml engineer", "machine learning", "ai engineer", "applied scientist",
    "research engineer", "data scientist", "search", "ranking",
    "recommendation", "nlp engineer", "ai researcher", "ml scientist",
    "platform engineer", "backend engineer", "software engineer",
    "data engineer", "founding engineer", "staff engineer",
    "principal engineer", "tech lead", "engineering manager",
]

# Titles that strongly suggest NOT a technical IC role, even if skills list
# is keyword-stuffed (HR/marketing/sales/support roles tagged with ML buzzwords)
NON_TECHNICAL_TITLE_KEYWORDS = [
    "hr manager", "human resources", "marketing manager", "sales",
    "customer support", "customer success", "recruiter", "talent acquisition",
    "operations manager", "business development", "content writer",
    "account manager", "product marketing",
]

# Disqualifiers, taken almost verbatim from the JD's "things we explicitly
# do NOT want" + "we will not move forward" sections.
CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]

NON_NLP_DOMAINS = ["computer vision", "speech recognition", "robotics", "image classification"]

PREFERRED_LOCATIONS_TIER1 = ["pune", "noida"]
PREFERRED_LOCATIONS_TIER2 = ["hyderabad", "mumbai", "delhi", "ncr", "gurugram", "gurgaon", "bangalore", "bengaluru"]

EXPERIENCE_BAND = (5, 9)       # years; soft band, JD says flexible if other signals strong
IDEAL_NOTICE_PERIOD_DAYS = 30  # JD: "we'd love sub-30-day notice"

# Component weights for the final composite score. These are a judgment call,
# made explicit and editable -- not hidden inside code. The relative weights
# follow directly from how heavily the JD itself emphasizes each axis
# (skills + career fit dominate; behavioral signals are a modifier, not a
# primary driver, per the JD's own framing of "down-weight appropriately").
WEIGHTS = {
    "semantic_similarity": 0.20,   # TF-IDF cosine vs JD text
    "skill_match": 0.25,           # trust-weighted required-skill coverage
    "title_career_fit": 0.25,      # is the *role history* actually this kind of work
    "experience_fit": 0.10,        # closeness to 5-9 yr band
    "location_fit": 0.05,          # Pune/Noida preferred, other metros ok
    "behavioral_modifier": 0.15,   # availability/responsiveness/recency
}
