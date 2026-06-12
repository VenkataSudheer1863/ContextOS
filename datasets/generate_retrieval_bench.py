"""
Retrieval Benchmark Dataset Generator for ContextOS Research.

Generates retrieval evaluation data:
  - 25000 train + 2500 val + 2500 test examples
  - Each: {id, query, corpus_items, relevant_ids, irrelevant_ids, metadata}
  - corpus_items: list of 50-100 candidate items
  - relevant_ids: 1-5 truly relevant items
  - Difficulty levels: easy (obvious), medium (paraphrase), hard (indirect)

Only stdlib imports used.
"""

import json
import random
import hashlib
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducible RNG
# ---------------------------------------------------------------------------
SEED = 7

# ---------------------------------------------------------------------------
# Domain + query/document templates
# ---------------------------------------------------------------------------

DOMAINS = ["technology", "science", "business", "medical"]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# ------- Query templates per domain -----------------------------------------

QUERY_TEMPLATES = {
    "technology": [
        "How does {concept} improve {outcome} in {system}?",
        "What are the main challenges in implementing {concept}?",
        "Compare {concept_a} and {concept_b} for {use_case}.",
        "What is the role of {concept} in {system}?",
        "Explain the benefits of {concept} for {audience}.",
        "How can {concept} be used to optimize {outcome}?",
        "What security considerations apply to {concept}?",
        "Describe the architecture of a {system} using {concept}.",
    ],
    "science": [
        "What is the mechanism of {concept} in {system}?",
        "How does {concept} affect {outcome}?",
        "What evidence supports {concept} in the context of {system}?",
        "Describe the relationship between {concept_a} and {concept_b}.",
        "What are the implications of {concept} for {field}?",
        "How is {concept} measured or quantified in {system}?",
        "What are the current research frontiers in {concept}?",
        "Why is {concept} important in understanding {system}?",
    ],
    "business": [
        "How does {concept} impact {outcome} for {audience}?",
        "What strategies are used to improve {concept} in {system}?",
        "Explain the role of {concept} in {use_case}.",
        "What metrics are used to evaluate {concept}?",
        "How do companies implement {concept} to gain competitive advantage?",
        "What are the risks associated with {concept} in {system}?",
        "Compare approaches to {concept} in large and small organizations.",
        "What best practices apply to {concept} in {use_case}?",
    ],
    "medical": [
        "What is the standard treatment for {concept} in {audience}?",
        "How does {concept} affect {outcome} in patients?",
        "What diagnostic criteria are used for {concept}?",
        "Describe the pathophysiology of {concept}.",
        "What are the side effects of {concept} treatment?",
        "How is {concept} managed in {system} settings?",
        "What research supports the use of {concept} for {outcome}?",
        "Explain the difference between {concept_a} and {concept_b} in clinical practice.",
    ],
}

# ------- Concept/entity pools per domain ------------------------------------

DOMAIN_CONCEPTS = {
    "technology": [
        "machine learning", "containerization", "microservices", "encryption",
        "load balancing", "serverless computing", "graph databases", "CI/CD pipelines",
        "Kubernetes", "federated learning", "model quantization", "edge computing",
        "zero-trust security", "WebAssembly", "API gateways", "in-memory caching",
        "blockchain", "quantum computing", "neural networks", "transformer models",
    ],
    "science": [
        "CRISPR gene editing", "quantum entanglement", "dark matter", "neuroplasticity",
        "photosynthesis", "antibiotic resistance", "exoplanet detection", "stem cell therapy",
        "protein folding", "nuclear fusion", "mRNA vaccines", "bioluminescence",
        "plate tectonics", "the Higgs boson", "horizontal gene transfer", "epigenetics",
        "climate modeling", "supernovae nucleosynthesis", "microbiome diversity", "synthetic biology",
    ],
    "business": [
        "supply chain optimization", "customer lifetime value", "agile methodology",
        "brand equity", "ESG investing", "market segmentation", "Porter's Five Forces",
        "blue ocean strategy", "lean manufacturing", "net promoter score",
        "mergers and acquisitions", "digital transformation", "revenue forecasting",
        "intellectual property management", "economies of scale", "crowdsourcing",
        "working capital management", "corporate governance", "price elasticity", "venture capital",
    ],
    "medical": [
        "immunotherapy", "pharmacogenomics", "telemedicine", "precision medicine",
        "sepsis management", "anticoagulation therapy", "organ transplantation",
        "minimally invasive surgery", "Alzheimer's disease", "mRNA therapeutics",
        "clinical trial design", "palliative care", "antibiotic stewardship",
        "neonatal screening", "gene therapy", "biomarker discovery",
        "robotic surgery", "blood-brain barrier", "stroke intervention", "type 2 diabetes",
    ],
}

DOMAIN_SYSTEMS = {
    "technology": [
        "distributed systems", "cloud platforms", "web applications", "mobile apps",
        "data pipelines", "IoT networks", "enterprise software", "real-time systems",
    ],
    "science": [
        "living organisms", "the universe", "ecosystems", "neural tissue",
        "the atmosphere", "bacterial populations", "human cells", "quantum systems",
    ],
    "business": [
        "global enterprises", "start-ups", "retail operations", "financial institutions",
        "healthcare organizations", "manufacturing firms", "tech companies", "non-profits",
    ],
    "medical": [
        "intensive care units", "outpatient clinics", "surgical settings",
        "pediatric wards", "oncology departments", "primary care", "emergency medicine",
    ],
}

DOMAIN_OUTCOMES = {
    "technology": [
        "performance", "scalability", "reliability", "security", "developer productivity",
        "latency reduction", "cost efficiency", "fault tolerance",
    ],
    "science": [
        "disease resistance", "energy efficiency", "biodiversity", "cognitive function",
        "climate stability", "genetic expression", "structural integrity", "reaction rate",
    ],
    "business": [
        "profitability", "customer retention", "market share", "operational efficiency",
        "employee engagement", "innovation rate", "brand loyalty", "risk mitigation",
    ],
    "medical": [
        "patient outcomes", "survival rates", "quality of life", "diagnostic accuracy",
        "treatment adherence", "hospital readmission", "pain management", "recovery time",
    ],
}

DOMAIN_AUDIENCES = {
    "technology": ["software engineers", "data scientists", "DevOps teams", "enterprise architects"],
    "science": ["researchers", "clinicians", "policy makers", "the general public"],
    "business": ["executives", "investors", "small business owners", "supply chain managers"],
    "medical": ["adult patients", "pediatric patients", "elderly populations", "high-risk groups"],
}

# ------- Document sentence pools per domain ---------------------------------

DOC_SENTENCES = {
    "technology": [
        "This approach reduces system latency by processing requests closer to the data source.",
        "Security is enhanced through layered authentication and encrypted communication channels.",
        "The framework provides automatic scaling based on real-time traffic patterns.",
        "Developers can deploy updates without downtime using rolling release strategies.",
        "Resource utilization improves when workloads are distributed across heterogeneous nodes.",
        "Monitoring tools track key metrics to detect anomalies before they escalate.",
        "Data consistency is maintained using consensus algorithms across distributed nodes.",
        "The pipeline processes millions of events per second with sub-millisecond latency.",
        "Integration with existing systems is simplified through standardized API contracts.",
        "Fault tolerance is achieved by replicating state across multiple availability zones.",
        "Testing automation ensures that regressions are caught before reaching production.",
        "Configuration management tools enforce infrastructure as code best practices.",
        "Performance benchmarks show a threefold improvement over the previous architecture.",
        "Access controls restrict sensitive operations to authorized roles only.",
        "The solution complies with data residency and privacy regulations across regions.",
    ],
    "science": [
        "Experimental results confirm the theoretical predictions with high statistical significance.",
        "The organism exhibits adaptive responses to environmental stressors over generations.",
        "Spectroscopic analysis reveals the chemical composition of distant celestial objects.",
        "The interaction between these two pathways regulates cellular homeostasis.",
        "Longitudinal studies track changes in populations over decades to identify trends.",
        "Computer simulations model complex phenomena that are difficult to observe directly.",
        "The discovery challenges previously held assumptions about the fundamental mechanisms.",
        "Peer-reviewed studies replicate these findings across independent laboratories.",
        "The genetic variation correlates with observable phenotypic differences in the study group.",
        "Environmental factors interact with genetic predispositions to determine outcomes.",
        "Isotopic dating places the sample origin at approximately 4.5 billion years ago.",
        "The reaction proceeds through a series of intermediate states before reaching equilibrium.",
        "Imaging techniques reveal structural changes at the nanometer scale.",
        "Field observations align with laboratory findings, strengthening the hypothesis.",
        "The compound shows high binding affinity and selectivity for the target receptor.",
    ],
    "business": [
        "Companies that invest in this strategy report higher customer retention rates.",
        "The analysis identifies three core drivers of profitability in this sector.",
        "Agile teams deliver value faster by reducing cycle times through iterative sprints.",
        "Risk diversification protects portfolios from sector-specific downturns.",
        "Market research reveals unmet demand in the mid-tier consumer segment.",
        "Automation reduces operational costs without sacrificing service quality.",
        "Stakeholder alignment is critical before initiating large-scale organizational change.",
        "Pricing strategy must account for competitive positioning and perceived value.",
        "Cross-functional collaboration breaks down silos and accelerates decision making.",
        "The acquisition created synergies worth approximately $200 million annually.",
        "ESG commitments increasingly influence institutional investor allocation decisions.",
        "Talent retention strategies reduce the high costs associated with employee turnover.",
        "A/B testing enables data-driven decisions about product features and pricing.",
        "The expansion into emerging markets diversifies revenue and reduces dependency.",
        "Regulatory compliance programs protect against fines and reputational damage.",
    ],
    "medical": [
        "Clinical data supports the efficacy of this intervention in reducing mortality.",
        "The treatment protocol is well-tolerated with a manageable side effect profile.",
        "Early diagnosis significantly improves prognosis for patients with this condition.",
        "Biomarkers enable personalized dosing regimens that optimize therapeutic outcomes.",
        "Multidisciplinary teams coordinate care to address the full spectrum of patient needs.",
        "Guidelines recommend screening at-risk populations to facilitate early intervention.",
        "The drug inhibits a key enzyme in the inflammatory pathway, reducing symptoms.",
        "Patient adherence is improved by simplified dosing schedules and counseling programs.",
        "Surgical outcomes depend on both technical proficiency and post-operative management.",
        "Epidemiological data shows declining incidence following widespread vaccination programs.",
        "The mechanism of action differs from existing therapies, offering an alternative pathway.",
        "Health economic analyses demonstrate cost-effectiveness compared to standard care.",
        "Contraindications must be evaluated carefully in patients with comorbid conditions.",
        "Randomized controlled trials remain the gold standard for evaluating new interventions.",
        "Long-term follow-up studies assess durability of response and late adverse effects.",
    ],
}

# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _fill_template(template: str, domain: str, rng: random.Random) -> str:
    concepts = DOMAIN_CONCEPTS[domain]
    systems = DOMAIN_SYSTEMS[domain]
    outcomes = DOMAIN_OUTCOMES[domain]
    audiences = DOMAIN_AUDIENCES[domain]

    concept_a = rng.choice(concepts)
    concept_b = rng.choice([c for c in concepts if c != concept_a]) if len(concepts) > 1 else concept_a
    replacements = {
        "{concept}": rng.choice(concepts),
        "{concept_a}": concept_a,
        "{concept_b}": concept_b,
        "{system}": rng.choice(systems),
        "{outcome}": rng.choice(outcomes),
        "{audience}": rng.choice(audiences),
        "{use_case}": rng.choice(systems),
        "{field}": domain,
    }
    for key, val in replacements.items():
        template = template.replace(key, val)
    return template


def _build_query(domain: str, difficulty: str, rng: random.Random) -> str:
    template = rng.choice(QUERY_TEMPLATES[domain])
    query = _fill_template(template, domain, rng)

    if difficulty == "medium":
        # Paraphrase: add a qualifier
        qualifiers = [
            "In detail, ", "Provide a thorough explanation of how ", "Briefly explain ",
            "From a practical standpoint, ", "From a research perspective, ",
        ]
        prefix = rng.choice(qualifiers)
        query = prefix + query[0].lower() + query[1:]
    elif difficulty == "hard":
        # Indirect: reframe as a problem-solving scenario
        scenarios = [
            "A team is struggling with {outcome}. How might {concept} address this?",
            "An organization wants to improve {outcome}. What role could {concept} play?",
            "Given constraints in {system}, what approach involving {concept} would be most effective?",
            "A researcher needs to understand {outcome} in {system}. What does the literature say about {concept}?",
        ]
        template_h = rng.choice(scenarios)
        query = _fill_template(template_h, domain, rng)

    return query


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def _build_document(domain: str, concept: str, is_relevant: bool,
                     difficulty: str, rng: random.Random) -> str:
    """Build a short document (3-7 sentences)."""
    sentences = DOC_SENTENCES[domain][:]
    rng.shuffle(sentences)
    n_sents = rng.randint(3, 7)
    chosen = sentences[:n_sents]

    if is_relevant:
        # Insert the concept name to make it topically relevant
        intro_templates = [
            f"This document discusses {concept} and its applications.",
            f"{concept.capitalize()} is the focus of this analysis.",
            f"Recent developments in {concept} are reviewed here.",
            f"An overview of {concept} and related considerations follows.",
        ]
        doc = rng.choice(intro_templates) + " " + " ".join(chosen)
    else:
        # Irrelevant: use sentences from a different domain, no concept mention
        other_domains = [d for d in DOMAINS if d != domain]
        alt_domain = rng.choice(other_domains)
        alt_sentences = DOC_SENTENCES[alt_domain][:]
        rng.shuffle(alt_sentences)
        doc = " ".join(alt_sentences[:n_sents])

    return doc.strip()


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def _make_example(idx: int, split: str, rng: random.Random) -> dict:
    domain = rng.choice(DOMAINS)
    difficulty = rng.choice(DIFFICULTY_LEVELS)
    query = _build_query(domain, difficulty, rng)

    # Choose a main concept from the query (best-effort extraction)
    concepts = DOMAIN_CONCEPTS[domain]
    main_concept = rng.choice(concepts)

    # Corpus size: 50-100 items
    corpus_size = rng.randint(50, 100)

    # Relevant count: 1-5
    n_relevant = rng.randint(1, 5)
    n_relevant = min(n_relevant, corpus_size - 1)

    corpus_items = []
    relevant_ids = []
    irrelevant_ids = []

    # Generate relevant documents first
    for r in range(n_relevant):
        doc_id = f"doc_{idx}_{r:03d}"
        text = _build_document(domain, main_concept, is_relevant=True,
                                difficulty=difficulty, rng=rng)
        corpus_items.append({"id": doc_id, "text": text})
        relevant_ids.append(doc_id)

    # Fill the rest with irrelevant documents
    for r in range(corpus_size - n_relevant):
        doc_id = f"doc_{idx}_{r + n_relevant:03d}"
        text = _build_document(domain, main_concept, is_relevant=False,
                                difficulty=difficulty, rng=rng)
        corpus_items.append({"id": doc_id, "text": text})
        irrelevant_ids.append(doc_id)

    # Shuffle corpus so relevant items are not always first
    rng.shuffle(corpus_items)

    uid = hashlib.md5(f"{split}_{idx}_{domain}_{difficulty}".encode()).hexdigest()[:16]

    return {
        "id": uid,
        "query": query,
        "corpus_items": corpus_items,
        "relevant_ids": relevant_ids,
        "irrelevant_ids": irrelevant_ids,
        "metadata": {
            "split": split,
            "domain": domain,
            "difficulty": difficulty,
            "corpus_size": corpus_size,
            "n_relevant": n_relevant,
            "index": idx,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_split(split: str, n: int, rng: random.Random, out_path: Path) -> None:
    print(f"  Generating {n:,} {split} examples -> {out_path.name} ...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            example = _make_example(i, split, rng)
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print(f"    {split}: {i + 1:,} / {n:,} written", flush=True)
    print(f"  Done: {out_path}", flush=True)


def main() -> None:
    base_dir = Path(__file__).parent
    rng = random.Random(SEED)

    splits = [
        ("train", 25000),
        ("val", 2500),
        ("test", 2500),
    ]

    print("=== Retrieval Benchmark Dataset Generator ===")
    for split_name, count in splits:
        out_path = base_dir / f"retrieval_bench_{split_name}.jsonl"
        generate_split(split_name, count, rng, out_path)

    print("\nAll retrieval benchmark files generated successfully.")


if __name__ == "__main__":
    main()
