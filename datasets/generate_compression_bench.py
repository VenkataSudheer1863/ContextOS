"""
Compression Benchmark Dataset Generator for ContextOS Research.

Generates pairs for evaluating compression quality:
  - 25000 train + 2500 val + 2500 test examples
  - Each example: {id, original_text, compressed_text, compression_ratio,
                   rouge_l_score, semantic_similarity, information_preserved, metadata}
  - original_text: 200-1000 word passages from domains (tech, science, business, medical)
  - compressed_text: ~40% of original (simulate summarization)

Only stdlib imports used.
"""

import json
import random
import hashlib
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducible RNG
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Domain content building blocks
# ---------------------------------------------------------------------------

DOMAINS = ["technology", "science", "business", "medical"]

TECH_SENTENCES = [
    "Machine learning models are trained on large datasets to recognize patterns and make predictions.",
    "Transformer architectures have revolutionized natural language processing by enabling attention mechanisms.",
    "Cloud computing allows organizations to scale their infrastructure dynamically based on demand.",
    "Containerization technology like Docker packages applications with their dependencies for consistent deployment.",
    "Microservices architecture breaks monolithic applications into smaller, independently deployable services.",
    "API gateways serve as the entry point for client requests in distributed systems.",
    "Version control systems like Git track changes to source code over time.",
    "Continuous integration pipelines automate the building, testing, and validation of code changes.",
    "Neural networks learn hierarchical representations of data through multiple layers of computation.",
    "Database indexing improves query performance by creating data structures for faster lookups.",
    "Encryption algorithms protect sensitive data by transforming it into an unreadable format.",
    "Load balancers distribute incoming network traffic across multiple servers to ensure availability.",
    "Serverless computing abstracts away infrastructure management, letting developers focus on code.",
    "Graph databases store data in nodes and edges, optimizing for relationship-heavy queries.",
    "DevOps practices bridge the gap between development and operations teams for faster delivery.",
    "Quantum computing uses quantum mechanical phenomena to perform computations on qubits.",
    "Edge computing processes data closer to where it is generated, reducing latency.",
    "Blockchain technology creates immutable, distributed ledgers for recording transactions.",
    "Reinforcement learning trains agents to take actions by maximizing cumulative reward signals.",
    "Kubernetes orchestrates containerized workloads across clusters of machines automatically.",
    "Zero-trust security models verify every request regardless of network location.",
    "WebAssembly enables high-performance code execution in web browsers across languages.",
    "Federated learning trains machine learning models across decentralized devices without sharing raw data.",
    "In-memory databases store data in RAM to achieve extremely low read and write latencies.",
    "Model quantization reduces neural network precision to decrease memory footprint and inference time.",
]

SCIENCE_SENTENCES = [
    "CRISPR-Cas9 technology enables precise editing of DNA sequences within living organisms.",
    "The Standard Model of particle physics describes fundamental particles and their interactions.",
    "Climate change is driven by increasing concentrations of greenhouse gases in the atmosphere.",
    "Epigenetics studies heritable changes in gene expression that do not involve DNA sequence alterations.",
    "Quantum entanglement allows particles to be correlated regardless of the distance between them.",
    "Stem cells have the remarkable ability to differentiate into specialized cell types.",
    "The discovery of gravitational waves confirmed a major prediction of general relativity.",
    "Neuroplasticity allows the brain to reorganize itself by forming new neural connections throughout life.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "Dark matter constitutes approximately 27% of the universe but has not been directly observed.",
    "Antibiotic resistance is an escalating threat caused by the overuse and misuse of antibiotics.",
    "The human genome contains approximately 3 billion base pairs encoding roughly 20,000 genes.",
    "Exoplanets are planets that orbit stars outside our solar system and may harbor life.",
    "mRNA vaccines instruct cells to produce a protein that triggers an immune response.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "The microbiome plays a critical role in digestion, immune function, and mental health.",
    "Plate tectonics explains the movement of Earth's lithospheric plates over geological timescales.",
    "Supernovae are powerful stellar explosions that synthesize and distribute heavy elements.",
    "Nuclear fusion promises nearly limitless clean energy by mimicking reactions in the sun.",
    "Synthetic biology engineers biological systems and organisms for useful purposes.",
    "Bioluminescence is the production of light by living organisms through chemical reactions.",
    "The ozone layer absorbs most of the sun's ultraviolet radiation, protecting life on Earth.",
    "Protein folding determines the three-dimensional structure that governs a protein's function.",
    "Horizontal gene transfer allows bacteria to rapidly acquire resistance and new traits.",
    "The Higgs boson gives other fundamental particles their mass through the Higgs field.",
]

BUSINESS_SENTENCES = [
    "Supply chain optimization reduces costs and improves delivery times through data-driven decisions.",
    "Market segmentation divides consumers into groups based on shared characteristics or behaviors.",
    "Venture capital firms provide funding to early-stage companies in exchange for equity stakes.",
    "The balanced scorecard translates strategy into operational objectives across four perspectives.",
    "Mergers and acquisitions enable companies to grow quickly by combining with or buying other firms.",
    "Digital transformation reshapes business models by integrating digital technology throughout the organization.",
    "Customer lifetime value measures the total revenue a business can expect from a single customer.",
    "Agile project management iteratively delivers value through short development cycles called sprints.",
    "Brand equity reflects the premium a company can charge due to brand recognition and loyalty.",
    "Net Promoter Score measures customer loyalty by asking how likely they are to recommend a product.",
    "Porter's Five Forces analyzes industry competitiveness through suppliers, buyers, rivals, entrants, and substitutes.",
    "Corporate governance structures ensure accountability and transparency in organizational decision-making.",
    "Revenue forecasting uses historical data and market trends to predict future income streams.",
    "Economies of scale reduce per-unit costs as production volume increases over time.",
    "Environmental, social, and governance criteria guide sustainable and ethical investment decisions.",
    "Price elasticity measures how consumer demand changes in response to price fluctuations.",
    "Key performance indicators provide quantifiable measures of progress toward business objectives.",
    "Strategic alliances allow companies to collaborate without merging, sharing resources and expertise.",
    "Lean methodology eliminates waste in business processes to maximize customer value.",
    "E-commerce platforms enable businesses to sell products and services directly over the internet.",
    "Churn rate measures the proportion of customers who stop using a product within a given period.",
    "Intellectual property rights protect creations of the mind, providing competitive advantages.",
    "Working capital management ensures a company has sufficient liquidity to meet short-term obligations.",
    "Blue ocean strategy creates uncontested market space rather than competing in existing markets.",
    "Crowdsourcing taps into the collective intelligence of a large group to solve problems or generate ideas.",
]

MEDICAL_SENTENCES = [
    "Immunotherapy harnesses the patient's own immune system to fight cancer cells.",
    "Type 2 diabetes results from insulin resistance and is closely linked to obesity and lifestyle.",
    "Minimally invasive surgery reduces recovery time by using small incisions and specialized instruments.",
    "Pharmacogenomics studies how genetic variation influences an individual's response to drugs.",
    "Sepsis is a life-threatening condition caused by the body's extreme response to an infection.",
    "Telemedicine allows patients to consult healthcare providers remotely using digital communication tools.",
    "Alzheimer's disease is characterized by amyloid plaques and tau tangles that destroy neurons.",
    "Precision medicine tailors treatment strategies to the individual characteristics of each patient.",
    "Electronic health records improve care coordination by providing clinicians with comprehensive patient data.",
    "Stroke treatment outcomes improve dramatically when intervention occurs within the golden hour.",
    "Chronic obstructive pulmonary disease is a progressive lung condition caused primarily by smoking.",
    "Organ transplantation requires careful matching to minimize the risk of immune rejection.",
    "Anticoagulants prevent blood clots in high-risk patients by inhibiting clotting factor activity.",
    "MRI uses magnetic fields and radio waves to produce detailed images of internal body structures.",
    "Mental health conditions affect hundreds of millions globally and remain underdiagnosed and undertreated.",
    "Vaccines have eliminated or dramatically reduced the burden of numerous infectious diseases worldwide.",
    "Neonatal screening programs identify metabolic disorders in newborns before symptoms appear.",
    "Clinical trials are the gold standard for evaluating the safety and efficacy of new treatments.",
    "Palliative care focuses on improving quality of life for patients with serious illness.",
    "Biomarkers are measurable indicators of biological states used for diagnosis and treatment monitoring.",
    "Antibiotic stewardship programs promote the appropriate use of antibiotics to combat resistance.",
    "Gene therapy replaces or repairs defective genes to treat or prevent disease.",
    "The blood-brain barrier restricts the passage of substances from the bloodstream to the brain.",
    "Epidemiology studies the distribution and determinants of health conditions in populations.",
    "Robotic surgery systems enhance precision, flexibility, and control during complex procedures.",
]

DOMAIN_SENTENCES = {
    "technology": TECH_SENTENCES,
    "science": SCIENCE_SENTENCES,
    "business": BUSINESS_SENTENCES,
    "medical": MEDICAL_SENTENCES,
}

# Topic phrases per domain used to enrich passage context
DOMAIN_TOPICS = {
    "technology": [
        "software engineering", "artificial intelligence", "cybersecurity", "cloud infrastructure",
        "data engineering", "distributed systems", "human-computer interaction", "embedded systems",
    ],
    "science": [
        "genomics research", "astrophysics", "climate science", "neuroscience",
        "materials science", "quantum physics", "evolutionary biology", "oceanography",
    ],
    "business": [
        "financial management", "marketing strategy", "operations management", "entrepreneurship",
        "global supply chains", "organizational behavior", "corporate finance", "risk management",
    ],
    "medical": [
        "oncology treatment", "cardiovascular health", "infectious disease", "neurology",
        "pediatric care", "surgical innovation", "public health", "pharmacology",
    ],
}

# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def _target_word_count(rng: random.Random) -> int:
    """Return a target word count between 200 and 1000."""
    return rng.randint(200, 1000)


def _build_passage(domain: str, rng: random.Random) -> str:
    """Build a coherent-looking passage for the given domain."""
    sentences = DOMAIN_SENTENCES[domain]
    topic = rng.choice(DOMAIN_TOPICS[domain])
    target = _target_word_count(rng)

    # Opening sentence
    openers = [
        f"This report examines key developments in {topic}.",
        f"Recent advances in {topic} have attracted significant attention.",
        f"Understanding {topic} is essential for professionals in this field.",
        f"The following overview highlights critical aspects of {topic}.",
        f"Progress in {topic} continues to reshape the landscape of {domain}.",
    ]
    words: list[str] = []
    words.extend(rng.choice(openers).split())

    pool = sentences[:]
    rng.shuffle(pool)

    # Keep adding sentences until we reach or exceed the target
    idx = 0
    while len(words) < target:
        sent = pool[idx % len(pool)]
        idx += 1
        words.extend(sent.split())
        # Occasionally insert a transitional connector
        if rng.random() < 0.25 and len(words) < target - 20:
            connectors = [
                "Furthermore,", "In addition,", "Consequently,", "As a result,",
                "Building on this,", "Moreover,", "Notably,", "At the same time,",
            ]
            words.append(rng.choice(connectors))

    # Trim to within range (200-1000 words)
    words = words[:min(target, 1000)]
    if len(words) < 200:
        # Pad from pool
        extra = pool[:]
        rng.shuffle(extra)
        for s in extra:
            words.extend(s.split())
            if len(words) >= 200:
                break

    return " ".join(words)


def _compress_passage(passage: str, rng: random.Random) -> str:
    """Simulate ~40% summarization of a passage."""
    sentences_raw = re.split(r'(?<=[.!?])\s+', passage.strip())
    sentences_raw = [s for s in sentences_raw if s.strip()]

    # Target about 40% of word count
    passage_words = passage.split()
    target_compressed_words = max(30, int(len(passage_words) * 0.40))

    # Pick sentences greedily until we reach the target
    # Prefer earlier sentences (they carry more salient info in simulated text)
    rng.shuffle(sentences_raw[2:])  # keep first 2 sentences, shuffle rest
    ordered = sentences_raw[:2] + sentences_raw[2:]

    compressed_words: list[str] = []
    for sent in ordered:
        sent_words = sent.split()
        compressed_words.extend(sent_words)
        if len(compressed_words) >= target_compressed_words:
            break

    # Trim to exactly target
    compressed_words = compressed_words[:target_compressed_words]
    return " ".join(compressed_words)


# ---------------------------------------------------------------------------
# Metric helpers (stdlib-only approximations)
# ---------------------------------------------------------------------------

def _rouge_l(reference: str, hypothesis: str) -> float:
    """
    Fast approximate ROUGE-L F1 using unigram overlap (ROUGE-1 recall/precision
    harmonic mean).  True LCS is O(n*m) and too slow for passages of 200-1000
    words; this approximation is O(n+m) and gives comparable benchmark signal.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    hyp_counts: dict[str, int] = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1
    overlap = sum(min(ref_counts.get(t, 0), cnt) for t, cnt in hyp_counts.items())
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word sets as a proxy for semantic similarity."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _information_preserved(original: str, compressed: str) -> float:
    """
    Approximate information preservation as a weighted combination of
    ROUGE-L and Jaccard similarity, clamped to [0, 1].
    """
    rouge = _rouge_l(original, compressed)
    jaccard = _jaccard_similarity(original, compressed)
    score = 0.6 * rouge + 0.4 * jaccard
    # Add slight noise to simulate real metric variance
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def _make_example(idx: int, split: str, rng: random.Random) -> dict:
    domain = rng.choice(DOMAINS)
    original = _build_passage(domain, rng)
    compressed = _compress_passage(original, rng)

    original_words = len(original.split())
    compressed_words = len(compressed.split())
    compression_ratio = round(compressed_words / original_words, 4) if original_words else 0.0

    rouge_l = round(_rouge_l(original, compressed), 4)
    sem_sim = round(_jaccard_similarity(original, compressed), 4)
    info_pres = _information_preserved(original, compressed)

    uid = hashlib.md5(f"{split}_{idx}_{domain}".encode()).hexdigest()[:16]

    return {
        "id": uid,
        "original_text": original,
        "compressed_text": compressed,
        "compression_ratio": compression_ratio,
        "rouge_l_score": rouge_l,
        "semantic_similarity": sem_sim,
        "information_preserved": info_pres,
        "metadata": {
            "split": split,
            "domain": domain,
            "original_word_count": original_words,
            "compressed_word_count": compressed_words,
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

    print("=== Compression Benchmark Dataset Generator ===")
    for split_name, count in splits:
        out_path = base_dir / f"compression_bench_{split_name}.jsonl"
        generate_split(split_name, count, rng, out_path)

    print("\nAll compression benchmark files generated successfully.")


if __name__ == "__main__":
    main()
