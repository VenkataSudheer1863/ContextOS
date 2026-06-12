"""
Generate contrastive learning datasets for embedding model fine-tuning.
Targets: BGE-M3, E5-Large-V2
Outputs:
  datasets/embedding_finetune_train.jsonl  (120,000 pairs)
  datasets/embedding_finetune_val.jsonl    ( 10,000 pairs)
  datasets/embedding_finetune_test.jsonl   ( 10,000 pairs)
"""

import json
import random
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Domain-specific content pools
# ---------------------------------------------------------------------------

DOMAINS = {
    "software_engineering": {
        "entities": [
            "authentication system", "database schema", "CI/CD pipeline",
            "API gateway", "microservice", "load balancer", "cache layer",
            "message queue", "service mesh", "container orchestration",
            "logging infrastructure", "monitoring stack", "rate limiter",
            "GraphQL schema", "REST endpoint", "WebSocket handler",
            "OAuth2 provider", "JWT validation", "session management",
            "feature flag service", "A/B testing framework", "rollout strategy",
            "database migration", "ORM configuration", "connection pooling",
            "search index", "full-text search", "vector store integration",
            "event bus", "dead-letter queue", "retry mechanism",
        ],
        "decisions": [
            "OAuth2 with PKCE flow was selected for all API authentication endpoints after evaluating Auth0, Cognito, and Keycloak",
            "PostgreSQL 15 with partitioning was chosen over MongoDB for the transactional data store due to ACID requirements",
            "Kubernetes with Helm charts was adopted for container orchestration replacing the previous Docker Swarm setup",
            "Redis Cluster with 3 shards was provisioned for session storage and rate limiting with 99.99% SLA",
            "GraphQL Federation v2 was selected to unify the product, order, and user microservice APIs",
            "Apache Kafka with 8 partitions was configured for the event streaming backbone replacing RabbitMQ",
            "gRPC with Protobuf v3 was standardised for inter-service communication to reduce latency by 40%",
            "Elasticsearch 8.x was deployed for full-text and vector search replacing the legacy Solr cluster",
            "Istio service mesh was adopted for mTLS, circuit breaking, and observability across 12 microservices",
            "Feature flags via LaunchDarkly were integrated to enable zero-downtime progressive rollouts",
        ],
        "procedures": [
            "To deploy a new microservice: create Helm chart, push to registry, update ArgoCD ApplicationSet, verify health checks",
            "Database migration procedure: write Flyway script, run on staging, obtain DBA approval, apply during maintenance window",
            "Incident response: page on-call via PagerDuty, join war-room Slack channel, assign incident commander, document in Jira",
            "Code review process: open PR with linked ticket, two approvals required, pass CI checks, squash-merge to main",
            "On-boarding: provision IAM role, grant GitHub team access, set up local dev environment using Docker Compose",
        ],
        "states": [
            "The authentication service is currently in read-only mode pending the OAuth2 migration scheduled for Friday",
            "Database replication lag is 3 seconds after the schema migration; write traffic is throttled to 60% capacity",
            "The Kubernetes cluster autoscaler scaled the API pods from 6 to 14 replicas during the Black Friday traffic spike",
            "Feature flag `new_checkout_flow` is enabled for 25% of users as part of the canary rollout",
            "The message queue depth reached 2.1 million messages; consumer group lag alert was acknowledged by the platform team",
        ],
    },
    "research": {
        "entities": [
            "experiment design", "baseline model", "ablation study",
            "evaluation metric", "benchmark dataset", "training regime",
            "hyperparameter search", "loss function", "attention mechanism",
            "pre-training corpus", "fine-tuning strategy", "prompt template",
            "retrieval augmentation", "context window", "token budget",
            "embedding space", "contrastive objective", "cross-encoder",
            "bi-encoder", "reranker", "distillation pipeline",
        ],
        "decisions": [
            "BM25 + dense retrieval hybrid was selected as the retrieval baseline after comparing seven retrieval strategies",
            "The context window was set to 8,192 tokens balancing memory footprint and long-document recall",
            "Contrastive loss with in-batch negatives was chosen over triplet loss due to faster convergence on MS-MARCO",
            "The ablation removed positional embeddings in the cross-attention layer, yielding a 2.1 BLEU improvement",
            "GPT-4o was used as the judge model for automated evaluation, achieving 0.89 Spearman correlation with human raters",
        ],
        "procedures": [
            "Experiment procedure: register config in MLflow, launch training on A100 cluster, log metrics every 100 steps, run eval on held-out set",
            "Ablation protocol: fix all hyperparameters, disable one component at a time, report delta over full model on three seeds",
            "Human evaluation: recruit 5 annotators, provide calibration examples, compute Fleiss kappa, adjudicate disagreements",
        ],
        "states": [
            "The retrieval model achieved 72.4 nDCG@10 on BEIR after 3 epochs; training is paused for hyperparameter review",
            "Experiment run #47 is currently queued behind two higher-priority jobs on the A100 cluster",
            "The embedding model fine-tuning completed; checkpoint at step 50,000 shows 3.2% improvement over the BGE-M3 baseline",
        ],
    },
    "healthcare": {
        "entities": [
            "patient record", "clinical trial", "drug interaction",
            "diagnostic protocol", "treatment plan", "lab result",
            "medication dosage", "surgical procedure", "discharge summary",
            "referral pathway", "consent form", "adverse event report",
            "pharmacy order", "imaging report", "vital signs",
        ],
        "decisions": [
            "The cardiology team decided to initiate anticoagulation therapy with apixaban 5mg twice daily for the AF patient",
            "The ethics committee approved the Phase II trial protocol for the novel immunotherapy agent with an N=120 cohort",
            "The formulary committee added linagliptin to the preferred DPP-4 inhibitor tier effective Q3",
            "The discharge planning team arranged home health nursing visits three times weekly post hip replacement",
            "The tumour board recommended FOLFOX chemotherapy followed by surgical resection for the Stage IIIB colorectal case",
        ],
        "procedures": [
            "Pre-op checklist: confirm NPO status, verify consent, mark surgical site, administer prophylactic antibiotics, timeout",
            "Sepsis protocol: blood cultures x2, lactate level, 30 ml/kg IV crystalloid bolus, broad-spectrum antibiotics within 1 hour",
            "Medication reconciliation: compare pre-admission list with current orders, resolve discrepancies, document in EHR",
        ],
        "states": [
            "Patient in Bay 4 is awaiting cardiology consult; potassium is 3.1 mEq/L and supplementation has been ordered",
            "The MRI scanner in Suite B is offline for calibration; urgent studies are being diverted to Suite A",
            "Clinical trial enrollment is at 87 of 120 participants; interim analysis is scheduled for next month",
        ],
    },
    "legal": {
        "entities": [
            "contract clause", "litigation hold", "discovery request",
            "deposition transcript", "settlement agreement", "motion to dismiss",
            "privilege log", "regulatory filing", "IP assignment",
            "non-disclosure agreement", "arbitration clause", "due diligence",
            "merger agreement", "employment contract", "data processing agreement",
        ],
        "decisions": [
            "Outside counsel recommended settling the patent dispute for $4.2M to avoid prolonged litigation risk",
            "The data processing agreement was amended to include Standard Contractual Clauses for EU data transfers post-Schrems II",
            "The board approved the Delaware reincorporation to benefit from the established corporate law framework",
            "Legal determined the non-compete clause is unenforceable in California and must be excised from all offer letters",
            "The arbitration clause was modified to allow class arbitration following the AAA policy update",
        ],
        "procedures": [
            "Contract review process: initial red-line by associate, partner sign-off on material terms, legal ops upload to contract repository",
            "Litigation hold procedure: identify custodians, issue hold notice, suspend auto-deletion, collect from email and Slack",
            "Due diligence checklist: IP ownership, pending litigation, regulatory compliance, key contracts, employment agreements",
        ],
        "states": [
            "Discovery responses are due in 21 days; the review team has processed 40% of the 120,000 document set",
            "The patent application is in examination; office action response is due by the 15th of next month",
            "The merger agreement is in final markup; three open issues remain on indemnification caps and rep survival periods",
        ],
    },
    "finance": {
        "entities": [
            "budget allocation", "revenue forecast", "expense report",
            "cash flow model", "risk exposure", "portfolio rebalancing",
            "audit finding", "tax provision", "capital expenditure",
            "credit facility", "covenant compliance", "hedge position",
            "earnings guidance", "valuation model", "working capital",
        ],
        "decisions": [
            "The investment committee approved a $50M allocation to the infrastructure debt fund at a 7.2% target IRR",
            "The CFO approved a $12M capex for the ERP system upgrade to SAP S/4HANA, phased over 18 months",
            "The board authorised a $200M share buyback programme to be executed over 24 months via 10b5-1 plan",
            "Treasury decided to hedge 70% of EUR exposure for the next 12 months using vanilla FX forwards",
            "The audit committee accepted management's assessment that the internal control deficiency is not material",
        ],
        "procedures": [
            "Monthly close process: sub-ledger reconciliations by day 3, intercompany eliminations by day 5, MD&A draft by day 8",
            "Budget revision procedure: department heads submit change requests, FP&A consolidates, CFO approval, board notification if >5%",
            "Expense reimbursement: submit receipts within 30 days, manager approval, finance audit for items over $500",
        ],
        "states": [
            "Q3 revenue is tracking $8M below plan; the revised forecast has been submitted to the board for review",
            "The revolving credit facility utilisation is at 65%; the leverage covenant requires it remain below 3.5x EBITDA",
            "The tax provision for deferred liabilities has been adjusted by $2.3M following the transfer pricing study",
        ],
    },
    "customer_service": {
        "entities": [
            "support ticket", "escalation policy", "SLA breach",
            "customer complaint", "refund request", "product recall",
            "account suspension", "knowledge base article", "chatbot flow",
            "CSAT score", "NPS survey", "agent performance",
            "queue management", "omnichannel routing", "voice of customer",
        ],
        "decisions": [
            "The support leadership decided to extend the refund window from 30 to 60 days for all premium subscribers",
            "Tier-2 escalation was re-routed to the specialised billing team after a 22% CSAT drop in the previous quarter",
            "The chatbot was configured to hand off to a human agent after two failed resolution attempts instead of three",
            "The product recall notice was sent to 14,200 affected customers via email and in-app notification within 4 hours",
            "The SLA was revised to 4-hour first response for P1 tickets and 24-hour for P2 tickets effective next quarter",
        ],
        "procedures": [
            "Escalation procedure: agent documents steps taken, tags ticket as escalated, assigns to tier-2 queue, updates customer ETA",
            "Refund processing: verify purchase in order management, confirm eligibility, initiate refund in payment gateway, notify customer",
            "Knowledge base update: agent identifies gap, drafts article, peer review, QA approval, publish, tag related tickets",
        ],
        "states": [
            "The P1 queue has 7 open tickets; average wait time is 18 minutes against the 4-hour SLA target",
            "CSAT for the billing team improved to 4.3/5.0 this week following the new escalation training rollout",
            "The chatbot deflection rate is 61%; the target is 70% by end of quarter with two new FAQ flows planned",
        ],
    },
}

QUERY_TEMPLATES = {
    "decision_retrieval": [
        "What decision was made about the {entity}?",
        "What was decided regarding the {entity}?",
        "Which option was chosen for the {entity}?",
        "What approach was approved for the {entity}?",
        "What was the final call on the {entity}?",
        "What did the team decide about the {entity}?",
        "How was the {entity} decision resolved?",
        "What conclusion was reached on the {entity}?",
    ],
    "factual_recall": [
        "What is the current status of the {entity}?",
        "What details were recorded about the {entity}?",
        "What information is available about the {entity}?",
        "Summarise the key facts about the {entity}.",
        "What was documented regarding the {entity}?",
        "What are the specifics of the {entity}?",
    ],
    "state_tracking": [
        "What is the current state of the {entity}?",
        "Has the {entity} changed recently?",
        "What is the latest update on the {entity}?",
        "Where does the {entity} stand right now?",
        "What is the most recent status of the {entity}?",
        "Is the {entity} still in the same state?",
    ],
    "procedure_lookup": [
        "What are the steps for the {entity} process?",
        "How should the {entity} be handled?",
        "What is the procedure for {entity}?",
        "Walk me through the {entity} workflow.",
        "What is the correct process for {entity}?",
        "How do we execute the {entity}?",
    ],
    "timeline_query": [
        "When was the {entity} last updated?",
        "What happened with the {entity} recently?",
        "Give me the timeline of events for the {entity}.",
        "What changes occurred to the {entity} over time?",
        "When did the {entity} change?",
    ],
}

NEGATIVE_TEMPLATES = [
    # Cross-domain distractors
    "The {neg_entity} was updated to reflect the new requirements from the stakeholder review meeting.",
    "Sprint planning for Q{q} was completed with {pts} story points allocated to the {neg_entity} workstream.",
    "The {neg_entity} configuration was reviewed and approved with no changes required at this stage.",
    "An automated report on {neg_entity} performance was distributed to the leadership team on Friday.",
    "The team noted that {neg_entity} metrics were within acceptable bounds for the current reporting period.",
    "A cross-functional working group was formed to evaluate options for {neg_entity} optimisation.",
    "Training materials for {neg_entity} have been updated and are available on the internal wiki.",
    "The {neg_entity} roadmap for the next fiscal year was presented to the steering committee.",
    "Compliance requirements for {neg_entity} were reviewed by the legal and security teams.",
    "The {neg_entity} vendor contract is up for renewal and three alternative providers have been shortlisted.",
    "Budget for {neg_entity} initiatives has been provisionally approved pending board sign-off.",
    "The {neg_entity} team completed their annual review and submitted recommendations for process improvement.",
    "A pilot programme for {neg_entity} was launched in two regions with results expected in six weeks.",
    "Documentation for the {neg_entity} has been migrated to Confluence and is now discoverable via search.",
    "The {neg_entity} working group meets bi-weekly on Tuesdays at 10am to review progress and blockers.",
]


def get_all_entities():
    entities = []
    for domain_data in DOMAINS.values():
        entities.extend(domain_data["entities"])
    return entities


ALL_ENTITIES = get_all_entities()


def pick_positive(domain_name, query_type):
    domain = DOMAINS[domain_name]
    if query_type == "decision_retrieval":
        return random.choice(domain["decisions"])
    elif query_type == "state_tracking":
        return random.choice(domain["states"])
    elif query_type == "procedure_lookup":
        return random.choice(domain["procedures"])
    else:
        # factual_recall and timeline_query: mix from decisions + states
        pool = domain["decisions"] + domain["states"]
        return random.choice(pool)


def build_negative(entity):
    template = random.choice(NEGATIVE_TEMPLATES)
    neg_entity = random.choice([e for e in ALL_ENTITIES if e != entity])
    return template.format(
        neg_entity=neg_entity,
        q=random.randint(1, 4),
        pts=random.choice([8, 13, 21, 34, 5]),
    )


def generate_sample():
    domain_name = random.choice(list(DOMAINS.keys()))
    query_type = random.choice(list(QUERY_TEMPLATES.keys()))
    entity = random.choice(DOMAINS[domain_name]["entities"])

    # Build query
    q_template = random.choice(QUERY_TEMPLATES[query_type])
    query = q_template.format(entity=entity)

    # Positive
    positive = pick_positive(domain_name, query_type)

    # Negatives: 3 hard negatives from different domains / off-topic content
    negatives = []
    other_domains = [d for d in DOMAINS if d != domain_name]
    # 1 negative from a different domain (same category type)
    other_domain = random.choice(other_domains)
    other_type_pool = (
        DOMAINS[other_domain]["decisions"]
        + DOMAINS[other_domain]["states"]
        + DOMAINS[other_domain]["procedures"]
    )
    negatives.append(random.choice(other_type_pool))
    # 2 template-based noisy negatives
    for _ in range(2):
        negatives.append(build_negative(entity))

    random.shuffle(negatives)

    return {
        "query": query,
        "positive": positive,
        "negatives": negatives,
        "metadata": {
            "domain": domain_name,
            "query_type": query_type,
            "entity": entity,
        },
    }


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples):,} records -> {path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    splits = {
        "train": 120_000,
        "val":    10_000,
        "test":   10_000,
    }

    total = sum(splits.values())
    print(f"Generating {total:,} total embedding fine-tuning samples...")

    for split_name, count in splits.items():
        print(f"  Generating {count:,} {split_name} samples...")
        samples = [generate_sample() for _ in range(count)]
        out_path = os.path.join(base_dir, f"embedding_finetune_{split_name}.jsonl")
        write_jsonl(out_path, samples)

    print("Done. Embedding fine-tuning dataset generation complete.")


if __name__ == "__main__":
    main()
