"""
ContextBench-Long Dataset Generator
Generates realistic context management examples for ContextOS research.
70K total examples: 60K train, 5K val, 5K test
Uses only stdlib: random, json, os, time
"""

import random
import json
import os
import time

SEED = 42
random.seed(SEED)

TASK_TYPES = [
    "summarize_session",
    "track_state",
    "detect_contradiction",
    "recall_procedure",
    "synthesize_facts",
    "answer_from_memory",
    "identify_gaps",
    "timeline_reconstruction",
]

DOMAINS = [
    "software_engineering",
    "research",
    "customer_service",
    "healthcare",
    "legal",
    "finance",
]

DIFFICULTIES = ["easy", "medium", "hard"]
ITEM_TYPES = ["observation", "decision", "statement", "action", "question", "fact", "event", "note"]

# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific content banks
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_CONTENT = {
    "software_engineering": {
        "entities": [
            "authentication module", "payment gateway", "user dashboard", "REST API",
            "database schema", "CI/CD pipeline", "microservice", "frontend component",
            "load balancer", "caching layer", "message queue", "WebSocket server",
            "OAuth2 provider", "GraphQL endpoint", "container orchestration",
        ],
        "actions": [
            "refactored", "deployed", "reviewed", "merged", "reverted", "tested",
            "optimized", "migrated", "scaled", "deprecated", "integrated", "secured",
        ],
        "decisions": [
            "use PostgreSQL instead of MySQL",
            "implement JWT tokens with 24-hour expiry",
            "adopt microservices architecture",
            "use Redis for session caching",
            "migrate to Kubernetes for orchestration",
            "implement rate limiting at 100 req/min",
            "switch from REST to GraphQL",
            "enforce 2FA for admin accounts",
            "use blue-green deployment strategy",
            "adopt trunk-based development workflow",
        ],
        "issues": [
            "memory leak in the worker process",
            "race condition in the payment handler",
            "N+1 query problem in the user listing endpoint",
            "missing index on the transactions table",
            "CORS misconfiguration blocking mobile clients",
            "token refresh logic not handling edge cases",
        ],
        "people": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"],
        "task_templates": {
            "summarize_session": [
                "Based on the conversation history, summarize what decisions were made about the {entity}",
                "What were the key outcomes discussed regarding the {entity} in this session?",
                "Summarize the team's discussion about improving the {entity}",
            ],
            "track_state": [
                "What is the current status of the {entity} based on all context items?",
                "Track the progress of {entity} development from the context provided",
                "What stage is the {entity} currently in based on the conversation?",
            ],
            "detect_contradiction": [
                "Are there any contradictory statements about the {entity} in the context?",
                "Identify any conflicting decisions made about the {entity}",
                "Find inconsistencies in the team's approach to the {entity}",
            ],
            "recall_procedure": [
                "What steps were agreed upon for deploying the {entity}?",
                "Describe the procedure for onboarding new users according to the context",
                "What is the rollback procedure discussed in the context?",
            ],
            "synthesize_facts": [
                "Synthesize all information about the {entity} from the context into a coherent summary",
                "What do we know about the performance characteristics of the {entity}?",
                "Compile all requirements mentioned for the {entity}",
            ],
            "answer_from_memory": [
                "Who was assigned responsibility for the {entity}?",
                "What deadline was set for the {entity} according to the context?",
                "Which technology was chosen for implementing the {entity}?",
            ],
            "identify_gaps": [
                "What information is missing about the {entity} that would be needed to proceed?",
                "What questions remain unanswered about the {entity} in this context?",
                "Identify what decisions still need to be made about the {entity}",
            ],
            "timeline_reconstruction": [
                "Reconstruct the sequence of events related to the {entity}",
                "In what order were decisions made about the {entity}?",
                "Create a timeline of changes to the {entity} from the context",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The team decided to {decision}. {person} was assigned to lead the implementation. The deadline was set for the end of sprint.",
                "Key decisions included: {decision}. The {entity} will be prioritized in the next quarter.",
                "The discussion concluded with agreement to {decision}. Follow-up items were assigned to {person}.",
            ],
            "track_state": [
                "The {entity} is currently in the testing phase. {person} completed the initial implementation and it is awaiting code review.",
                "The {entity} has been deployed to staging. Integration tests are passing and production deployment is scheduled.",
                "Development of the {entity} is 70% complete. The remaining work involves edge case handling and documentation.",
            ],
            "detect_contradiction": [
                "Yes, there is a contradiction: initially the team decided to use JWT tokens, but later decided to use session cookies instead.",
                "A conflict exists: {person} recommended using PostgreSQL while {person2} argued for MongoDB for the same use case.",
                "No direct contradiction found, but there is ambiguity about whether rate limiting should be applied at the gateway or service level.",
            ],
            "recall_procedure": [
                "The agreed deployment procedure is: 1) run tests, 2) create a release branch, 3) deploy to staging, 4) run smoke tests, 5) deploy to production with blue-green switch.",
                "The rollback procedure involves: reverting the deployment, restoring the database snapshot, and notifying the on-call team.",
                "New user onboarding requires: account verification email, profile setup guide, and a 30-day trial activation.",
            ],
            "synthesize_facts": [
                "The {entity} handles approximately 10K requests per day, uses {tech} for storage, and was originally built by {person}. It has known performance issues under high load.",
                "Based on context: the {entity} was built in Q3, requires Redis caching, and supports up to 1000 concurrent connections.",
                "The {entity} integrates with three external services, has 99.5% uptime SLA, and is maintained by the platform team.",
            ],
            "answer_from_memory": [
                "{person} was assigned responsibility for the {entity} during the planning session.",
                "The deadline for the {entity} was set to the end of Q4 based on the roadmap discussion.",
                "The team chose PostgreSQL as the database for the {entity} after evaluating three options.",
            ],
            "identify_gaps": [
                "The context does not specify: the budget allocation, the rollback strategy, or who will handle on-call duties for the {entity}.",
                "Missing information includes: performance benchmarks, security requirements, and approval from the architecture review board.",
                "No decision was made about scaling strategy or disaster recovery for the {entity}.",
            ],
            "timeline_reconstruction": [
                "Timeline: T1 - Initial design proposed; T2 - {person} raised security concerns; T3 - architecture revised; T4 - implementation started; T5 - first PR merged.",
                "Events in order: requirements gathered (day 1), design reviewed (day 3), implementation begun (day 5), first issue found (day 8), resolved and deployed (day 12).",
                "The {entity} went through: planning → design → implementation → testing → hotfix → stable release over 3 sprints.",
            ],
        },
    },

    "research": {
        "entities": [
            "neural network model", "experiment results", "dataset", "hypothesis",
            "literature review", "methodology", "benchmark", "ablation study",
            "training pipeline", "evaluation metric", "baseline comparison", "paper draft",
        ],
        "actions": [
            "analyzed", "validated", "replicated", "published", "submitted", "rejected",
            "revised", "benchmarked", "collected", "annotated", "trained", "evaluated",
        ],
        "decisions": [
            "use cross-validation instead of a held-out test set",
            "increase the model size to 1B parameters",
            "use human evaluation alongside automated metrics",
            "focus on low-resource languages for the study",
            "adopt a transformer-based architecture",
            "use BLEU and ROUGE for evaluation",
            "collect additional data from domain experts",
        ],
        "issues": [
            "overfitting on the training set",
            "evaluation metric does not correlate with human judgment",
            "dataset contains label noise",
            "reproducibility issues with the baseline",
            "high variance across random seeds",
        ],
        "people": ["Dr. Smith", "Dr. Lee", "Prof. Patel", "Dr. Chen", "Dr. Martinez"],
        "task_templates": {
            "summarize_session": [
                "Summarize the key findings discussed about the {entity}",
                "What conclusions were reached about the {entity} in this research session?",
            ],
            "track_state": [
                "What is the current status of the {entity} research?",
                "Where does the {entity} stand in the research pipeline?",
            ],
            "detect_contradiction": [
                "Are there conflicting findings about the {entity} in the literature cited?",
                "Identify any contradictory experimental results regarding {entity}",
            ],
            "recall_procedure": [
                "What experimental procedure was described for evaluating the {entity}?",
                "Describe the data collection methodology mentioned for the {entity}",
            ],
            "synthesize_facts": [
                "Synthesize all findings about {entity} from the context",
                "What does the collected evidence say about {entity}?",
            ],
            "answer_from_memory": [
                "What accuracy did the {entity} achieve on the benchmark?",
                "Which researcher proposed the {entity} approach?",
            ],
            "identify_gaps": [
                "What aspects of the {entity} remain unexplored based on the context?",
                "What additional experiments are needed for the {entity}?",
            ],
            "timeline_reconstruction": [
                "Reconstruct the research timeline for the {entity}",
                "In what order were experiments on the {entity} conducted?",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The team found that the {entity} achieves state-of-the-art results on 3 out of 5 benchmarks. Key limitations include computational cost and data requirements.",
                "Discussion concluded that the {entity} shows promise but requires further validation on diverse datasets.",
            ],
            "track_state": [
                "The {entity} is currently in the evaluation phase. Initial results are promising with 92% accuracy on the validation set.",
                "The {entity} research is at the writing stage. Experiments are complete and results are being compiled for submission.",
            ],
            "detect_contradiction": [
                "A contradiction exists: early experiments showed 85% accuracy while the latest run shows 78%, suggesting implementation differences.",
                "No contradiction found, but results vary significantly across domains, which warrants further investigation.",
            ],
            "recall_procedure": [
                "The evaluation procedure: 1) preprocess data with standard tokenization, 2) fine-tune for 10 epochs, 3) evaluate on held-out test set, 4) report mean and std over 3 seeds.",
            ],
            "synthesize_facts": [
                "The {entity} was trained on 500K examples, achieves 91% F1 on the primary benchmark, and outperforms the baseline by 4.2 percentage points.",
            ],
            "answer_from_memory": [
                "The {entity} achieved 89.3% accuracy on the standard benchmark, as reported in the third experiment.",
            ],
            "identify_gaps": [
                "Missing: evaluation on out-of-domain data, ablation study for individual components, and analysis of failure cases.",
            ],
            "timeline_reconstruction": [
                "Timeline: literature review (week 1-2), dataset collection (week 3-4), model training (week 5-7), evaluation (week 8), paper writing (week 9-10).",
            ],
        },
    },

    "customer_service": {
        "entities": [
            "customer complaint", "refund request", "account issue", "billing dispute",
            "product return", "service outage", "subscription upgrade", "password reset",
            "order tracking", "warranty claim", "technical support ticket", "escalation",
        ],
        "actions": [
            "escalated", "resolved", "refunded", "replaced", "investigated", "contacted",
            "credited", "canceled", "renewed", "updated", "verified", "closed",
        ],
        "decisions": [
            "issue a full refund within 3-5 business days",
            "escalate to Tier 2 support",
            "provide a 20% discount on the next purchase",
            "replace the defective product at no cost",
            "schedule a callback within 24 hours",
            "waive the cancellation fee as a goodwill gesture",
        ],
        "issues": [
            "customer charged twice for the same order",
            "product arrived damaged",
            "account locked due to too many failed login attempts",
            "subscription not canceled despite request",
            "incorrect item shipped",
        ],
        "people": ["Agent Maria", "Agent John", "Supervisor Lisa", "Agent Raj", "Agent Sam"],
        "task_templates": {
            "summarize_session": [
                "Summarize the customer interaction regarding the {entity}",
                "What was the outcome of the support session about the {entity}?",
            ],
            "track_state": [
                "What is the current status of the {entity}?",
                "Where does the {entity} stand in the resolution process?",
            ],
            "detect_contradiction": [
                "Are there inconsistencies in the information provided about the {entity}?",
                "Identify any conflicting statements made during the {entity} interaction",
            ],
            "recall_procedure": [
                "What steps were taken to resolve the {entity}?",
                "Describe the escalation procedure followed for the {entity}",
            ],
            "synthesize_facts": [
                "Compile all relevant information about the {entity} from the interaction",
                "What is the complete picture of the {entity} based on the context?",
            ],
            "answer_from_memory": [
                "What resolution was offered to the customer for the {entity}?",
                "How long has the {entity} been open?",
            ],
            "identify_gaps": [
                "What information is still needed to resolve the {entity}?",
                "What follow-up actions are outstanding for the {entity}?",
            ],
            "timeline_reconstruction": [
                "Reconstruct the timeline of events for the {entity}",
                "In what order did events occur related to the {entity}?",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The customer contacted support about a {entity}. {person} investigated and approved a full refund. The case was closed after confirmation email was sent.",
                "Interaction summary: customer reported {entity}, agent verified the issue, offered resolution of credit to account, customer accepted.",
            ],
            "track_state": [
                "The {entity} is currently open and awaiting manager approval. Initial contact was made 2 days ago and a follow-up is scheduled for tomorrow.",
                "The {entity} has been resolved. A replacement was shipped and tracking information provided to the customer.",
            ],
            "detect_contradiction": [
                "Inconsistency found: the customer stated the order was placed on the 5th, but the system shows the 7th as the order date.",
                "No contradictions found; all statements are consistent with the account history.",
            ],
            "recall_procedure": [
                "Resolution steps: 1) verify customer identity, 2) confirm issue details, 3) check policy eligibility, 4) process refund/replacement, 5) send confirmation email.",
            ],
            "synthesize_facts": [
                "The {entity} involves order #45821, customer account active for 3 years, original purchase of $89.99, first contact on Monday, now pending Tier 2 review.",
            ],
            "answer_from_memory": [
                "The customer was offered a full refund of $89.99 plus a $15 store credit as a goodwill gesture.",
            ],
            "identify_gaps": [
                "Still needed: proof of purchase photo, confirmation of shipping address, and manager authorization for refund over $100.",
            ],
            "timeline_reconstruction": [
                "Day 1: customer placed order. Day 3: order shipped. Day 7: customer reported non-delivery. Day 8: investigation opened. Day 10: replacement shipped.",
            ],
        },
    },

    "healthcare": {
        "entities": [
            "patient record", "treatment plan", "medication regimen", "diagnostic result",
            "clinical trial", "discharge summary", "follow-up appointment", "lab results",
            "surgical procedure", "allergy profile", "insurance authorization", "referral",
        ],
        "actions": [
            "prescribed", "administered", "diagnosed", "referred", "discharged", "admitted",
            "scheduled", "reviewed", "updated", "discontinued", "monitored", "assessed",
        ],
        "decisions": [
            "start the patient on metformin 500mg twice daily",
            "schedule a follow-up MRI in 6 weeks",
            "refer to cardiology for further evaluation",
            "discontinue the current antibiotic course",
            "admit for observation overnight",
            "adjust insulin dosage based on glucose readings",
        ],
        "issues": [
            "patient allergic to penicillin - alternative required",
            "conflicting lab results requiring repeat testing",
            "medication interaction between warfarin and aspirin",
            "insurance prior authorization pending",
            "patient non-compliant with prescribed regimen",
        ],
        "people": ["Dr. Williams", "Dr. Nguyen", "Nurse Thompson", "Dr. Garcia", "Dr. Patel"],
        "task_templates": {
            "summarize_session": [
                "Summarize the clinical discussion about the {entity}",
                "What decisions were made regarding the patient's {entity}?",
            ],
            "track_state": [
                "What is the current status of the {entity}?",
                "Where does the {entity} stand in the care plan?",
            ],
            "detect_contradiction": [
                "Are there any contradictory findings in the {entity}?",
                "Identify inconsistencies in the {entity} documentation",
            ],
            "recall_procedure": [
                "What procedure was outlined for managing the {entity}?",
                "Describe the clinical protocol mentioned for the {entity}",
            ],
            "synthesize_facts": [
                "Compile all clinical information about the {entity} from the context",
                "Synthesize the patient's {entity} history from the available records",
            ],
            "answer_from_memory": [
                "What medication was prescribed for the {entity}?",
                "What were the key findings in the {entity}?",
            ],
            "identify_gaps": [
                "What information is missing from the {entity} that is needed for treatment?",
                "What diagnostic steps are still pending for the {entity}?",
            ],
            "timeline_reconstruction": [
                "Reconstruct the timeline of events related to the {entity}",
                "In what order were clinical decisions made regarding the {entity}?",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The clinical team reviewed the {entity} and agreed to adjust the treatment plan. Dr. Williams ordered additional labs and scheduled a follow-up in 2 weeks.",
                "Discussion of {entity} resulted in: medication adjustment, referral to specialist, and patient education session scheduled.",
            ],
            "track_state": [
                "The {entity} is currently under review by the specialist. Labs are pending and results expected within 48 hours.",
                "The {entity} has been updated following the latest consultation. Patient is stable and responding to treatment.",
            ],
            "detect_contradiction": [
                "Contradiction identified: the patient reported no prior cardiac history, but the ECG shows evidence of a previous myocardial event.",
                "Inconsistency in {entity}: blood pressure recorded as 140/90 in morning notes and 118/75 in afternoon notes on the same day.",
            ],
            "recall_procedure": [
                "Protocol for {entity}: 1) initial assessment, 2) order relevant diagnostics, 3) consult specialist if indicated, 4) develop treatment plan, 5) patient education, 6) schedule follow-up.",
            ],
            "synthesize_facts": [
                "Patient has Type 2 diabetes diagnosed 5 years ago, currently on metformin, HbA1c of 7.8%, and is due for annual eye exam. No known drug allergies.",
            ],
            "answer_from_memory": [
                "Metformin 500mg twice daily was prescribed for the {entity}, with instructions to monitor blood glucose weekly.",
            ],
            "identify_gaps": [
                "Missing from {entity}: recent kidney function tests, full medication reconciliation, and family history documentation.",
            ],
            "timeline_reconstruction": [
                "Timeline: initial visit (week 1), diagnosis confirmed (week 2), treatment started (week 2), follow-up (week 6), labs reviewed (week 7), dosage adjusted (week 8).",
            ],
        },
    },

    "legal": {
        "entities": [
            "contract clause", "litigation case", "compliance requirement", "intellectual property",
            "regulatory filing", "settlement agreement", "due diligence", "merger document",
            "employment agreement", "liability assessment", "court ruling", "evidence",
        ],
        "actions": [
            "filed", "reviewed", "negotiated", "appealed", "settled", "drafted",
            "executed", "challenged", "disclosed", "redacted", "subpoenaed", "enjoined",
        ],
        "decisions": [
            "pursue arbitration instead of litigation",
            "accept the settlement offer of $2.5M",
            "file a motion to dismiss on jurisdictional grounds",
            "extend the contract review period by 30 days",
            "invoke the force majeure clause",
            "require additional representations and warranties",
        ],
        "issues": [
            "ambiguous indemnification language",
            "missing signatures on key documents",
            "statute of limitations concern",
            "conflict of interest identified",
            "breach of confidentiality agreement",
        ],
        "people": ["Attorney Davis", "Counsel Rivera", "Partner Johnson", "Associate Kim", "Paralegal Wong"],
        "task_templates": {
            "summarize_session": [
                "Summarize the legal team's discussion about the {entity}",
                "What were the key legal positions established regarding the {entity}?",
            ],
            "track_state": [
                "What is the current status of the {entity}?",
                "Where does the {entity} stand in the legal process?",
            ],
            "detect_contradiction": [
                "Are there contradictory positions taken regarding the {entity}?",
                "Identify inconsistencies in the {entity} documentation",
            ],
            "recall_procedure": [
                "What procedure was outlined for handling the {entity}?",
                "Describe the legal process agreed upon for the {entity}",
            ],
            "synthesize_facts": [
                "Compile all relevant legal facts about the {entity}",
                "Synthesize the legal position on {entity} from all context items",
            ],
            "answer_from_memory": [
                "What was the key argument made regarding the {entity}?",
                "What deadline applies to the {entity}?",
            ],
            "identify_gaps": [
                "What legal research is still needed for the {entity}?",
                "What information is missing from the {entity} case file?",
            ],
            "timeline_reconstruction": [
                "Reconstruct the procedural timeline for the {entity}",
                "In what order did legal events related to the {entity} occur?",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The legal team reviewed the {entity} and agreed to {decision}. The matter will be escalated to the partner level if not resolved by Friday.",
                "Session outcome: identified material risks in the {entity}, recommended additional due diligence, and flagged for senior review.",
            ],
            "track_state": [
                "The {entity} is currently in the discovery phase. Depositions are scheduled for next month and trial date is set for Q3.",
                "The {entity} has been executed by all parties. It is now in the monitoring phase to ensure compliance with its terms.",
            ],
            "detect_contradiction": [
                "Contradiction found: Section 4.2 grants unlimited liability indemnification while Section 12.1 caps liability at contract value.",
                "Inconsistency: the client asserted no prior IP filings, but due diligence revealed two pending patent applications in the same field.",
            ],
            "recall_procedure": [
                "Agreed procedure for {entity}: 1) internal review, 2) client briefing, 3) opposing counsel negotiation, 4) draft agreement, 5) execution and filing.",
            ],
            "synthesize_facts": [
                "The {entity} involves a $5M claim, was filed 8 months ago, is governed by New York law, and the primary dispute centers on breach of exclusivity provisions.",
            ],
            "answer_from_memory": [
                "The statute of limitations for the {entity} expires on March 15, requiring filing no later than March 10.",
            ],
            "identify_gaps": [
                "Outstanding items for {entity}: expert witness identification, document production from third party, and research on recent case law in this jurisdiction.",
            ],
            "timeline_reconstruction": [
                "Legal timeline: contract signed (Jan), breach alleged (Mar), demand letter sent (Apr), response received (May), mediation attempted (Jul), litigation filed (Sep).",
            ],
        },
    },

    "finance": {
        "entities": [
            "investment portfolio", "quarterly earnings", "budget allocation", "risk assessment",
            "loan application", "financial model", "audit finding", "cash flow",
            "merger valuation", "tax strategy", "compliance report", "trading position",
        ],
        "actions": [
            "approved", "rejected", "allocated", "divested", "hedged", "audited",
            "forecasted", "rebalanced", "written off", "capitalized", "provisioned", "reconciled",
        ],
        "decisions": [
            "increase equity allocation by 10%",
            "hedge the currency exposure with forward contracts",
            "defer the capital expenditure to Q2",
            "write down the asset by $1.2M",
            "pursue the acquisition at a 12x EBITDA multiple",
            "implement zero-based budgeting for the next fiscal year",
            "repatriate offshore cash holdings",
        ],
        "issues": [
            "unexpected variance in operating expenses",
            "covenant breach on the credit facility",
            "delayed revenue recognition due to contract dispute",
            "FX exposure not properly hedged",
            "underfunded pension liability discovered during audit",
        ],
        "people": ["CFO Anderson", "Analyst Chen", "Controller Patel", "Treasurer Moore", "Auditor Walsh"],
        "task_templates": {
            "summarize_session": [
                "Summarize the financial decisions made about the {entity}",
                "What were the key takeaways from the discussion about the {entity}?",
            ],
            "track_state": [
                "What is the current status of the {entity}?",
                "Where does the {entity} stand relative to targets?",
            ],
            "detect_contradiction": [
                "Are there contradictory figures or positions about the {entity}?",
                "Identify inconsistencies in the {entity} reporting",
            ],
            "recall_procedure": [
                "What financial procedure was described for the {entity}?",
                "Describe the approval process outlined for the {entity}",
            ],
            "synthesize_facts": [
                "Compile all financial information about the {entity} from the context",
                "Synthesize the financial position on {entity} from all available data",
            ],
            "answer_from_memory": [
                "What was the approved budget for the {entity}?",
                "What return was projected for the {entity}?",
            ],
            "identify_gaps": [
                "What financial analysis is still needed for the {entity}?",
                "What data is missing for a complete assessment of the {entity}?",
            ],
            "timeline_reconstruction": [
                "Reconstruct the financial timeline for the {entity}",
                "In what order did financial events related to the {entity} occur?",
            ],
        },
        "ground_truth_templates": {
            "summarize_session": [
                "The finance team reviewed the {entity} and decided to {decision}. The action will be implemented in the next reporting period.",
                "Session summary: {entity} analysis revealed a 15% variance from forecast. Decision made to reforecast for Q4 with conservative assumptions.",
            ],
            "track_state": [
                "The {entity} is currently 8% below the annual target. YTD performance shows improvement in operating margins despite revenue shortfall.",
                "The {entity} is on track. Q3 results came in at $4.2M against a $4.0M forecast, driven by stronger than expected volume.",
            ],
            "detect_contradiction": [
                "Contradiction: the revenue figure reported in the board deck ($12.4M) differs from the management accounts ($11.9M) for the same period.",
                "Inconsistency in {entity}: the budget model uses 8% growth assumption while the strategic plan assumes 12% for the same line.",
            ],
            "recall_procedure": [
                "Approval process for {entity}: 1) analyst prepares model, 2) controller reviews assumptions, 3) CFO approval if over $500K, 4) board notification if over $2M.",
            ],
            "synthesize_facts": [
                "The {entity} has a current value of $8.3M, was established in 2019, has returned 11.2% annualized, and carries moderate risk per the latest assessment.",
            ],
            "answer_from_memory": [
                "The approved budget for the {entity} was $2.8M for the fiscal year, with a contingency reserve of $200K.",
            ],
            "identify_gaps": [
                "Missing for {entity} analysis: Q4 cash flow projections, sensitivity analysis on interest rate assumptions, and board approval documentation.",
            ],
            "timeline_reconstruction": [
                "Financial timeline: budget set (Jan), Q1 close (Apr), mid-year review (Jul), reforecast (Aug), Q3 close (Oct), year-end audit (Dec).",
            ],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Context item content generators
# ─────────────────────────────────────────────────────────────────────────────

def make_context_item_content(domain, entity, item_type, idx):
    dc = DOMAIN_CONTENT[domain]
    person = random.choice(dc["people"])
    action = random.choice(dc["actions"])
    decision = random.choice(dc["decisions"])
    issue = random.choice(dc["issues"])
    other_person = random.choice(dc["people"])

    templates = {
        "observation": [
            f"{person} observed that the {entity} is showing signs of {random.choice(['improvement','regression','instability','progress'])}.",
            f"Current metrics for the {entity} indicate {random.choice(['above','below','on'])} target performance.",
            f"The {entity} was {action} successfully by {person} at {random.choice(['09:00','11:30','14:00','16:45'])}.",
            f"Status update: {entity} has been {action}. No critical issues reported.",
            f"{person} noted that the {entity} requires attention due to {issue}.",
        ],
        "decision": [
            f"Team agreed to {decision} regarding the {entity}.",
            f"{person} made the final call: {decision}.",
            f"After discussion, the group decided to {decision} for the {entity}.",
            f"Decision recorded: {decision}. Responsible party: {person}.",
            f"The {entity} approach was finalized: {decision}.",
        ],
        "statement": [
            f"{person} stated: 'The {entity} needs to be addressed before the deadline.'",
            f"{other_person} clarified that the {entity} {random.choice(['meets','does not meet','partially meets'])} the requirements.",
            f"{person} confirmed that the {entity} has been reviewed and {random.choice(['approved','rejected','sent back for revision'])}.",
            f"According to {person}, the {entity} performance is {random.choice(['excellent','satisfactory','below expectations','critical'])}.",
            f"{person} reported: the {entity} encountered {issue} which has since been {random.choice(['resolved','escalated','under investigation'])}.",
        ],
        "action": [
            f"{person} {action} the {entity} as part of the scheduled maintenance.",
            f"Action taken: {person} {action} the {entity} in response to {issue}.",
            f"The {entity} was {action} by {other_person} following the review meeting.",
            f"{person} completed the {action} of the {entity}. Documentation updated.",
            f"Emergency action: {entity} was {action} by {person} to prevent further issues.",
        ],
        "question": [
            f"{person} asked: 'What is the current status of the {entity}?'",
            f"Open question from {other_person}: 'How should we handle the {issue} in the {entity}?'",
            f"{person} raised: 'Should we {decision} for the {entity}?'",
            f"Unresolved: who is responsible for the {entity} after {person} leaves the project?",
            f"{person} queried: 'What are the performance targets for the {entity} in Q4?'",
        ],
        "fact": [
            f"The {entity} was established in {random.randint(2018, 2024)} and has been in continuous operation since.",
            f"Performance data: the {entity} handles {random.randint(100, 100000)} transactions per {random.choice(['hour','day','week'])}.",
            f"The {entity} is owned by {person} and is classified as {random.choice(['critical','high priority','standard','low priority'])}.",
            f"Regulatory requirement: the {entity} must comply with {random.choice(['ISO 27001','SOC 2','HIPAA','GDPR','PCI-DSS'])}.",
            f"The {entity} has {random.randint(2, 20)} dependencies and {random.randint(1, 8)} known stakeholders.",
        ],
        "event": [
            f"Event logged: {entity} experienced {issue} at {random.choice(['02:14','08:33','12:07','17:55','23:41'])} UTC.",
            f"Milestone reached: {entity} successfully {action} for the first time in production.",
            f"{person} completed the review of {entity}. Result: {random.choice(['passed','failed','conditional pass'])}.",
            f"Scheduled event: {entity} will be {action} during the maintenance window on Saturday.",
            f"Incident: {entity} was unavailable for {random.randint(5, 180)} minutes due to {issue}.",
        ],
        "note": [
            f"Note from {person}: the {entity} may need re-evaluation after the next quarter.",
            f"Reminder: {entity} SLA review is due in {random.randint(1, 30)} days.",
            f"Follow-up required: {person} to provide update on {entity} by {random.choice(['Monday','Wednesday','Friday','EOD'])}.",
            f"Context note: the {entity} was previously {action} under a different owner.",
            f"Historical context: the {entity} has been {random.choice(['reliable','problematic','underperforming','exceeding expectations'])} for the past {random.randint(1,24)} months.",
        ],
    }

    return random.choice(templates.get(item_type, templates["observation"]))


def generate_context_items(domain, entity, n_items):
    base_ts = int(time.time()) - random.randint(86400, 86400 * 90)
    items = []
    for i in range(n_items):
        item_type = random.choice(ITEM_TYPES)
        ts = base_ts + i * random.randint(60, 3600)
        content = make_context_item_content(domain, entity, item_type, i)
        importance = round(random.uniform(0.1, 1.0), 2)
        items.append({
            "id": f"ctx-{i+1:03d}",
            "content": content,
            "type": item_type,
            "timestamp": ts,
            "importance": importance,
        })
    return items


def pick_relevant_ids(context_items, n_relevant):
    # Prefer high-importance items as relevant
    sorted_items = sorted(context_items, key=lambda x: x["importance"], reverse=True)
    pool = [it["id"] for it in sorted_items[:max(n_relevant * 2, 5)]]
    chosen = random.sample(pool, min(n_relevant, len(pool)))
    return chosen


def estimate_tokens(context_items, task):
    total_chars = len(task) + sum(len(it["content"]) for it in context_items)
    return int(total_chars / 4)


def generate_example(example_id, split):
    domain = random.choice(DOMAINS)
    task_type = random.choice(TASK_TYPES)
    dc = DOMAIN_CONTENT[domain]

    entity = random.choice(dc["entities"])
    person = random.choice(dc["people"])
    person2 = random.choice(dc["people"])
    decision = random.choice(dc["decisions"])
    tech_options = ["PostgreSQL", "Redis", "Kafka", "Elasticsearch", "MongoDB", "DynamoDB"]
    tech = random.choice(tech_options)

    # Select task template
    task_tmpl_list = dc["task_templates"].get(task_type, dc["task_templates"]["summarize_session"])
    task_tmpl = random.choice(task_tmpl_list)
    task = task_tmpl.format(entity=entity, person=person, decision=decision)

    # Select ground truth template
    gt_tmpl_list = dc["ground_truth_templates"].get(task_type, dc["ground_truth_templates"]["summarize_session"])
    gt_tmpl = random.choice(gt_tmpl_list)
    ground_truth = gt_tmpl.format(
        entity=entity, person=person, person2=person2,
        decision=decision, tech=tech,
    )

    # Number of context items: 10-50
    n_items = random.randint(10, 50)
    context_items = generate_context_items(domain, entity, n_items)

    # Difficulty
    if n_items <= 20:
        difficulty = "easy"
    elif n_items <= 35:
        difficulty = "medium"
    else:
        difficulty = "hard"

    n_relevant = random.randint(2, min(8, n_items))
    relevant_ids = pick_relevant_ids(context_items, n_relevant)
    token_count = estimate_tokens(context_items, task)

    return {
        "id": example_id,
        "task": task,
        "context_items": context_items,
        "ground_truth": ground_truth,
        "relevant_item_ids": relevant_ids,
        "metadata": {
            "task_type": task_type,
            "domain": domain,
            "context_length_tokens": token_count,
            "num_relevant_items": len(relevant_ids),
            "difficulty": difficulty,
        },
    }


def write_split(split, n_examples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Generating {n_examples} examples for '{split}' -> {output_path}")
    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_examples):
            example_id = f"cb-{split}-{i+1:05d}"
            example = generate_example(example_id, split)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{n_examples} examples written ({elapsed:.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"  Done: {n_examples} examples in {elapsed:.1f}s")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits = [
        ("train", 60000, os.path.join(base_dir, "context_bench_train.jsonl")),
        ("val",   5000,  os.path.join(base_dir, "context_bench_val.jsonl")),
        ("test",  5000,  os.path.join(base_dir, "context_bench_test.jsonl")),
    ]

    print("=" * 60)
    print("ContextBench-Long Dataset Generator")
    print(f"Seed: {SEED}")
    print(f"Total examples: {sum(n for _, n, _ in splits):,}")
    print("=" * 60)

    total_t0 = time.time()
    for split, n, path in splits:
        write_split(split, n, path)

    total_elapsed = time.time() - total_t0
    print("=" * 60)
    print(f"All splits generated in {total_elapsed:.1f}s")

    # Print file sizes
    for _, _, path in splits:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {os.path.basename(path)}: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
