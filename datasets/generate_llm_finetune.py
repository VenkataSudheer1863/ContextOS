"""
Generate instruction fine-tuning datasets for LLMs (Qwen3, GLM-4.5).
Outputs:
  datasets/llm_finetune_train.jsonl  (25,000 samples)
  datasets/llm_finetune_val.jsonl    ( 2,500 samples)
  datasets/llm_finetune_test.jsonl   ( 2,500 samples)

Format per record:
{
  "instruction": "...",
  "input": "<context>...</context>\n\nQuestion: ...",
  "output": "Based on the context provided, ...",
  "metadata": {"task_type": "...", "context_tokens": int, "model": "..."}
}
"""

import json
import random
import os

random.seed(99)

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

MODELS = ["qwen3", "glm-4.5"]

TASK_TYPES = [
    "factual_recall",
    "decision_retrieval",
    "state_tracking",
    "procedure_lookup",
    "timeline_query",
    "multi_hop_reasoning",
    "summarisation",
    "comparison",
    "causal_reasoning",
    "negative_lookup",    # query whose answer is NOT in context
]

DOMAINS = {
    "software_engineering": {
        "memory_items": [
            "2024-03-15 | Decision | OAuth2 with PKCE flow was approved for all API authentication endpoints after a security review. Alternatives Auth0 and Cognito were considered but rejected due to vendor lock-in concerns.",
            "2024-03-16 | State | The user-service is currently running version 2.4.1. A migration to 2.5.0 is scheduled for the next maintenance window on Saturday at 02:00 UTC.",
            "2024-03-17 | Procedure | Deployment process: (1) merge PR to main, (2) CI builds Docker image and tags with git SHA, (3) ArgoCD syncs to staging, (4) QA sign-off required within 2 hours, (5) promote to production via ArgoCD.",
            "2024-03-18 | Decision | PostgreSQL 15 with row-level partitioning was chosen over MongoDB for the orders table. ACID compliance was the primary driver. Estimated 60% query performance improvement over the legacy schema.",
            "2024-03-19 | Event | Kubernetes autoscaler triggered: API pods scaled from 6 to 18 replicas at 14:32 UTC in response to Black Friday traffic. Peak RPS reached 42,000. No SLA breach observed.",
            "2024-03-20 | Decision | Redis Cluster with 3 primary shards was provisioned for session storage. TTL set to 24 hours. Eviction policy: allkeys-lru.",
            "2024-03-21 | State | Message queue depth for the `order-events` topic is currently 1.8 million messages. Consumer lag alert has been acknowledged; additional consumer instances are being deployed.",
            "2024-03-22 | Procedure | Incident response runbook: page on-call via PagerDuty, open #incident-YYYY-MM-DD Slack channel, assign incident commander, declare severity within 15 minutes, post updates every 30 minutes.",
            "2024-03-23 | Decision | gRPC with Protobuf v3 was standardised for all inter-service communication. REST kept only for public-facing APIs. Measured 40% latency reduction in inventory-to-warehouse calls.",
            "2024-03-24 | Event | Database failover occurred at 03:14 UTC. Replica promoted in 47 seconds. Root cause: primary host disk I/O saturation. Post-mortem scheduled for Monday.",
            "2024-03-25 | State | Feature flag `new_checkout_v2` is enabled for 15% of users. Conversion rate improvement: +3.2%. Rollout to 50% is gated on p99 latency remaining below 200ms.",
            "2024-03-26 | Decision | Elasticsearch 8.x deployed for full-text and semantic search. kNN vector search enabled on the product catalogue. BM25 hybrid scoring weight: 0.4 lexical + 0.6 dense.",
            "2024-03-27 | Procedure | On-boarding checklist: provision IAM role with least-privilege policy, grant GitHub team membership, set up MFA, clone mono-repo, run `make dev-up` to start local services via Docker Compose.",
            "2024-03-28 | State | TLS certificate for api.example.com expires in 14 days. Auto-renewal via cert-manager is configured; manual fallback procedure documented in runbook.",
            "2024-03-29 | Decision | Istio service mesh adopted for mTLS, circuit breaking, and distributed tracing. Sidecar injection enabled on all namespaces except `legacy-jobs`.",
        ],
        "questions": {
            "factual_recall": [
                ("What authentication approach was chosen for the API?",
                 "OAuth2 with PKCE flow was approved for all API authentication endpoints. Auth0 and Cognito were evaluated but rejected to avoid vendor lock-in."),
                ("What database was selected for the orders table and why?",
                 "PostgreSQL 15 with row-level partitioning was chosen over MongoDB. The primary driver was ACID compliance, with an estimated 60% query performance improvement."),
                ("What are the Redis Cluster configuration details?",
                 "Redis Cluster was provisioned with 3 primary shards for session storage. The TTL is set to 24 hours and the eviction policy is allkeys-lru."),
            ],
            "decision_retrieval": [
                ("What inter-service communication protocol was decided upon?",
                 "gRPC with Protobuf v3 was standardised for all inter-service communication, replacing REST for internal calls. This delivered a 40% latency reduction on inventory-to-warehouse traffic."),
                ("What search technology was deployed and how is scoring configured?",
                 "Elasticsearch 8.x was deployed for full-text and semantic search with kNN vector search on the product catalogue. Hybrid scoring uses a 0.4 lexical and 0.6 dense weighting."),
                ("Which service mesh was adopted and what does it provide?",
                 "Istio service mesh was adopted, providing mTLS, circuit breaking, and distributed tracing. Sidecar injection is enabled on all namespaces except `legacy-jobs`."),
            ],
            "state_tracking": [
                ("What is the current state of the `new_checkout_v2` feature flag?",
                 "The feature flag `new_checkout_v2` is currently enabled for 15% of users. It shows a +3.2% conversion rate improvement. Rollout to 50% is gated on p99 latency staying below 200ms."),
                ("What is the current message queue depth for order-events?",
                 "The `order-events` topic queue depth is currently 1.8 million messages. The consumer lag alert has been acknowledged and additional consumer instances are being deployed."),
                ("How long until the TLS certificate expires?",
                 "The TLS certificate for api.example.com expires in 14 days. Auto-renewal via cert-manager is configured, with a manual fallback procedure in the runbook."),
            ],
            "procedure_lookup": [
                ("What are the steps to deploy a new service version?",
                 "The deployment process is: (1) merge PR to main, (2) CI builds and tags a Docker image with the git SHA, (3) ArgoCD syncs to staging, (4) QA sign-off required within 2 hours, (5) promote to production via ArgoCD."),
                ("What is the incident response procedure?",
                 "Incident response: page on-call via PagerDuty, open a dedicated Slack channel, assign an incident commander, declare severity within 15 minutes, and post updates every 30 minutes."),
                ("How should a new engineer be on-boarded?",
                 "On-boarding steps: provision an IAM role with least-privilege policy, grant GitHub team membership, set up MFA, clone the mono-repo, and run `make dev-up` to start local services via Docker Compose."),
            ],
            "timeline_query": [
                ("When did the database failover occur and what was the impact?",
                 "The database failover occurred at 03:14 UTC on 2024-03-24. The replica was promoted in 47 seconds. The root cause was primary host disk I/O saturation, with a post-mortem scheduled for Monday."),
                ("What happened during the Black Friday traffic event?",
                 "On 2024-03-19, Kubernetes autoscaler scaled API pods from 6 to 18 replicas at 14:32 UTC in response to Black Friday traffic. Peak RPS reached 42,000 with no SLA breach."),
            ],
            "negative_lookup": [
                ("What is the password policy for service accounts?",
                 "The provided context does not contain information about the service account password policy. This detail was not recorded in the available memory items."),
                ("What is the allocated budget for the Kubernetes cluster?",
                 "The context does not include any budget figures for the Kubernetes cluster. A separate budget or finance document would need to be consulted."),
            ],
        },
    },
    "research": {
        "memory_items": [
            "2024-04-01 | Decision | Hybrid retrieval (BM25 + dense bi-encoder) was selected as the retrieval baseline. Pure dense retrieval achieved 68.2 nDCG@10 on BEIR; hybrid improved this to 72.4.",
            "2024-04-02 | State | Experiment run #47 is queued behind two higher-priority jobs on the A100 cluster. ETA for GPU allocation: 6 hours. Config stored in MLflow run ID e3a9f.",
            "2024-04-03 | Decision | Contrastive loss with in-batch negatives was selected over triplet loss. Convergence was 30% faster on MS-MARCO with identical final performance.",
            "2024-04-04 | Procedure | Ablation protocol: fix all hyperparameters, disable one component at a time across three random seeds, report mean delta and 95% CI over the full model baseline.",
            "2024-04-05 | Event | Fine-tuned checkpoint at step 50,000 evaluated: 3.2% NDCG improvement over BGE-M3 baseline. Training resumed with learning rate warmup reset.",
            "2024-04-06 | Decision | Context window fixed at 8,192 tokens after a sweep from 2,048 to 16,384. Recall improved marginally beyond 8k but GPU memory cost doubled.",
            "2024-04-07 | State | Human evaluation study is in progress. 5 annotators have completed 60% of 500 comparison pairs. Fleiss kappa is currently 0.73, above the 0.70 threshold.",
            "2024-04-08 | Decision | GPT-4o was selected as the automated evaluation judge, achieving 0.89 Spearman correlation with human raters versus 0.81 for Claude 3 Opus.",
            "2024-04-09 | Procedure | Experiment registration: create config YAML, register in MLflow, submit SLURM job with `sbatch train.sh`, monitor via Grafana dashboard, archive final checkpoint to S3.",
            "2024-04-10 | Event | Ablation removing positional embeddings in the cross-attention layer yielded a +2.1 BLEU score improvement. Result marked as significant; architecture decision to be revisited.",
        ],
        "questions": {
            "factual_recall": [
                ("What retrieval strategy was selected as the baseline and what were its scores?",
                 "Hybrid retrieval combining BM25 and a dense bi-encoder was selected as the baseline. Pure dense retrieval achieved 68.2 nDCG@10 on BEIR, which the hybrid approach improved to 72.4."),
                ("What context window size was chosen and why?",
                 "The context window was fixed at 8,192 tokens after sweeping from 2,048 to 16,384. Recall improved only marginally beyond 8k while GPU memory cost doubled, making 8k the optimal trade-off."),
            ],
            "decision_retrieval": [
                ("Which loss function was chosen for training the embedding model?",
                 "Contrastive loss with in-batch negatives was chosen over triplet loss. It converged 30% faster on MS-MARCO while achieving identical final performance."),
                ("Which model was selected as the automated evaluation judge?",
                 "GPT-4o was selected as the automated evaluation judge. It achieved 0.89 Spearman correlation with human raters, outperforming Claude 3 Opus at 0.81."),
            ],
            "state_tracking": [
                ("What is the current status of experiment run #47?",
                 "Experiment run #47 is currently queued behind two higher-priority A100 cluster jobs. The expected GPU allocation is in approximately 6 hours. Its configuration is stored in MLflow run ID e3a9f."),
                ("How far along is the human evaluation study?",
                 "The human evaluation study is 60% complete, with 5 annotators having reviewed 300 of 500 comparison pairs. The current Fleiss kappa is 0.73, which exceeds the 0.70 threshold."),
            ],
            "procedure_lookup": [
                ("What is the procedure for registering and running a new experiment?",
                 "The experiment registration procedure is: create a config YAML, register it in MLflow, submit a SLURM job via `sbatch train.sh`, monitor progress through the Grafana dashboard, and archive the final checkpoint to S3."),
                ("What is the ablation study protocol?",
                 "The ablation protocol is: fix all hyperparameters, disable one component at a time across three random seeds, then report the mean delta and 95% confidence interval relative to the full model baseline."),
            ],
            "timeline_query": [
                ("What was observed at checkpoint step 50,000?",
                 "At step 50,000, the fine-tuned checkpoint showed a 3.2% NDCG improvement over the BGE-M3 baseline. Training was then resumed with a learning rate warmup reset."),
                ("What happened when positional embeddings were removed from the cross-attention layer?",
                 "The ablation removing positional embeddings from the cross-attention layer yielded a +2.1 BLEU improvement. The result was marked as statistically significant, triggering a revisit of the architecture decision."),
            ],
            "negative_lookup": [
                ("What was the batch size used during training?",
                 "The context does not specify the batch size used during training. This hyperparameter was not recorded in the available memory items."),
            ],
        },
    },
    "healthcare": {
        "memory_items": [
            "2024-05-01 | Decision | Cardiology team approved anticoagulation therapy: apixaban 5mg twice daily for patient with persistent AF and CHA2DS2-VASc score of 3.",
            "2024-05-02 | Procedure | Sepsis bundle: blood cultures x2 before antibiotics, measure serum lactate, administer 30 ml/kg IV crystalloid bolus, start broad-spectrum antibiotics within 1 hour of recognition.",
            "2024-05-03 | State | Patient in Bay 4: potassium 3.1 mEq/L, potassium supplementation ordered, cardiology consult pending. Next scheduled labs at 18:00.",
            "2024-05-04 | Decision | Ethics committee approved Phase II immunotherapy trial (N=120). Primary endpoint: ORR at 24 weeks. Enrollment opens next Monday.",
            "2024-05-05 | Event | MRI scanner in Suite B offline for calibration until 14:00. Urgent studies diverted to Suite A. Backlog of 6 non-urgent scans rescheduled.",
            "2024-05-06 | State | Clinical trial enrollment: 87 of 120 participants. Interim analysis scheduled when 100th participant reaches week 12.",
            "2024-05-07 | Procedure | Pre-operative checklist: confirm NPO status >6 hours, verify signed consent, surgeon marks site, anaesthesia pre-assessment complete, prophylactic antibiotics 30-60 min before incision, surgical timeout.",
            "2024-05-08 | Decision | Tumour board recommended FOLFOX chemotherapy followed by surgical resection for Stage IIIB colorectal adenocarcinoma. Response assessment after 4 cycles.",
            "2024-05-09 | State | Pharmacy: linagliptin added to preferred DPP-4 inhibitor tier on formulary effective Q3. Prior authorisation required for non-preferred agents.",
            "2024-05-10 | Event | Discharge planning: post-hip-replacement patient discharged home with home health nursing 3x/week, PT referral, follow-up with orthopaedics in 2 weeks.",
        ],
        "questions": {
            "factual_recall": [
                ("What anticoagulation therapy was approved for the AF patient?",
                 "Apixaban 5mg twice daily was approved for the patient with persistent atrial fibrillation. The decision was based on a CHA2DS2-VASc score of 3."),
                ("What are the details of the approved Phase II trial?",
                 "The ethics committee approved a Phase II immunotherapy trial with 120 participants. The primary endpoint is overall response rate at 24 weeks, with enrollment opening the following Monday."),
            ],
            "state_tracking": [
                ("What is the current status of the Bay 4 patient?",
                 "The patient in Bay 4 has a potassium level of 3.1 mEq/L. Potassium supplementation has been ordered, a cardiology consult is pending, and the next labs are scheduled for 18:00."),
                ("What is the current trial enrollment status?",
                 "The clinical trial has enrolled 87 of its 120 target participants. The interim analysis is scheduled when the 100th participant reaches their week 12 visit."),
            ],
            "procedure_lookup": [
                ("What is the sepsis management bundle?",
                 "The sepsis bundle includes: drawing blood cultures x2 before starting antibiotics, measuring serum lactate, giving a 30 ml/kg IV crystalloid bolus, and starting broad-spectrum antibiotics within 1 hour of recognition."),
                ("What does the pre-operative checklist include?",
                 "The pre-op checklist covers: confirming NPO status for over 6 hours, verifying signed consent, surgeon site marking, completed anaesthesia pre-assessment, prophylactic antibiotics 30-60 minutes before incision, and a surgical timeout."),
            ],
            "timeline_query": [
                ("What post-discharge arrangements were made for the hip replacement patient?",
                 "The hip replacement patient was discharged home with home health nursing three times per week, a physiotherapy referral, and a follow-up appointment with orthopaedics scheduled for two weeks later."),
            ],
            "negative_lookup": [
                ("What is the patient's full medication list?",
                 "The context does not contain a comprehensive medication list for the patient. Only the newly approved apixaban and the pending potassium supplementation are documented in the available memory items."),
            ],
        },
    },
    "legal": {
        "memory_items": [
            "2024-06-01 | Decision | Outside counsel recommended settling the patent dispute for $4.2M. Litigation risk assessed at 35% probability of adverse verdict; settlement avoids 18+ months of proceedings.",
            "2024-06-02 | State | Discovery: 120,000 document set under review; 48,000 (40%) processed. Privilege review ongoing. Responses due in 21 days.",
            "2024-06-03 | Decision | Data processing agreement amended to include SCCs (Standard Contractual Clauses) for EU data transfers following Schrems II guidance. Effective immediately.",
            "2024-06-04 | Procedure | Contract review workflow: initial red-line by associate, senior associate review of defined terms, partner sign-off on indemnification, IP, and liability caps, then legal ops upload to contract repository with metadata tags.",
            "2024-06-05 | State | Patent application US-2024-XXXXX is in examination. Office action received; response due by the 15th of next month. Claim 3 requires amendment per examiner guidance.",
            "2024-06-06 | Decision | Board approved Delaware reincorporation. Rationale: established corporate law, Court of Chancery expertise, and investor familiarity. Shareholder vote scheduled Q3.",
            "2024-06-07 | State | Merger agreement in final markup. Three open issues: (1) indemnification cap at 15% vs 20% of deal value, (2) rep survival period 18 vs 24 months, (3) definition of Material Adverse Effect.",
            "2024-06-08 | Decision | Non-compete clause deemed unenforceable in California. Legal issued guidance: excise non-compete from all California offer letters; replace with narrowly tailored non-solicitation clause.",
            "2024-06-09 | Procedure | Litigation hold: identify custodians, issue hold notice within 24 hours of trigger event, disable auto-deletion in email and Slack, collect ESI from key custodians within 7 days, log all hold notices.",
            "2024-06-10 | Decision | Arbitration clause modified to permit class arbitration following AAA policy update. Amendment executed and appended to master services agreement.",
        ],
        "questions": {
            "decision_retrieval": [
                ("What was the recommendation regarding the patent dispute?",
                 "Outside counsel recommended settling the patent dispute for $4.2M. The litigation risk was assessed at a 35% probability of an adverse verdict, and settlement avoids over 18 months of proceedings."),
                ("What change was made to the non-compete clause in California?",
                 "The non-compete clause was deemed unenforceable in California. Legal issued guidance to excise it from all California offer letters and replace it with a narrowly tailored non-solicitation clause."),
            ],
            "state_tracking": [
                ("What is the current status of the discovery review?",
                 "Of the 120,000-document discovery set, 48,000 documents (40%) have been processed. Privilege review is ongoing and responses are due within 21 days."),
                ("What open issues remain in the merger agreement?",
                 "Three issues remain open: (1) the indemnification cap, contested at 15% vs 20% of deal value; (2) the rep survival period, disputed at 18 vs 24 months; and (3) the definition of Material Adverse Effect."),
            ],
            "procedure_lookup": [
                ("What is the contract review workflow?",
                 "The contract review process is: initial red-line by an associate, senior associate review of defined terms, partner sign-off on indemnification, IP, and liability caps, followed by legal ops uploading to the contract repository with metadata tags."),
                ("What are the steps for issuing a litigation hold?",
                 "The litigation hold procedure is: identify custodians, issue a hold notice within 24 hours of the trigger event, disable auto-deletion in email and Slack, collect ESI from key custodians within 7 days, and log all hold notices."),
            ],
        },
    },
    "finance": {
        "memory_items": [
            "2024-07-01 | Decision | Investment committee approved $50M allocation to infrastructure debt fund at 7.2% target IRR. 5-year lock-up. Investment memo on file.",
            "2024-07-02 | State | Q3 revenue tracking $8M below plan ($142M actual vs $150M target). Revised forecast submitted to board. Key variance: enterprise deal slippage.",
            "2024-07-03 | Decision | CFO approved $12M capex for SAP S/4HANA ERP upgrade phased over 18 months. Phase 1 (finance module) starts Q4.",
            "2024-07-04 | Procedure | Monthly close process: sub-ledger reconciliations by business day 3, intercompany eliminations by day 5, trial balance reviewed by controller day 6, MD&A draft by day 8, CFO sign-off day 10.",
            "2024-07-05 | State | Revolving credit facility: $130M drawn of $200M facility (65% utilisation). Leverage covenant requires net debt/EBITDA below 3.5x. Current ratio: 2.8x.",
            "2024-07-06 | Decision | Board authorised $200M share buyback over 24 months via 10b5-1 plan. First tranche: $50M in Q4. Programme managed by Goldman Sachs.",
            "2024-07-07 | State | Tax provision adjusted by $2.3M following transfer pricing study. Deferred tax liability increased. External tax counsel review scheduled before filing.",
            "2024-07-08 | Decision | Treasury approved hedging 70% of EUR/USD exposure for 12 months using vanilla FX forwards. Notional: €84M. Average forward rate: 1.082.",
            "2024-07-09 | Procedure | Expense reimbursement policy: submit receipts within 30 days via Concur, manager approval required, finance audit for items over $500, CFO approval for items over $5,000.",
            "2024-07-10 | Event | Audit committee accepted management's assessment that the internal control deficiency (manual journal approval threshold) is not material. Remediation plan due Q1 next year.",
        ],
        "questions": {
            "decision_retrieval": [
                ("What investment was approved by the investment committee?",
                 "The investment committee approved a $50M allocation to an infrastructure debt fund targeting a 7.2% IRR, with a 5-year lock-up period."),
                ("What was the board's decision on share buybacks?",
                 "The board authorised a $200M share buyback programme to be executed over 24 months via a 10b5-1 plan. The first tranche of $50M is planned for Q4, managed by Goldman Sachs."),
            ],
            "state_tracking": [
                ("What is the Q3 revenue status?",
                 "Q3 revenue is tracking $8M below plan, at $142M actual against a $150M target. The key variance driver is enterprise deal slippage. A revised forecast has been submitted to the board."),
                ("What is the current utilisation of the revolving credit facility?",
                 "The revolving credit facility is 65% utilised at $130M drawn of a $200M facility. The leverage covenant requires net debt/EBITDA to remain below 3.5x; the current ratio is 2.8x, within compliance."),
            ],
            "procedure_lookup": [
                ("What is the monthly financial close timeline?",
                 "The monthly close timeline is: sub-ledger reconciliations by day 3, intercompany eliminations by day 5, trial balance review by the controller on day 6, MD&A draft by day 8, and CFO sign-off on day 10."),
            ],
        },
    },
    "customer_service": {
        "memory_items": [
            "2024-08-01 | Decision | Support leadership extended refund window from 30 to 60 days for all premium subscribers effective immediately. Standard plan remains at 30 days.",
            "2024-08-02 | State | P1 queue: 7 open tickets, average wait 18 minutes. SLA target is 4-hour first response. Oldest open ticket is 2.5 hours old.",
            "2024-08-03 | Procedure | Escalation procedure: agent documents all steps taken, tags ticket as `escalated`, assigns to tier-2 queue, sets customer ETA to 4 hours, sends acknowledgement email.",
            "2024-08-04 | Decision | Chatbot configured to hand off to human agent after 2 failed resolution attempts (reduced from 3). Change aimed at reducing repeat contacts.",
            "2024-08-05 | State | CSAT for billing team: 4.3/5.0 this week, up from 3.8 last week. Improvement attributed to new escalation training rollout.",
            "2024-08-06 | Event | Product recall notice sent to 14,200 affected customers via email and in-app notification within 4 hours of engineering confirmation. 1,840 customers have responded.",
            "2024-08-07 | Decision | SLA revised: P1 first response 4 hours (from 8), P2 first response 24 hours, P3 72 hours. Effective next quarter. Staffing plan under review.",
            "2024-08-08 | State | Chatbot deflection rate: 61% this week. Target is 70% by end of quarter. Two new FAQ flows (billing disputes and password reset) are in development.",
            "2024-08-09 | Procedure | Refund processing: verify purchase in order management system, confirm eligibility per refund policy, initiate refund in payment gateway (2-5 business days), send confirmation email to customer, update ticket.",
            "2024-08-10 | Procedure | Knowledge base update cycle: agent identifies content gap, drafts article in draft folder, peer review by senior agent, QA approval, publish and tag related open tickets for proactive customer outreach.",
        ],
        "questions": {
            "decision_retrieval": [
                ("What change was made to the refund window policy?",
                 "The refund window was extended from 30 to 60 days for all premium subscribers, effective immediately. The standard plan remains at 30 days."),
                ("What change was made to the chatbot escalation threshold?",
                 "The chatbot was reconfigured to hand off to a human agent after 2 failed resolution attempts, reduced from the previous threshold of 3. The goal is to reduce repeat contacts."),
            ],
            "state_tracking": [
                ("What is the current P1 queue status?",
                 "There are currently 7 open P1 tickets with an average wait time of 18 minutes. The oldest open ticket is 2.5 hours old, against a 4-hour SLA first response target."),
                ("What is the current chatbot deflection rate and the target?",
                 "The chatbot deflection rate is currently 61%. The target is 70% by end of quarter. Two new FAQ flows for billing disputes and password reset are in development."),
            ],
            "procedure_lookup": [
                ("What is the ticket escalation procedure?",
                 "The escalation procedure is: document all steps taken, tag the ticket as `escalated`, assign it to the tier-2 queue, set the customer ETA to 4 hours, and send an acknowledgement email."),
                ("What are the steps to process a refund?",
                 "Refund processing steps: verify the purchase in the order management system, confirm eligibility per policy, initiate the refund in the payment gateway (2-5 business days), send a confirmation email, and update the ticket."),
            ],
        },
    },
}

INSTRUCTION_VARIANTS = [
    "You are a context-aware agent. Given the following memory context, answer the question.",
    "You are an intelligent assistant with access to episodic memory. Use the provided context to answer accurately.",
    "You are a context retrieval system. Analyse the memory items below and respond to the query.",
    "You are a helpful AI assistant. The following context contains relevant memory entries. Answer based only on what is provided.",
    "You are a knowledge assistant. Review the context and provide a precise answer to the question asked.",
    "Given the following memory context from the ContextOS system, answer the user's question as accurately as possible.",
    "You have access to the following retrieved memory context. Use it to answer the question precisely.",
    "As a context-aware AI, use the provided memory items to answer the following question.",
]

CONTEXT_TOKEN_BUCKETS = [512, 1024, 2048, 4096, 8192]


def build_context_block(domain_name, n_items=None):
    items = DOMAINS[domain_name]["memory_items"]
    if n_items is None:
        n_items = random.randint(3, min(len(items), 8))
    chosen = random.sample(items, min(n_items, len(items)))
    random.shuffle(chosen)
    lines = []
    for i, item in enumerate(chosen, 1):
        lines.append(f"[{i}] {item}")
    return "\n".join(lines)


def estimate_tokens(text):
    # Rough estimate: ~4 chars per token
    return len(text) // 4


def generate_llm_sample():
    domain_name = random.choice(list(DOMAINS.keys()))
    domain = DOMAINS[domain_name]

    # Pick a task type that has questions defined for this domain
    available_types = [
        t for t in domain["questions"]
        if domain["questions"][t]
    ]
    task_type = random.choice(available_types)
    qa_pair = random.choice(domain["questions"][task_type])
    question, answer_core = qa_pair

    # Build context
    context_block = build_context_block(domain_name)
    context_tokens = estimate_tokens(context_block)
    # Round to nearest bucket
    bucket = min(CONTEXT_TOKEN_BUCKETS, key=lambda b: abs(b - context_tokens))

    instruction = random.choice(INSTRUCTION_VARIANTS)
    model = random.choice(MODELS)

    input_text = f"<context>\n{context_block}\n</context>\n\nQuestion: {question}"

    # Build output with a framing prefix
    prefixes = [
        "Based on the context provided, ",
        "According to the memory context, ",
        "The context indicates that ",
        "From the available memory items, ",
        "Based on the retrieved context, ",
        "The recorded information shows that ",
    ]
    output = random.choice(prefixes) + answer_core

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "metadata": {
            "task_type": task_type,
            "domain": domain_name,
            "context_tokens": bucket,
            "model": model,
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
        "train": 25_000,
        "val":    2_500,
        "test":   2_500,
    }

    total = sum(splits.values())
    print(f"Generating {total:,} total LLM fine-tuning samples...")

    for split_name, count in splits.items():
        print(f"  Generating {count:,} {split_name} samples...")
        samples = [generate_llm_sample() for _ in range(count)]
        out_path = os.path.join(base_dir, f"llm_finetune_{split_name}.jsonl")
        write_jsonl(out_path, samples)

    print("Done. LLM fine-tuning dataset generation complete.")


if __name__ == "__main__":
    main()
