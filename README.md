# faq-support-chatbot

A small RAG-style FAQ support tool that builds a local embedding index from a plain-text FAQ, performs nearest-neighbor retrieval, queries an LLM for an answer, and records simple request metrics (tokens, latency, estimated cost).

---

## Features
- Build a local vector index (Chromadb) from a plain text FAQ.
- Query the index and produce a JSON response containing:
  - user_question
  - system_answer
  - chunks_related (retrieved passages + metadata + relevance)
- Record per-request metrics (input/output tokens, latency, estimated cost) and persist them to CSV.
- Simple CLI scripts for index build and query.

---

## Requirements
- Python 3.10+
- Windows (PowerShell examples provided)
- Recommended: virtual environment
- Key Python packages: chromadb, sentence-transformers, torch (or CPU-only alternatives), pytest (for tests)
- See `requirements.txt` for full dependency list.

---

## Setup

1. Clone / open project:
   - Project root: `faq-support-chatbot`

2. Create and activate venv (PowerShell):
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1

3. Install dependencies:
   - pip install -r requirements.txt

4. Create `.env` file in project root and set your API key:
   - OPENROUTER_API_KEY=<your_openrouter_api_key_here>

## Steps to get OpenRouter API key:

1. Login to OpenRouter
2. Go to credits section (https://openrouter.ai/settings/credits) and add credits
3. Go to API keys section (https://openrouter.ai/settings/keys) and click on create API key
4. Copy that API key and save it in a .env file as specified in Setup 4th step

---

# Project Structure:
- faq-support-chatbot
  - chroma_store
  - data
    - hr_saas.txt
  - outputs
    - expected_answers.json
    - metrics.csv
    - sample_queries.json
  - src
    - build_index.py
    - chroma_client.py
    - evaluator_agent.py
    - llm_client.py
    - query.py
    - utils.py
  - tests
    - test_utils.py
  - .env.example
  - .gitignore
  - README.md
  - requirements.txt


# Metrics: what is recorded & how to reproduce

Per-request metrics captured:
- timestamp
- latency_ms
- tokens_prompt (input tokens)
- tokens_completion (output tokens)
- tokens_total
- estimated_cost_usd
- Where metrics are saved:

The scripts print metrics to stdout and saved to outputs/metrics.csv

Example reproduction:
Build index (step above).
Run query.py with a question.
Check the console output for token/latency logs and the metrics.csv file (create metrics dir if the script expects it).

---

# How to Run:
1. Create virtual environment and install requirements

```powershell
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Build the index from your text file

```powershell
(.venv_m2) PS C:\work\projects\faq-support-chatbot> python src\build_index.py --input data\HR_SaaS_FAQ.txt --persist_dir .\chroma_store --collection_name hr_faq
```
This will:
- Split into chunks (default ~1000 chars per chunk with 200 chars overlap)
- Create embeddings using all-mpnet-base-v2
- Upsert into Chromadb persisted at ./chroma_store

## Sample Response:
```
(.venv_m2) PS C:\work\projects\faq-support-chatbot> python src\build_index.py --input data\hr_saas.txt --persist_dir .\chroma_store --collection_name hr_faq    
2025-11-16 20:41:05,684 INFO Split input into 116 sentences/paragraphs
2025-11-16 20:41:05,685 INFO Created 20 chunks (approx_chars=800 overlap=200)
2025-11-16 20:41:05,690 INFO Use pytorch device_name: cpu
2025-11-16 20:41:05,690 INFO Load pretrained SentenceTransformer: all-mpnet-base-v2
Embedding batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.57s/it]
2025-11-16 20:41:19,810 INFO Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
Upserting %d documents into Chromadb collection '%s' 20 hr_faq
2025-11-16 20:41:20,239 INFO Index build complete. Persist dir: .\chroma_store
{
  "input_path": "C:\\work\\projects\\faq-support-chatbot\\data\\hr_saas.txt",
  "collection_name": "hr_faq",
  "persist_dir": "C:\\work\\projects\\faq-support-chatbot\\chroma_store",
  "n_chunks": 20
}
```

3. Query the Index:

Example Run 1:

```powershell
(.venv_m2) PS C:\work\projects\faq-support-chatbot> python src\query.py --question "What are the key steps involved in the employee onboarding process in this HR SaaS system?" --persist_dir .\chroma_store --collection_name hr_faq --k 2
Input Tokens: 474
Output Tokens: 97
Request Cost(USD): $0.000129
Request metrics: {'timestamp': '2025-11-16 15:19:49', 'latency_ms': '9741.746800', 'tokens_prompt': 474, 'tokens_completion': 97, 'tokens_total': 571, 'estimated_cost_usd': '0.000129'}
Saving metrics to CSV...
----- Final Output -----

{
  "user_question": "What are the key steps involved in the employee onboarding process in this HR SaaS system?",
  "system_answer": "The key steps involved in the employee onboarding process in this HR SaaS system include:\n\n1. Creation of the employee record.\n2. Document submission (e.g., ID proof, tax forms).\n3. Assigning equipment/assets.\n4. Setting up access.\n5. Orientation scheduling.\n6. Learning modules assignment.\n7. Setting up manager and peer introductions.\n\nThe employee uses the self-service portal to complete forms, upload documents, and track their onboarding progress [chunk_0006].",
  "chunks_related": [
    {
      "doc_id": "chunk_0000",
      "document": "HR SaaS Platform — Frequently Asked Questions 1. General Overview Q: What is the platform and what does it do? Our HR SaaS platform is designed to support the full lifecycle of employee management — from recruitment, onboarding, attendance and time tracking, performance reviews, compensation and benefits, learning & development, offboarding, and analytics. It is cloud‑based, accessible from anywhere, and integrates with other business systems. Q: Who uses the platform? The platform is used by HR professionals, team managers, payroll teams, learning & development staff, and employees themselves via self‑service portals. Q: What are the advantages compared to traditional on‑premise HR systems? Unlike traditional on‑premise systems, our SaaS platform avoids large upfront hardware/infrastructure costs, offers automatic updates, better scalability, faster deployment, remote access, and easier integrations with other cloud or on‑premise systems. 2. Key Features",
      "metadata": {
        "chunk_id": "chunk_0000",
        "chunk_index": 0,
        "source": "HR_SaaS_FAQ.txt"
      },
      "relevance_score": 0.5765063166618347
    },
    {
      "doc_id": "chunk_0006",
      "document": "Q: How does the recruitment module work? The recruitment or applicant-tracking (ATS) module enables you to post job requisitions, track applicants, schedule interviews, evaluate candidates, send offer letters, and onboard new hires. Workflow automation ensures consistent processing of candidates through stages (screening, interview, offer, hire). Q: What happens when a new employee joins? Once a new hire is confirmed, the system triggers onboarding workflows: creation of employee record, document submission (e.g., ID proof, tax forms), assigning equipment/asset, setting up access, orientation scheduling, learning modules assignment, setting up manager and peer introductions. The employee uses the self-service portal to complete forms, upload documents, and track their onboarding progress. Q: Can we customise onboarding for different roles or business units? Yes — the platform supports configurable workflows and templates.",
      "metadata": {
        "chunk_index": 6,
        "source": "HR_SaaS_FAQ.txt",
        "chunk_id": "chunk_0006"
      },
      "relevance_score": 0.7292017340660095
    }
  ]
}
```

Example run 2 (with evaluator_agent output):

```
(.venv_m2) PS C:\work\projects\faq-support-chatbot> python src\query.py --question "What are security and compliance features to protect employee data?" --persist_dir .\chroma_store --collection_name hr_faq --k 2
Input Tokens: 455
Output Tokens: 71
Request Cost(USD): $0.000111
Request metrics: {'timestamp': '2025-11-16 21:13:47', 'latency_ms': '12796.063100', 'tokens_prompt': 455, 'tokens_completion': 71, 'tokens_total': 526, 'estimated_cost_usd': '0.000111'}
Saving metrics to CSV...
----- Final Output -----

{
  "user_question": "What are security and compliance features to protect employee data?",
  "system_answer": "The security and compliance features to protect employee data include data encryption in transit and at rest, role-based user access, multi-factor authentication, and the maintenance of audit logs. The platform also complies with relevant data-protection and labor-law regulations, supports data-retention policies, and ensures sensitive information is encrypted [chunk_0003][chunk_0014].",
  "chunks_related": [
    {
      "doc_id": "chunk_0003",
      "document": "Data is encrypted in transit and at rest, user access is role‑based, multi‑factor authentication is supported, audit logs are maintained, and the platform complies with relevant data‑protection and labour‑law regulations. 3. Policies & Procedures Q: What HR policies can I enforce or manage through the platform? You can manage and enforce policies such as leave/holiday policy, flexible/hybrid‑work policy, business‑travel policy, expenses reimbursement policy, dress code policy, performance‑management policy, data‑security/BYOD (bring your own device) policy, code‑of‑conduct, anti‑harassment, remote‑work policy, timesheet/attendance policy. Q: What is the difference between an HR policy and procedure? A policy is a high‑level guideline that expresses the organisation’s intent. A procedure is the step‑by‑step process for how that policy is implemented. Q: Why are HR policies and procedures important?",
      "metadata": {
        "source": "HR_SaaS_FAQ.txt",
        "chunk_index": 3,
        "chunk_id": "chunk_0003"
      },
      "relevance_score": 0.6951243877410889
    },
    {
      "doc_id": "chunk_0014",
      "document": "The platform allows you to define policies (e.g., anti-harassment, equal-opportunity, child-labour prohibition, data privacy), enforce workflows (reporting incidents, investigations), track training completions, maintain audit trails, schedule policy reviews, and generate compliance reports. It helps you maintain consistent treatment of employees and defend decisions with recorded evidence. Q: What about data privacy and access controls? Role-based access ensures only authorised users see specific modules or data. Sensitive information is encrypted. Multi-factor authentication is supported. Regular audits and logging ensure data integrity. The system also supports data-retention policies, archiving of former employees’ data, and deletion or anonymisation of records as required. Q: Are there dashboards and analytics for HR and business leaders?",
      "metadata": {
        "source": "HR_SaaS_FAQ.txt",
        "chunk_id": "chunk_0014",
        "chunk_index": 14
      },
      "relevance_score": 0.7162110805511475
    }
  ]
}

----- Evaluation -----

{
  "final_score": 8.65,
  "components": {
    "support_score": 4.0,
    "citation_score": 2.0,
    "completeness_score": 1.65,
    "clarity_score": 1.0
  },
  "metadata": {
    "sentences": 2,
    "supported_sentences": 2,
    "cited_ids": [
      "chunk_0014",
      "chunk_0003"
    ],
    "valid_cited_ids": 2,
    "available_chunk_ids": [
      "chunk_0014",
      "chunk_0003"
    ],
    "top_keywords": [
      "policy",
      "data",
      "policies",
      "access",
      "platform",
      "what",
      "enforce",
      "encrypted",
      "role",
      "based",
      "multi",
      "factor",
      "authentication",
      "supported",
      "audit",
      "labour",
      "procedures",
      "manage",
      "work",
      "business"
    ]
  },
  "explanation": "Support: 4.00/4 — 100.0% of sentences supported. Citations: 2.00/2 — found ['chunk_0014', 'chunk_0003'], valid: 2. Completeness: 1.65/3 — keyword coverage: 55.0%. Clarity: 1.00/1 — avg sentence length: 27.0 tokens."
}
```

JSON output will contain:
- user_question (the user input)
- system_answer (LLM-generated answer)
- chunks_related (list of retrieved chunks, metadata, distances).

---

# Run tests:
```powershell
(.venv_m2) PS C:\work\projects\faq-support-chatbot> python -m pytest -q .\tests\test_utils.py::test_sentence_split --verbose
================================================== test session starts ===================================================
platform win32 -- Python 3.13.1, pytest-9.0.1, pluggy-1.6.0
rootdir: C:\work\projects\faq-support-chatbot
plugins: anyio-4.11.0
collected 1 item                                                                                                          

tests\test_utils.py .                                                                                               [100%] 

=================================================== 1 passed in 14.40s =================================================== 
```

---

# Development notes / troubleshooting

- Ensure you run commands from the project root so relative paths resolve correctly.
- If tests fail due to imports, verify src is on Python path when running pytest from the project root.
- To inspect raw LLM outputs for debugging, enable debug logging in the scripts or check printed output before JSON parsing.
- To change chunking behavior, adjust --approx_chars and --overlap_chars in build_index.py.

## Chunking strategy:
chunk_size = 800 characters
chunk_overlap = 200 characters
Rough result: ≈ 20-24 chunks — works well for precise retrieval and low hallucination

Overlap prevents a single concept/sentence from being split across two chunks, improving retrieval relevance and context for the LLM.

- for small chunks (e.g., 300–400 chars), we should use a larger overlap (100–150 chars) to preserve continuity.
- for large chunks (1000+ chars), we shouldvreduce overlap (100–150 chars) because the chunk already contains more context.
So, we have used a balanced chunk_size of 800 chars and chunk_overlap of 200 chars


# Known limitations

- No production authentication or rate limiting
- Token and cost estimates are as per the llm model used
- The query step depends on an available LLM (API key) unless you wire a local model. If API key is missing, the query script will fail (scripts may raise exceptions at import-time if the client is eagerly initialized).
- Chromadb usage and embedding generation may be CPU-bound
- No integrated monitoring dashboard — metrics are CSV/log-based for manual inspection.
