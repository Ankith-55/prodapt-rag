# Prodapt RAG — Dataset & EDA Summary

## 1. Project Goal

The goal of this project is to build a **Retrieval-Augmented Generation (RAG) system for customer complaint resolution**.

At inference time, the system is expected to receive a **raw natural-language customer complaint**, for example:

> "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."

The system should retrieve historically similar cases and use their recorded resolutions as grounded evidence for producing a useful response.

The important design principle is that the user/tester **does not provide structured fields** such as `complaint_type`, `descriptor`, or `descriptor_2`. Those fields exist in the historical dataset and are used to represent the historical cases, while the actual query is a raw customer complaint.

---

## 2. Why We Need a Historical Dataset

A RAG system needs a collection of documents/cases that can be retrieved when a new query arrives.

For this project, each historical customer-service ticket acts as a **retrieval unit**:

```text
Historical complaint information
        +
Historical resolution
        ↓
Retrievable case
```

When a new raw complaint is received, the system can find semantically similar historical cases and use their resolutions as evidence.

This is different from a simple classification system.

A classification approach would try to learn:

```text
Complaint → Fixed class
```

Our RAG approach instead retrieves:

```text
New complaint
    ↓
Similar historical cases
    ↓
Historical resolutions
    ↓
Grounded response
```

This is particularly useful because the same type of complaint does not always have a single historical resolution.

---

## 3. Dataset Source

The dataset was collected from the official NYC 311 Service Requests API:

`https://data.cityofnewyork.us/resource/erm2-nwe9.json`

The extraction used:

- Maximum records: **100,000**
- Start date: **2026-03-01**
- Records ordered by `created_date`

The extracted dataset contains:

- **100,000 records**
- **44 columns**
- Date range: **2026-03-01 to 2026-03-09**

The dataset represents historical service requests/tickets and contains both complaint-side information and resolution information.

---

## 4. Important Dataset Fields

The original dataset contains 44 fields. For the RAG dataset, we focus on a small subset that is relevant to historical complaint retrieval and resolution analysis.

### Final RAG dataset fields

| Column | Purpose |
|---|---|
| `unique_key` | Unique identifier for the historical ticket |
| `created_date` | Time at which the ticket was created |
| `closed_date` | Time at which the ticket was closed |
| `complaint_type` | High-level category of the complaint |
| `descriptor` | More specific description of the complaint |
| `descriptor_2` | Additional complaint-specific detail when available |
| `resolution_description` | Recorded historical resolution/action |

Only **closed tickets** are retained for the final historical RAG dataset because a closed ticket provides an actual historical resolution.

The final dataset is saved as:

```text
closed_tickets_rag.csv
```

---

## 5. Why Only Closed Tickets?

The RAG system is intended to retrieve **historical evidence about how similar complaints were resolved**.

An open or in-progress ticket does not necessarily contain a final resolution.

Therefore, the historical corpus is filtered using:

```python
df[df["status"] == "Closed"]
```

and cases without a meaningful `resolution_description` are excluded from the usable historical analysis.

This makes the retrieval corpus more appropriate for the task:

```text
Past complaint → observed resolution
```

rather than:

```text
Past complaint → unresolved/current status
```

---

## 6. Initial Dataset Analysis

The original dataset contains a wide range of complaint categories and descriptors.

The analysis found:

- **161 unique complaint types**
- **677 unique descriptors** in the analyzed dataset
- Many different resolution descriptions
- Significant variation in the resolution associated with similar complaint patterns

This indicates that the dataset is not simply a collection of one-to-one mappings between complaint categories and resolutions.

---

# 7. RAG Input Representation

The main historical input representation investigated was:

```text
complaint_type
+
descriptor
+
descriptor_2
```

`descriptor_2` is optional because it is not available for every historical ticket.

The corresponding historical outcome is:

```text
resolution_description
```

Conceptually:

```text
INPUT / COMPLAINT CONTEXT
    complaint_type
    descriptor
    descriptor_2 (optional)
            ↓
HISTORICAL OUTCOME
    resolution_description
```

### Important inference-time distinction

The historical dataset contains structured fields, but the actual RAG system will eventually receive a **raw customer complaint string**.

For example:

```text
"My broadband drops every evening around 8 and I've already
restarted the router twice. I work from home and this is costing me."
```

The system should embed this raw text and retrieve semantically similar historical cases.

It should **not require the customer/agent to manually provide**:

```text
complaint_type = ...
descriptor = ...
descriptor_2 = ...
```

---

# 8. Why `descriptor_2` Was Retained

`descriptor_2` was tested to determine whether it adds useful specificity to the historical cases.

Using:

```text
complaint_type
+ descriptor
+ resolution_description
```

there were approximately:

**3,088 unique representations**

After adding `descriptor_2`:

```text
complaint_type
+ descriptor
+ descriptor_2
+ resolution_description
```

there were approximately:

**4,330 unique representations**

This represents approximately a:

**40.22% increase in unique input → resolution representations.**

Therefore, `descriptor_2` provides additional information and is retained when available.

It is treated as **optional**, rather than requiring it for every case.

---

# 9. Descriptor 2 Coverage

`descriptor_2` is not universally populated.

For the usable historical records:

- Records with `descriptor_2`: **47,368**
- Records missing `descriptor_2`: **51,762**
- Coverage: **47.78%**

This means the dataset should **not** be restricted only to rows containing `descriptor_2`.

Instead:

```text
If descriptor_2 exists:
    use it

If descriptor_2 is missing:
    use complaint_type + descriptor
```

This preserves the majority of the historical dataset while still taking advantage of the additional signal when it exists.

---

# 10. Input Diversity Analysis

When `resolution_description` is excluded from the input representation, the combination:

```text
complaint_type
+ descriptor
+ descriptor_2
```

produced:

**1,306 unique input representations**

This is important because it shows that the retrieval input is not simply one repeated phrase.

However, many historical tickets naturally share the same complaint pattern.

That is not necessarily a problem for RAG.

Repeated historical cases provide multiple examples of how similar complaints were handled and therefore provide useful retrieval evidence.

---

# 11. Resolution Diversity

For the final input representation:

```text
complaint_type
+ descriptor
+ descriptor_2
```

the analysis produced:

**1,143 input patterns**

Their resolution behavior was:

| Number of unique resolutions | Number of input patterns |
|---:|---:|
| 1 | 536 |
| 2 | 173 |
| 3 | 99 |
| 4 | 88 |
| 5 | 53 |
| 6 | 38 |
| 7 | 35 |
| 8 | 50 |
| 9 | 20 |
| 10 | 18 |
| 11 | 8 |
| 12 | 5 |
| 13 | 5 |
| 14 | 6 |
| 15 | 3 |
| 16 | 4 |
| 18 | 1 |
| 22 | 1 |

Therefore:

- **607 / 1,143 (53.11%)** input patterns have multiple historical resolutions.
- **434 / 1,143 (37.97%)** have three or more historical resolutions.
- At least one input pattern has **22 different historical resolutions**.

This is an important reason for using retrieval rather than assuming a deterministic lookup.

---

# 12. Top-1 Resolution Analysis

For every input pattern, the most common historical resolution was measured using `top1_share`.

The distribution was:

| Top-1 resolution share | Input patterns |
|---|---:|
| ≤25% | 31 |
| 25–50% | 338 |
| 50–75% | 152 |
| 75–90% | 60 |
| 90–100% | 562 |

This demonstrates that the dataset contains a mixture of:

- Highly predictable complaint patterns
- Moderately ambiguous complaint patterns
- Highly diverse complaint patterns

This is useful for RAG evaluation because retrieval can be tested across different levels of historical ambiguity.

---

# 13. Examples of Resolution Ambiguity

Some input patterns have relatively many different historical resolutions.

Examples observed during EDA include patterns such as:

```text
Street Condition
+ Cave-in
+ N/A
→ 22 different resolutions
```

and:

```text
Street Condition
+ Rough, Pitted or Cracked Roads
+ N/A
→ 18 different resolutions
```

Other complaint patterns had many fewer possible outcomes.

This shows that the historical data contains genuine variation in how cases were resolved.

---

# 14. Why This Is a RAG Problem

The dataset supports a retrieval-based formulation because:

### 1. Historical cases contain useful evidence

Each closed ticket contains:

```text
Complaint context
+
Observed resolution
```

That gives the RAG system a corpus of real historical examples.

### 2. Similar complaints can have multiple resolutions

More than half of the input patterns have multiple historical resolutions.

Therefore, retrieving several similar cases can provide more information than simply mapping an input to one fixed label.

### 3. The query is naturally expressed as free text

The actual system receives:

```text
Raw customer complaint
```

rather than a structured database key.

This makes semantic retrieval important.

### 4. The response can be grounded in retrieved evidence

Instead of asking an LLM to invent a resolution from its pretrained knowledge, the system can provide historical cases as context:

```text
Raw complaint
      ↓
Embedding
      ↓
Vector search
      ↓
Top-K historical cases
      ↓
Historical resolutions
      ↓
LLM-generated grounded response
```

---

# 15. Proposed RAG Architecture

The planned system is:

```text
                 ┌─────────────────────────┐
                 │ Raw Customer Complaint  │
                 └────────────┬────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Embedding Model │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vector Database │
                    │  / FAISS        │
                    └────────┬────────┘
                             │
                         Top-K cases
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Historical 311 Cases       │
              │                             │
              │ Complaint information       │
              │ Resolution description      │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ LLM / Agent     │
                    └────────┬────────┘
                             │
                             ▼
                    Grounded response
```

---

# 16. Important Leakage Consideration

`resolution_description` should **not** be included in the query representation.

The query is the unresolved customer complaint.

Therefore:

```text
Query embedding:
    complaint information only
```

while:

```text
Retrieved historical case:
    complaint information
    +
    resolution_description
```

This avoids outcome leakage.

The system is therefore testing whether:

> **A raw complaint can retrieve historical cases whose recorded resolutions are useful for answering that complaint.**

---

# 17. Resolution Clustering

Initially, the project should **not require pre-clustering all resolution descriptions**.

A simpler first implementation is:

```text
Raw complaint
    ↓
Retrieve Top-K historical cases
    ↓
Collect their resolution descriptions
    ↓
Provide them to the LLM
```

For example:

```text
Case 1 → Resolution A
Case 2 → Resolution A
Case 3 → Resolution B
Case 4 → Resolution A
Case 5 → Resolution C
```

The LLM can then summarize the historical evidence.

Semantic clustering of resolutions can be added later if the project requires grouping similar resolution descriptions.

---

# 18. Why This Dataset Is Suitable for the Project

The EDA supports the dataset as a reasonable RAG corpus because it has:

- A large historical case volume
- Structured complaint information
- Optional fine-grained complaint information through `descriptor_2`
- Recorded outcomes through `resolution_description`
- Significant resolution diversity
- Repeated historical examples for similar complaints
- A natural mapping from historical cases to retrieval evidence
- A clear distinction between query information and outcome information

The dataset therefore supports an experimental question such as:

> **Can semantic retrieval of historical customer-service cases provide useful evidence for resolving a new raw customer complaint?**

---

# 19. Current Dataset Pipeline

The current workflow is:

```text
NYC 311 API
    ↓
100,000 service requests
    ↓
Filter to closed tickets
    ↓
Keep tickets with meaningful historical resolutions
    ↓
Select RAG-relevant columns
    ↓
closed_tickets_rag.csv
    ↓
Build historical retrieval corpus
    ↓
Generate embeddings
    ↓
Build vector index
    ↓
Test with raw customer complaints
    ↓
Retrieve Top-K historical cases
    ↓
Evaluate retrieved resolutions
```

---

# 20. Final RAG Dataset

The reduced dataset saved for the project contains only:

```text
unique_key
created_date
closed_date
complaint_type
descriptor
descriptor_2
resolution_description
```

The final corpus contains **closed tickets only**.

`unique_key`, `created_date`, and `closed_date` are retained for traceability and analysis, while the primary retrieval information is:

```text
complaint_type
descriptor
descriptor_2
```

and the primary historical evidence is:

```text
resolution_description
```

---

# 21. Next Step

The EDA phase is now considered complete.

The next phase is to implement the actual RAG pipeline:

1. Prepare historical case documents.
2. Separate retrieval/query text from resolution information.
3. Generate embeddings using a GPU-backed embedding model.
4. Build a FAISS/vector index.
5. Accept a raw customer complaint as the query.
6. Retrieve the most similar historical cases.
7. Inspect the retrieved resolution descriptions.
8. Add an LLM/agent to synthesize a grounded response.
9. Evaluate retrieval quality and resolution usefulness.

The key experiment is therefore no longer:

```text
Does the dataset contain enough variation?
```

That has been investigated.

The next question is:

```text
Can semantic retrieval successfully connect a
raw natural-language customer complaint
to relevant historical cases and resolutions?
```

That is the core RAG experiment.
