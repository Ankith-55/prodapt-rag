# Architecture

Diagrams are Mermaid (rendered natively by GitHub and VS Code). Numbers come from the evals in `docs/ablations.md`.

## 1. System overview

```mermaid
flowchart LR
  subgraph Clients
    UI["Browser UI<br/>HTML / JS / CSS<br/>served by the same container"]
    DESK["Desk tooling / curl"]
  end

  subgraph SVC["Service: one stateless FastAPI container"]
    API["POST /ask<br/>POST /ingest<br/>POST /reload<br/>GET /health /ready /metrics"]

    subgraph PIPE["Assistant pipeline"]
      R["1 Retrieve<br/>bge-small embeddings + FAISS<br/>top-5 ticket patterns"]
      G{"2 Similarity gate<br/>top cosine at least 0.74?"}
      P["3 Parse - LLM<br/>pick a candidate or none<br/>severity, sentiment"]
      T{"4 Route by tier"}
      TPL["Template answer<br/>no LLM call"]
      GEN["5 Generate - LLM<br/>opening, summary, steps<br/>with source citations"]
      V["6 Validate<br/>known citations, numbers from sources,<br/>no predictions, retry once"]
      OUT["7 Assemble<br/>outcomes rendered from the table,<br/>policy notes, timings"]
      ABS["Abstain<br/>route to a human,<br/>show nearest types"]
    end
  end

  subgraph STATE["State directory (volume)"]
    TK[("tickets.parquet")]
    PT[("patterns.parquet<br/>resolutions.parquet")]
    IX[("index/ FAISS vectors + meta")]
    EX[("examples.jsonl<br/>ingest_log.jsonl")]
  end

  LLMC[("LLM response cache")]
  OAI["OpenAI API<br/>gpt-4o-mini"]
  PROM["Prometheus<br/>scrapes /metrics"]

  UI --> API
  DESK --> API
  API --> R --> G
  G -- "no" --> ABS
  G -- "yes" --> P
  P -- "none fits" --> ABS
  P --> T
  T -- "fast_lookup" --> TPL --> OUT
  T -- "rag_light / rag_core" --> GEN --> V --> OUT
  R -.-> IX
  R -.-> PT
  P -.-> OAI
  GEN -.-> OAI
  P -.-> LLMC
  GEN -.-> LLMC
  PROM -.-> API
```

If the LLM path fails, `ask_safe` skips steps 3 to 6 and returns a statistics-only answer rendered from the retrieved table (`method = template_degraded`).

## 2. One request, in order

```mermaid
sequenceDiagram
  autonumber
  participant A as Agent (browser)
  participant S as FastAPI /ask
  participant E as Retriever
  participant L as LLM (parse, generate)
  participant V as Validator

  A->>S: raw complaint text
  S->>E: embed + search (label and example vectors, max per pattern)
  E-->>S: top-5 patterns with cosine scores and resolution distributions
  alt top cosine below 0.74
    S-->>A: abstained, nearest types, no LLM cost
  else
    S->>L: parse: pick candidate 1 to 5 or none, severity, sentiment
    L-->>S: choice
    alt tier is fast_lookup
      S-->>A: template answer, no generation call
    else
      S->>L: generate from numbered sources P# and R#
      L-->>S: opening, summary, steps with citations
      S->>V: check citations, numbers, no predictions
      V-->>S: violations, if any, as feedback
      opt violations
        S->>L: retry once with the feedback
      end
      S-->>A: answer + outcomes from the table + sources + notes
    end
  end
```

## 3. Evolving data: how new tickets and new classes get in

```mermaid
flowchart TD
  A["New closed tickets<br/>POST /ingest or scripts/ingest.py"] --> B["Clean<br/>closed only, real resolution,<br/>repair double-encoded text"]
  B --> C["Dedupe by unique_key<br/>re-ingesting is a no-op"]
  C --> D["Append to tickets.parquet"]
  D --> E["Recompute pattern table<br/>counts, resolution shares,<br/>median / p90 close time, tiers"]
  E --> F{"New pattern<br/>never seen before?"}
  F -- "yes" --> G["LLM writes 4 example complaints"]
  G --> H["Embed label + examples<br/>upsert vectors, no rebuild"]
  F -- "no" --> I["Keep vectors,<br/>refresh statistics only"]
  H --> J["Write state + audit log"]
  I --> J
  J --> K["retriever.reload()<br/>atomic swap while serving"]
```

## 4. Data model in one picture

```mermaid
flowchart LR
  T["Ticket<br/>complaint_type, descriptor,<br/>descriptor_2, resolution text,<br/>close time"] -->|"group by the 3 labels"| P["Pattern<br/>tier, total cases,<br/>median / p90 close hours"]
  T -->|"dedupe by text"| R["Resolution template<br/>about 434 distinct texts"]
  P -->|"resolution_dist<br/>share per template"| R
  P -->|"1 label vector +<br/>4 example-complaint vectors"| V["FAISS index<br/>about 6,330 vectors<br/>id = pattern id, slot"]
```

Tier rules (computed on every ingest): `fast_lookup` if the top resolution has at least 90% share and at least 30 tickets; `rag_light` if the top share is at least 50%; otherwise `rag_core`.

## 5. Deployment shape (target)

```mermaid
flowchart LR
  U["Users"] --> ING["Ingress / load balancer"]
  ING --> SVCK["Service"]
  SVCK --> POD1["Pod: ticketrag<br/>readiness: /ready<br/>liveness: /health"]
  SVCK --> POD2["Pod: ticketrag"]
  HPA["Horizontal Pod Autoscaler<br/>on CPU and request rate"] -.-> POD1
  HPA -.-> POD2
  POD1 --- VOL[("Shared state volume<br/>or object storage + index version")]
  POD2 --- VOL
  POD1 --> OAI2["OpenAI API"]
  PROM2["Prometheus + alerts"] -.-> POD1
  PROM2 -.-> POD2
  CI["GitHub Actions<br/>lint, tests, retrieval gate,<br/>image build + push"] --> REG["Container registry"] --> POD1
```

With a flat in-process index every replica loads its own copy, so ingestion must refresh all replicas (or move to a shared vector database). See the scale notes in `docs/how_it_works.md`.
