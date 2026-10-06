# Architecture

Diagrams are Mermaid (rendered natively by GitHub and VS Code). Numbers come from the evals in `docs/ablations.md`.

## 0. The README figure (editable source)

The README shows this diagram as `docs/img/architecture.png`, exported from <https://mermaid.live> so it looks identical everywhere. To change it, edit the code below, re-export a PNG (Actions, PNG, larger scale), and replace the file.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'Arial, Helvetica, sans-serif', 'fontSize': '20px', 'lineColor': '#0b2440'}, 'flowchart': {'curve': 'basis', 'nodeSpacing': 55, 'rankSpacing': 75, 'padding': 24, 'htmlLabels': true}}}%%
flowchart LR
  subgraph IN["&nbsp;&nbsp;INPUT&nbsp;&nbsp;"]
    Q["<b>Raw customer<br/>complaint</b>"]
  end

  subgraph RET["&nbsp;&nbsp;RETRIEVE&nbsp;&nbsp;"]
    R["<b>1 · Retrieve</b><br/>bge-small + FAISS<br/>top-5 ticket types"]
    G{{"<b>2 · Similarity gate</b><br/>cosine ≥ 0.74"}}
  end

  subgraph UND["&nbsp;&nbsp;UNDERSTAND&nbsp;&nbsp;"]
    P["<b>3 · Parse (LLM)</b><br/>pick a type or none<br/>severity · sentiment"]
    T{{"<b>4 · Route</b><br/>by ticket history"}}
  end

  subgraph ANS["&nbsp;&nbsp;ANSWER&nbsp;&nbsp;"]
    TPL["<b>Template answer</b><br/>no LLM call"]
    GEN["<b>5 · Generate (LLM)</b><br/>numbered sources<br/>+ citations"]
    V["<b>6 · Validate</b><br/>citations · numbers<br/>no predictions · retry once"]
    OUT["<b>7 · Final answer</b><br/>outcome shares from the table<br/>+ cited sources"]
  end

  ABS["<b>ABSTAIN</b><br/>route to a human<br/>show nearest types"]
  DATA[("<b>Ticket history</b><br/>patterns · resolutions<br/>FAISS index")]
  ING["<b>Ingest new tickets</b><br/>new classes searchable<br/>without a rebuild"]

  Q --> R --> G
  G -- "no" --> ABS
  G -- "yes" --> P
  P -- "none fits" --> ABS
  P --> T
  T -- "consistent outcome" --> TPL --> OUT
  T -- "mixed outcomes" --> GEN --> V --> OUT
  DATA -.-> R
  ING -.-> DATA

  classDef input fill:#f4f2ec,stroke:#1b2330,stroke-width:3px,color:#1b2330;
  classDef retrieval fill:#dbe7f5,stroke:#0b2440,stroke-width:3px,color:#0b2440;
  classDef llm fill:#e6dcf5,stroke:#2e1a5c,stroke-width:3px,color:#2e1a5c;
  classDef guard fill:#fbe9b0,stroke:#5a4300,stroke-width:3px,color:#3d2e00;
  classDef output fill:#cfe9d3,stroke:#1b4d26,stroke-width:3px,color:#12331a;
  classDef store fill:#dfe3e9,stroke:#2b3340,stroke-width:3px,color:#1b2330;
  classDef stop fill:#f8d7d2,stroke:#7a1f12,stroke-width:3px,color:#5a140a;

  class Q input;
  class R retrieval;
  class G,T guard;
  class P,GEN llm;
  class V guard;
  class TPL,OUT output;
  class DATA,ING store;
  class ABS stop;

  style IN fill:#ffffff,stroke:#1b2330,stroke-width:4px,color:#1b2330
  style RET fill:#f3f7fc,stroke:#0b2440,stroke-width:4px,color:#0b2440
  style UND fill:#f8f4fd,stroke:#2e1a5c,stroke-width:4px,color:#2e1a5c
  style ANS fill:#f2faf3,stroke:#1b4d26,stroke-width:4px,color:#12331a

  linkStyle default stroke:#0b2440,stroke-width:3px
```

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
