# NYC 311 Service Request RAG — Dataset Analysis & Selection

## 1. Project Overview

The objective of this project is to explore NYC 311 service-request data and investigate whether it can support a **Retrieval-Augmented Generation (RAG)** system.

The intended system is to retrieve historically similar service requests and use their information, particularly historical resolutions, to provide context for resolving or explaining new service requests.

Before building the RAG pipeline, exploratory data analysis (EDA) was performed on an initial dataset to determine whether the available information was sufficiently rich for this purpose.

---

# 2. Initial Dataset

The initial dataset was provided as:

```text
tickets.csv
```

It contained:

- **25,921 rows**
- **9 columns**

### Columns

| Column | Description |
|---|---|
| `ticket_id` | Unique service request identifier |
| `created_date` | Date and time when the request was created |
| `closed_date` | Date and time when the request was closed |
| `agency` | Responding agency |
| `complaint_type` | Broad category of the complaint |
| `descriptor` | More specific description of the complaint |
| `borough` | NYC borough |
| `status` | Current status of the request |
| `resolution_description` | Description of the action taken to resolve the request |

---

# 3. Initial Dataset EDA

## 3.1 Dataset Distribution

The dataset contains 7 complaint types:

| Complaint Type | Number of Requests |
|---|---:|
| Noise - Residential | 13,384 |
| Illegal Parking | 3,759 |
| Street Condition | 2,136 |
| Rodent | 2,040 |
| Heat/Hot Water | 1,992 |
| Sanitation Condition | 1,422 |
| Water System | 1,188 |

The dataset is therefore highly concentrated in a small number of complaint categories, with **Noise - Residential accounting for more than half of all records**.

---

# 4. Descriptor Analysis

The relationship between `complaint_type` and `descriptor` was investigated.

The dataset contains a relatively small number of descriptors. Examples include:

- Banging/Pounding
- Blocked Driveway
- Blocked Hydrant
- Catch Basin Clogged
- Cave-In
- Dirty Sidewalk
- Double Parked
- Failed Street Repair
- Hydrant Leaking
- Inadequate Heat
- Loud Music/Party
- Loud Talking
- Mouse Sighting
- No Heat
- No Hot Water
- Overflowing Litter Basket
- Pothole
- Rat Sighting
- Signs of Rodents
- Water Main Break

The number of unique resolution descriptions associated with each descriptor was then examined.

---

# 5. Resolution Description Analysis

A key question for the RAG project was:

> How many different resolutions can occur for a given type of complaint?

The results showed that each descriptor generally has only **2–3 unique resolution descriptions**.

Examples:

### No Heat

Three possible resolutions were observed:

1. HPD was unable to gain access to inspect.
2. HPD inspected and violations were issued.
3. Heat was restored and no violation was issued.

### Mouse Sighting

Two possible resolutions:

1. No rodent activity was found.
2. Rodent activity was found and a violation was issued.

### Pothole

Two possible resolutions:

1. Repair was completed.
2. Repair was scheduled.

### Dirty Sidewalk

Two possible resolutions:

1. The location was cleaned.
2. A summons was issued to the property owner.

---

# 6. Important Finding: Resolution Text Is Highly Templated

The resolution descriptions are not arbitrary natural-language responses.

They are highly standardized templates.

For example, similar resolution descriptions occur across multiple descriptors:

```text
Pothole
Failed Street Repair
Cave-In
        ↓
Department of Transportation
        ↓
Inspected condition
        ↓
Repair completed / repair scheduled
```

Similarly:

```text
Mouse Sighting
Rat Sighting
Signs of Rodents
        ↓
Department of Health and Mental Hygiene
        ↓
Rodent activity found / not found
```

This means that the original dataset contains a relatively small resolution vocabulary.

---

# 7. Resolution Distribution

The frequency distribution of resolutions was also examined.

An important observation was that the possible resolutions were generally **not dominated by one outcome**.

Examples:

### Banging/Pounding

| Resolution | Percentage |
|---|---:|
| Resolution A | 50.03% |
| Resolution B | 49.97% |

### Blocked Driveway

| Resolution | Percentage |
|---|---:|
| Resolution A | 35.74% |
| Resolution B | 34.05% |
| Resolution C | 30.21% |

### No Heat

| Resolution | Percentage |
|---|---:|
| Resolution A | 35.25% |
| Resolution B | 32.47% |
| Resolution C | 32.28% |

### No Hot Water

| Resolution | Percentage |
|---|---:|
| Resolution A | 34.78% |
| Resolution B | 33.11% |
| Resolution C | 32.11% |

Therefore, although each descriptor has only a few possible outcomes, the resolution cannot simply be determined from the descriptor alone.

---

# 8. Borough Analysis

The relationship between:

```text
descriptor + borough → resolution
```

was investigated.

The results showed that borough generally did **not produce a dramatic separation between resolution outcomes**.

For example, `Banging/Pounding` remained approximately balanced across multiple boroughs.

Some differences were observed for individual combinations, but many had relatively small sample sizes.

Therefore:

> `borough` alone does not appear to provide sufficient additional information to justify a RAG system.

---

# 9. Resolution Time Analysis

The `created_date` and `closed_date` columns were converted to datetime values.

A new feature was created:

```text
resolution_time_hours =
closed_date - created_date
```

The median resolution times for different resolution descriptions were generally close to one another.

Observed median values were approximately:

```text
20–24 hours
```

Although some requests had very large maximum resolution times, resolution-time medians were not sufficiently different across resolution types to make this an obvious primary differentiating feature.

Therefore:

> Resolution time alone does not appear to provide a strong explanation for the different resolution outcomes.

---

# 10. Why the Initial Dataset Is Not Ideal for RAG

The main limitation of the initial dataset is **lack of contextual richness**.

The dataset essentially provides:

```text
Complaint Type
       ↓
Descriptor
       ↓
Resolution
```

with relatively little additional information.

Furthermore, the resolution descriptions are highly templated.

This creates a potential problem for a RAG system.

A simple retrieval system might learn:

```text
"No Heat"
      ↓
retrieve historical "No Heat" cases
      ↓
find one of 3 nearly identical resolution templates
```

This does not sufficiently demonstrate the value of semantic retrieval or RAG.

A simpler classification, lookup, or rule-based system could potentially reproduce a significant portion of this behavior.

Therefore, the initial dataset is useful for understanding the problem, but it is **not sufficiently rich for the intended RAG experiment**.

---

# 11. Decision: Move to a Larger and Richer Dataset

Instead of abandoning the NYC 311 problem, the project will move to the official:

**NYC 311 Service Requests from 2020 to Present**

dataset.

The official dataset is maintained by **NYC OpenData** and is updated daily. As of October 1, 2026, the dataset contains approximately **22.7 million rows and 44 columns**.

The dataset represents NYC 311 service requests that can be directed to specific city agencies.

NYC OpenData also states that the dataset was expanded with additional contextual fields, including:

- Council District
- Police Precinct
- Additional Details

These fields provide more information about individual service requests.

---

# 12. Additional Information Available in the Larger Dataset

Compared with the original 9-column dataset, the official dataset provides substantially more context.

Important fields include:

| Field | Potential usefulness |
|---|---|
| `unique_key` | Unique request identification |
| `created_date` | Temporal context |
| `closed_date` | Resolution timing |
| `agency` | Responding agency |
| `agency_name` | Full agency name |
| `complaint_type` | High-level problem |
| `descriptor` | Detailed problem |
| `descriptor_2` | Additional problem details |
| `location_type` | Type of location |
| `incident_zip` | Geographic context |
| `incident_address` | Incident location |
| `street_name` | Street context |
| `cross_street_1` | Nearby location |
| `cross_street_2` | Nearby location |
| `status` | Request status |
| `due_date` | Expected completion information |
| `resolution_description` | Historical resolution |
| `resolution_action_updated_date` | Last resolution update |
| `community_board` | Local geographic context |
| `council_district` | Political/geographic district |
| `police_precinct` | Police geographic context |
| `borough` | Borough |
| `latitude` | Geographic position |
| `longitude` | Geographic position |

The official documentation describes `descriptor_2` as a third level of detail beyond the problem and problem detail, although it is not used for every category.

---

# 13. New RAG Problem Formulation

The project will therefore move away from:

```text
Descriptor → Resolution
```

and investigate a richer formulation:

```text
New Service Request
        ↓
Semantic Representation
        ↓
Retrieve Similar Historical Service Requests
        ↓
Retrieve:
    - Similar problem
    - Additional details
    - Location context
    - Agency
    - Historical resolution
        ↓
LLM
        ↓
Evidence-grounded response
```

The goal is not simply to predict one of a few fixed resolution templates.

Instead, the system should investigate whether **historical service requests can provide useful contextual evidence for understanding or resolving a new request**.

---

# 14. Dataset Size for Initial Experiment

The official dataset is very large, with tens of millions of records.

Rather than loading the complete dataset into Google Colab, the initial experiment will use approximately:

```text
100,000 recent service requests
```

This provides a sufficiently large corpus for:

- EDA
- Statistical analysis
- Text analysis
- Embedding generation
- Semantic search
- Vector database experiments
- RAG evaluation

while remaining manageable within a Google Colab environment.

If the 100K-record experiment produces promising results, the dataset can later be expanded to 500K or more records.

---

# 15. Embedding Strategy

For the RAG experiment, the relevant fields will be combined into a textual representation of each historical service request.

An example representation could be:

```text
Problem: Heat/Hot Water
Problem Detail: No Heat
Additional Details: ...
Agency: HPD
Location Type: Residential Building
Borough: BRONX
Status: Closed
Resolution: Heat was restored at the time of inspection. No violation issued.
```

These documents can then be converted into semantic embeddings.

A lightweight sentence-transformer model such as:

```text
BAAI/bge-small-en-v1.5
```

can initially be evaluated for embedding generation.

The project will use the available Colab T4 GPU to accelerate embedding generation.

---

# 16. Vector Retrieval

The generated embeddings can be stored using a vector similarity-search library such as **FAISS**.

The proposed architecture is:

```text
100K Historical Tickets
        ↓
Text Construction
        ↓
BGE Embedding Model
        ↓
384-dimensional vectors
        ↓
FAISS Index
        ↓
Semantic Similarity Search
        ↓
Top-K Historical Cases
```

The retrieved historical cases can then be provided as context to an LLM.

---

# 17. Current Status

### Completed

- [x] Loaded initial `tickets.csv`
- [x] Verified dataset dimensions: 25,921 × 9
- [x] Examined complaint-type distribution
- [x] Investigated descriptors
- [x] Counted unique resolutions per descriptor
- [x] Examined actual resolution descriptions
- [x] Examined resolution distributions
- [x] Investigated borough-level resolution distributions
- [x] Calculated resolution time
- [x] Investigated resolution time by resolution
- [x] Identified limitations of the initial dataset
- [x] Identified the official NYC 311 dataset as a richer source

### Next Steps

- [ ] Download approximately 100K recent NYC 311 records
- [ ] Perform EDA on the larger dataset
- [ ] Analyze missing values and feature coverage
- [ ] Analyze `descriptor_2` / Additional Details
- [ ] Identify useful contextual fields
- [ ] Determine the best representation of a historical ticket
- [ ] Construct retrieval documents
- [ ] Generate embeddings
- [ ] Build FAISS vector index
- [ ] Evaluate semantic retrieval
- [ ] Design the RAG pipeline
- [ ] Evaluate retrieval quality and answer grounding

---

# 18. Dataset Source

The primary dataset for the next stage is:

**NYC OpenData — 311 Service Requests from 2020 to Present**

The dataset is publicly available and updated daily.

Official dataset:

[NYC 311 Service Requests — 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9?utm_source=chatgpt.com)

The NYC OpenData documentation also provides information about filtering and exporting subsets of the dataset, which will be useful when creating a manageable 100K-record corpus.

---

# 19. Key Conclusion

The initial 25,921-row dataset is **useful for preliminary EDA but insufficiently rich for demonstrating the full value of RAG**.

The main issue is not simply the number of rows.

The more important issue is the limited contextual information:

```text
9 columns
      +
highly templated resolutions
      +
few descriptors
      +
limited contextual features
```

The larger official NYC 311 dataset provides substantially more contextual information and millions of historical service requests. This creates a stronger foundation for investigating:

> **Can semantic retrieval of historically similar service requests provide useful evidence and context for resolving or understanding new NYC 311 service requests?**

The project will therefore proceed with a **100K-record subset of the official NYC 311 dataset**, followed by a second round of EDA before implementing the embedding and RAG pipeline.