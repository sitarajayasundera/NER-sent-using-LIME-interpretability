# Do Named Entities Predict Sentiment? A Compositional Analysis using LIME

**Author:** Sitara Jayasundera  
**Affiliation:** University of Trier, Master's NLP Program  
**Submission:** NLP Seminar Poster Examination (September 2026)  

---

##  Research Question

**Do named entity types (PERSON, ORG, LOC, etc.) predict sentiment in movie reviews, and through what linguistic mechanisms?**

### Hypothesis
Entity types correlate with sentiment at the statistical level, but linguistic context (not entity type alone) drives sentiment—a compositional model where **Sentiment = Entity Type × Context**.

---

## Dataset

**Source:** [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Metric | Value |
|--------|-------|
| Reviews analyzed | 5,000 |
| Negative reviews | 2,632 (52.6%) |
| Positive reviews | 2,368 (47.4%) |
| Reviews with entities | 4,894 (97.9%) |
| Total entity mentions | 37,919 |
| Unique entity types | 18 |

---

##  Methodology

### 1. **Named Entity Recognition (NER)**
- **Model:** spaCy `en_core_web_sm`
- **Entity types:** PERSON, ORG, GPE, LOC, TIME, DATE, MONEY, QUANTITY, CARDINAL, EVENT, PRODUCT, FAC, LAW, LANGUAGE, NORP, ORDINAL, PERCENT, WORK_OF_ART

### 2. **Sentiment Classification**
- **Model:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Labels:** NEGATIVE, POSITIVE
- **Max sequence length:** 512 tokens

### 3. **Statistical Analysis**
- **Chi-Square Test:** Association between entity type and sentiment
- **Cramér's V:** Effect size (range: 0–1)
- **Odds Ratios:** Directional influence of each entity type

### 4. **Explainability**
- **Method:** LIME (Local Interpretable Model-Agnostic Explanations)
- **Per-entity analysis:** Top 6 entity types (PERSON, ORG, LOC, GPE, TIME, MONEY)
- **Samples per entity:** 10 positive + 10 negative reviews (60 explanations total)
- **Feature importance:** 6 words per explanation

---

## Key Findings

### Overall Association
```
Chi-Square Test:
  χ² = 293.58, p = 2.38e-52 (HIGHLY SIGNIFICANT)
  DoF = 17
```

### Effect Sizes (Cramér's V) — All Negligible (<0.1)
| Entity Type | V | Odds Ratio | Direction |
|-------------|---|------------|-----------|
| TIME | 0.0562 | 0.48 | ↓ Pro-Negative |
| PERSON | 0.0428 | 1.19 | ↑ Pro-Positive |
| CARDINAL | 0.0373 | 0.89 | Neutral |
| MONEY | 0.0257 | 0.42 | ↓ Pro-Negative |
| GPE | 0.0188 | 1.15 | ↑ Pro-Positive |
| LOC | 0.0168 | 1.41 | ↑ Pro-Positive |

### Interpretation
- **Entity type explains <1% of sentiment variance**
- **Linguistic context explains ~99% of variance**
- **Pattern:** Temporal/financial framing → negative sentiment; Personal/locational context → positive sentiment

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local environment
- 4GB+ RAM (for model loading)

### Dependencies
```bash
pip install transformers datasets lime spacy scikit-learn matplotlib seaborn pandas numpy scipy
python -m spacy download en_core_web_sm
```

### Kaggle Dataset Access
The notebook downloads the dataset directly from Kaggle using:
```bash
pip install kaggle
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

**Note:** Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) first. See [Kaggle API docs](https://github.com/Kaggle/kaggle-api).

---

##  Repository Structure

```
.
├── README.md                           # This file
├── NER_POSTER_1847235.ipynb           # Full reproducible notebook
├── requirements.txt                    # Python dependencies
└── data/
    └── IMDB Dataset.csv               # (Auto-downloaded by notebook)
```

---

## 🚀 Running the Notebook

### Option 1: Google Colab (Recommended)
1. Upload `NER_POSTER_1847235.ipynb` to Google Colab
2. Run all cells sequentially
3. Expected runtime: **30–40 minutes** (model loading + LIME analysis)

### Option 2: Local Jupyter
```bash
jupyter notebook NER_POSTER_1847235.ipynb
```

### Cell-by-Cell Breakdown
| Cell | Task | Runtime |
|------|------|---------|
| 1–2 | Imports & model loading | 2–3 min |
| 3 | Load Kaggle IMDB dataset | 1 min |
| 4 | Process 5,000 reviews (NER + sentiment) | 10–15 min |
| 5 | Statistical analysis (Chi-Square, Cramér's V) | <1 min |
| 6 | LIME explainability on 6 entities | 20–25 min |
| 7 | Error analysis (200 samples) | 2 min |

---

##  Important Notes

### Training Data Overlap
- **Finding:** 100% accuracy on 200 random test samples
- **Cause:** DistilBERT was pre-trained on IMDB; evaluation data overlaps with training corpus
- **Implication:** Findings reflect genuine IMDB patterns but lack cross-dataset generalizability
- **Mitigation:** For production use, validate on non-overlapping corpora (e.g., Amazon reviews, SemEval)

### Model Limitations
- **DistilBERT:** Binary classification only (NEGATIVE/POSITIVE), no neutral class
- **spaCy NER:** General-purpose model; lower performance on domain-specific entities
- **LIME:** Local interpretability; explanations are sample-specific, not global

---

##  Outputs Generated

### Statistics & Analysis
- Chi-Square test results
- Cramér's V effect sizes (18 entity types)
- Odds ratios & directional effects
- Entity type distribution across sentiment labels

### Visualizations (6 generated)
1. **Cramér's V effect sizes** (all 18 entities)
2. **Odds ratios** (directional influence)
3. **LIME explanations (positive)** by entity (6 entities)
4. **LIME explanations (negative)** by entity (6 entities)
5. **Error analysis** summary
6. **Sentiment distribution** by entity type


