# Swiggy Review Scrapping
# 🧠 Google Play Store Review Trend Analysis

An automated, agentic AI pipeline that fetches Google Play Store reviews, normalizes text, classifies them into human‑explainable topics, and generates a **30‑day trend analysis report table** showing how issues evolve over time.

---

## Live Demo
(Not applicable — this is a Jupyter Notebook project. See sample outputs in `/output/`.)

---

## Tech Stack
- Python 3.10+
- Jupyter Notebook
- pandas (dataframes, aggregation, pivot)
- google-play-scraper (fetch reviews)
- sentence-transformers (semantic embeddings)
- rapidfuzz (fuzzy text matching)
- seaborn / matplotlib (optional visualization)

---

## ⚙️ Setup Instructions

** 1. Clone the repository**

Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook assignment.ipynb

4️⃣ Provide inputs
- App store link (e.g. https://play.google.com/store/apps/details?id=in.swiggy.android)
- Target date (e.g. 2025-12-20)
  
5️⃣ Generate outputs
- Normalized + classified reviews
- Daily topic counts (trend)
- Final 30‑day trend analysis table (last_30)
- Exported CSVs in /output/

✨ Key Features Implemented
📝 Automated Pipeline
- Fetches reviews directly from Google Play Store
- Cleans text with normalization function
- Classifies reviews into topics using fuzzy + semantic matching
  
📊 Trend Analysis
- Groups reviews by date and topic
- Builds a 30‑day pivot table for trend reporting
  
🎨 Human‑Explainable Classifier
- Topics defined with seed phrases (e.g. “Delivery issue”, “Food stale”)
- Transparent logic: fuzzy match → semantic similarity → fallback
  
📂 Sample Outputs
- Daily counts (trend_sample.csv)
- 30‑day trend table (trend_table_30days.csv)

🧩 Assumptions Made
- Reviews are fetched fresh at runtime (no cached dataset).
- Topics are predefined and limited to assignment scope.
- Analysis is focused on last 30 days only.
- No authentication or external API beyond google-play-scraper.

⏱ Time Spent on the Assignment
- Setup & environment: 1 hour
- Data fetching & cleaning: 1.5 hours
- Classifier design & testing: 2 hours
- Trend aggregation & pivot: 1 hour
- Documentation & outputs: 1 hour
Total: ~6.5 hours

📂Video Link:- https://drive.google.com/file/d/1V7F1wqJMfOG-cPfvIqvPpNkIkEhQq0aw/view?usp=sharing

```bash
git clone <your-repo-link>
cd swiggy-review-trends
