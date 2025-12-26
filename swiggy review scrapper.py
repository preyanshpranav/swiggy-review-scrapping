#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas numpy google-play-scraper sentence-transformers rapidfuzz matplotlib seaborn')


# In[7]:


get_ipython().system('pip install google-play-scraper')


# In[10]:


from google_play_scraper import reviews, Sort
import pandas as pd
import datetime as dt

app_id = "in.swiggy.android"
target_date = dt.date(2024, 6, 1)  # example date

result, _ = reviews(
    app_id,
    lang="en",
    country="in",
    sort=Sort.NEWEST,
    count=2000
)

df = pd.DataFrame(result)
df["at"] = pd.to_datetime(df["at"])

# Filter
df_day = df[df["at"].dt.date == target_date]

print("Reviews found:", len(df_day))
df_day.head()


# In[11]:


import re

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df_day["text_norm"] = df_day["content"].apply(normalize_text)
df_day.head()


# In[2]:


get_ipython().system('pip install sentence-transformers')


# In[4]:


get_ipython().system('pip install rapidfuzz')


# In[5]:


# TOPIC CLASSIFICATION AGENTIC APPROACH
# ---- 0) Imports
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# ---- 1) Seed topics (ontology) — clearly defined here
TOPICS = {
    "Delivery issue": [
        "late delivery", "order delayed", "delay", "delayed", "eta wrong",
        "no show", "wrong route", "driver took long", "stuck in traffic"
    ],
    "Food stale": [
        "cold food", "not fresh", "stale", "spoiled", "bad quality",
        "soggy", "tasteless", "smelly"
    ],
    "Delivery partner rude": [
        "rude", "impolite", "misbehaved", "abusive", "bad attitude",
        "unprofessional", "misbehavior", "behaved badly"
    ],
    "Maps not working properly": [
        "location error", "gps issue", "pin wrong", "map issue",
        "navigation problem", "wrong address", "maps not working"
    ],
    "Instamart should be open all night": [
        "24x7 instamart", "open late", "night availability", "midnight hours",
        "keep instamart open at night"
    ],
    "Bring back 10 minute bolt delivery": [
        "10 min delivery", "instant delivery", "bolt delivery", "bring back 10 minute",
        "return instant delivery"
    ],
    # Add high-recall operational topics often seen:
    "Payment/refund issue": [
        "refund not processed", "money deducted", "payment failed", "refund pending",
        "double charged", "upi failed", "card declined"
    ],
    "Order accuracy issue": [
        "wrong item", "missing item", "mixed up order", "incorrect order",
        "item not delivered", "item swapped"
    ],
    "App performance/login issue": [
        "app crashing", "keeps crashing", "login issue", "otp not received",
        "slow app", "hangs", "buggy", "freeze"
    ],
    "Pricing/fees issue": [
        "surge pricing", "hidden charges", "delivery fees high", "service charge high",
        "extra charges", "unreasonable fees"
    ],
    "Customer support unresponsive": [
        "support not helpful", "no callback", "ticket unresolved",
        "chat bot useless", "no response from support"
    ],
    "Subscription/membership issue": [
        "swiggy one not applied", "membership issue", "benefits not applied",
        "subscription problem", "prime not working"
    ],
}

# ---- 2) Normalization helper
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- 3) Prepare phrase memory + embeddings
topic_phrases, topic_map = [], []
for label, phrases in TOPICS.items():
    for p in phrases:
        topic_phrases.append(p)
        topic_map.append(label)

# Small, fast model suitable for Jupyter
model = SentenceTransformer("all-MiniLM-L6-v2")
topic_embs = model.encode(topic_phrases, normalize_embeddings=True)

# ---- 4) Agentic classifier (explainable steps)
def agent_classify_review(text):
    """
    Steps:
    1) Normalize text
    2) Fuzzy check (high-recall literal cues)
    3) Semantic check (meaning-based)
    4) Fallback to 'Unclassified'
    """
    text_norm = normalize_text(text)

    # Step 2: fuzzy check (quick wins)
    match = process.extractOne(text_norm, topic_phrases, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 78:
        return topic_map[topic_phrases.index(match[0])]

    # Step 3: semantic check (meaning)
    emb = model.encode([text_norm], normalize_embeddings=True)[0]
    sims = [float(emb @ t_emb) for t_emb in topic_embs]
    if sims:
        best_idx = int(pd.Series(sims).idxmax())
        best_topic = topic_map[best_idx]
        best_score = sims[best_idx]
        if best_score >= 0.35:
            return best_topic

    # Step 4: fallback
    return "Unclassified"

# ---- 5) Example usage (replace df with your reviews dataframe)
# df should have a 'content' column (review text)
# If you already have df from google_play_scraper, just run the three lines below:

# Ensure normalized column exists
# df["text_norm"] = df["content"].apply(normalize_text)

# Classify to topics
# df["topic"] = df["content"].apply(agent_classify_review)

# Inspect
# display(df[["text_norm", "topic"]].head(10))


# In[7]:


demo = pd.DataFrame({"content": [
    "Delivery guy was rude and impolite.",
    "Order delayed by 40 minutes, ETA was wrong.",
    "Food arrived cold and soggy, very bad quality.",
    "Maps took the driver to the wrong pin.",
    "Please keep Instamart open late at night.",
    "Payment failed, money deducted but refund not processed.",
    "App keeps crashing during login, OTP not received.",
    "Bring back the 10 minute instant delivery!",
]})

demo["text_norm"] = demo["content"].apply(normalize_text)
demo["topic"] = demo["content"].apply(agent_classify_review)
display(demo)


# In[5]:


# 1) Imports
import re
import pandas as pd
from google_play_scraper import reviews, Sort
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# 2) Normalize helper
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 3) Seed topics
TOPICS = {
    "Delivery issue": ["late delivery", "order delayed", "wrong route", "delay"],
    "Food stale": ["cold food", "not fresh", "stale", "spoiled"],
    "Delivery partner rude": ["rude", "impolite", "misbehaved", "abusive"],
    "Maps not working properly": ["location error", "gps issue", "pin wrong", "map issue"],
    "Instamart should be open all night": ["24x7 instamart", "open late", "night availability"],
    "Bring back 10 minute bolt delivery": ["10 min delivery", "instant delivery", "bolt delivery"],
}

# 4) Build embeddings
topic_phrases, topic_map = [], []
for label, phrases in TOPICS.items():
    for p in phrases:
        topic_phrases.append(p)
        topic_map.append(label)

model = SentenceTransformer("all-MiniLM-L6-v2")
topic_embs = model.encode(topic_phrases, normalize_embeddings=True)

# 5) Agentic classifier
def agent_classify_review(text):
    text_norm = normalize_text(text)
    # Fuzzy check
    match = process.extractOne(text_norm, topic_phrases, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 78:
        return topic_map[topic_phrases.index(match[0])]
    # Semantic check
    emb = model.encode([text_norm], normalize_embeddings=True)[0]
    sims = [float(emb @ t_emb) for t_emb in topic_embs]
    if sims:
        best_idx = int(pd.Series(sims).idxmax())
        best_topic = topic_map[best_idx]
        if sims[best_idx] >= 0.35:
            return best_topic
    return "Unclassified"

# 6) Fetch reviews (example: Swiggy)
app_id = "in.swiggy.android"
result, _ = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=500)
df = pd.DataFrame(result)
df["at"] = pd.to_datetime(df["at"])
df["date"] = df["at"].dt.date

# 7) Apply classifier
df["text_norm"] = df["content"].apply(normalize_text)
df["topic"] = df["content"].apply(agent_classify_review)

# 8) Daily topic counts
trend = df.groupby(["date", "topic"]).size().reset_index(name="count")
print(trend.head(10))

# 9) Pivot into trend table
pivot = trend.pivot(index="topic", columns="date", values="count").fillna(0).astype(int)
print(pivot)


# In[6]:


# Daily Topic Counts
 # Count how many reviews fall into each topic per day
trend = df.groupby(["date", "topic"]).size().reset_index(name="count")
trend.head(10)


# In[7]:


pivot = trend.pivot(index="topic", columns="date", values="count").fillna(0).astype(int)
pivot


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
sns.heatmap(pivot, cmap="Reds", linewidths=0.5)
plt.title("Topic trend over last 30 days")
plt.show()


# In[2]:


get_ipython().system('pip install google-play-scraper')


# In[3]:


from google_play_scraper import reviews, Sort


# In[5]:


# Example: Swiggy app reviews
import pandas as pd
app_id = "in.swiggy.android"

result, _ = reviews(
    app_id,
    lang="en",
    country="in",
    sort=Sort.NEWEST,
    count=500   # number of reviews to fetch
)

df = pd.DataFrame(result)
df["at"] = pd.to_datetime(df["at"])
df["date"] = df["at"].dt.date

df.head()


# In[8]:





# In[1]:


# ---- Imports
import re
import pandas as pd
from google_play_scraper import reviews, Sort
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# ---- Normalization helper
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- Seed topics
TOPICS = {
    "Delivery issue": ["late delivery", "order delayed", "wrong route", "delay"],
    "Food stale": ["cold food", "not fresh", "stale", "spoiled"],
    "Delivery partner rude": ["rude", "impolite", "misbehaved", "abusive"],
    "Maps not working properly": ["location error", "gps issue", "pin wrong", "map issue"],
    "Instamart should be open all night": ["24x7 instamart", "open late", "night availability"],
    "Bring back 10 minute bolt delivery": ["10 min delivery", "instant delivery", "bolt delivery"],
}

# ---- Build embeddings
topic_phrases, topic_map = [], []
for label, phrases in TOPICS.items():
    for p in phrases:
        topic_phrases.append(p)
        topic_map.append(label)

model = SentenceTransformer("all-MiniLM-L6-v2")
topic_embs = model.encode(topic_phrases, normalize_embeddings=True)

# ---- Classifier
def agent_classify_review(text):
    text_norm = normalize_text(text)
    match = process.extractOne(text_norm, topic_phrases, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 78:
        return topic_map[topic_phrases.index(match[0])]
    emb = model.encode([text_norm], normalize_embeddings=True)[0]
    sims = [float(emb @ t_emb) for t_emb in topic_embs]
    if sims:
        best_idx = int(pd.Series(sims).idxmax())
        if sims[best_idx] >= 0.35:
            return topic_map[best_idx]
    return "Unclassified"

# ---- Fetch reviews (Swiggy app)
app_id = "in.swiggy.android"
result, _ = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=500)
df = pd.DataFrame(result)
df["at"] = pd.to_datetime(df["at"])
df["date"] = df["at"].dt.date

# ---- Apply classifier
df["text_norm"] = df["content"].apply(normalize_text)
df["topic"] = df["content"].apply(agent_classify_review)

# ---- Daily topic counts
trend = df.groupby(["date", "topic"]).size().reset_index(name="count")
display(trend.head(10))

# ---- Pivot into 30‑day trend table
pivot = trend.pivot(index="topic", columns="date", values="count").fillna(0).astype(int)
last_30 = pivot.iloc[:, -30:]
display(last_30)


# In[2]:


df["text_norm"] = df["content"].apply(normalize_text)
df["topic"] = df["content"].apply(agent_classify_review)

trend = df.groupby(["date", "topic"]).size().reset_index(name="count")
display(trend.head(10))


# In[ ]:




