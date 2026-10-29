"""
Generate the 5 publication-ready Jupyter notebooks for the project.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# ---------------------------------------------------------------------------
# 01_data_exploration.ipynb
# ---------------------------------------------------------------------------
nb1_cells = [
    md("""# 01. Temporal Knowledge Graph & Dataset Exploration

This notebook provides an in-depth exploratory data analysis of the datasets driving the **TKG-Verify** framework:
1. **ICEWS14** (Integrated Crisis Early Warning System 2014): Daily event-based temporal knowledge graph quadruples.
2. **CronQuestions**: Complex temporal question answering benchmark over Wikidata quintuples.
"""),
    code("""import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json

# Ensure project root is in sys.path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from config import settings
from src.data_loader.icews_loader import ICEWS14Loader
from src.data_loader.cronquestions_loader import CronQuestionsLoader"""),
    md("""## 1. Exploring ICEWS14 Dataset

The ICEWS14 dataset contains daily international political events from the year 2014 formatted as temporal quadruples:
`$(e_s, r, e_o, t)$` where `t` is a discrete daily timestamp.
"""),
    code("""icews_loader = ICEWS14Loader(data_dir=settings.ICEWS14_DIR)
icews_stats = icews_loader.get_statistics()
print("ICEWS14 Statistics:")
for k, v in icews_stats.items():
    print(f"  {k}: {v}")"""),
    code("""splits = icews_loader.load_all()
train_facts = splits.get("train", [])
print(f"Loaded {len(train_facts):,} training facts.")
df_sample = pd.DataFrame(train_facts[:10])
df_sample"""),
    md("""### Temporal Event Distribution across 2014"""),
    code("""dates = [f["timestamp"] for f in train_facts]
df_dates = pd.DataFrame({"timestamp": pd.to_datetime(dates)})
monthly_counts = df_dates["timestamp"].dt.to_period("M").value_counts().sort_index()

plt.figure(figsize=(10, 4))
monthly_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("ICEWS14 Event Frequency by Month (2014)")
plt.xlabel("Month")
plt.ylabel("Number of Recorded Events")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()"""),
    md("""## 2. Exploring CronQuestions Dataset

CronQuestions evaluates temporal reasoning over Wikidata quintuples:
`$(e_s, r, e_o, t_{start}, t_{end})$` with corresponding natural language questions across multiple reasoning types (time-to-entity, entity-to-time, before/after, simultaneous).
"""),
    code("""cron_loader = CronQuestionsLoader(data_dir=settings.CRONQUESTIONS_DIR)
cron_stats = cron_loader.get_statistics()
print("CronQuestions Statistics:")
for k, v in cron_stats.items():
    print(f"  {k}: {v}")"""),
    code("""test_questions = cron_loader.load_questions(split="test")
print(f"Sample test questions ({len(test_questions)} total):")
for q in test_questions[:5]:
    print(f"Q: {q['question']}")
    print(f"   Type: {q['answer_type']} | Expected: {q['answer']}")""")
]

# ---------------------------------------------------------------------------
# 02_graph_analysis.ipynb
# ---------------------------------------------------------------------------
nb2_cells = [
    md("""# 02. In-Memory Temporal Knowledge Graph Analysis

This notebook analyzes the graph topology, degree distributions, and event-reification pattern of the instantiated temporal knowledge graph using `InMemoryKGStore` (NetworkX-based).
"""),
    code("""import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from config import settings
from src.knowledge_graph.memory_store import InMemoryKGStore"""),
    md("""## 1. Loading the Processed Knowledge Graph"""),
    code("""store = InMemoryKGStore()
graph_path = settings.PROCESSED_DATA_DIR / "knowledge_graph.json"
print(f"Loading cached graph from {graph_path}...")
store.load(graph_path)

stats = store.get_statistics()
print("Graph Store Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v:,}")"""),
    md("""## 2. Graph Topologies & Reified Event Pattern

Events are modeled using the **Event Node Reification pattern**:
`(Entity: Actor) -> [:ACTOR] -> (Event) -> [:TARGET] -> (Entity: Target)`
"""),
    code("""G = store.graph
print(f"Total NetworkX Nodes: {G.number_of_nodes():,}")
print(f"Total NetworkX Edges: {G.number_of_edges():,}")

entity_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]
event_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "event"]
print(f"Entity nodes: {len(entity_nodes)}")
print(f"Event nodes:  {len(event_nodes)}")"""),
    md("""### Degree Distribution of Top Political Entities"""),
    code("""degrees = dict(G.degree())
entity_degrees = {G.nodes[n].get("name", n): degrees[n] for n in entity_nodes}
top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:15]

df_top = pd.DataFrame(top_entities, columns=["Entity", "Degree"])
plt.figure(figsize=(10, 5))
plt.barh(df_top["Entity"][::-1], df_top["Degree"][::-1], color="teal")
plt.title("Top 15 Most Connected Entities in Knowledge Graph")
plt.xlabel("Degree (Event Connections)")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()"""),
    md("""## 3. Querying Entity Relationships Across Time"""),
    code("""sample_events = store.get_events_between_entities("United States", "Russia", limit=5)
print(f"Sample events between US and Russia:")
for ev in sample_events:
    print(f"  {ev.get('timestamp')}: {ev.get('subject')} --[{ev.get('predicate')}]--> {ev.get('object')}")""")
]

# ---------------------------------------------------------------------------
# 03_retrieval_demo.ipynb
# ---------------------------------------------------------------------------
nb3_cells = [
    md("""# 03. Temporal Fact Retrieval Pipeline Demo

Demonstration of the **Temporal Fact Retrieval** subsystem:
1. **Query Parsing**: Entity linking via fuzzy & known-entity resolution, plus regex temporal extraction.
2. **Template Selection**: Cypher/Query dispatch (`entity_at_time`, `entity_pair`, `events_before`, `events_after`).
3. **Relevance Ranking**: Temporal proximity scoring and context formatting.
"""),
    code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from config import settings
from src.knowledge_graph.memory_store import InMemoryKGStore
from src.retrieval.query_parser import QueryParser
from src.retrieval.temporal_retriever import TemporalRetriever"""),
    md("""## 1. Initializing Query Parser & Graph Store"""),
    code("""store = InMemoryKGStore()
store.load(settings.PROCESSED_DATA_DIR / "knowledge_graph.json")

entity_names = store.get_all_entity_names()
parser = QueryParser(known_entities=entity_names)
retriever = TemporalRetriever(graph_store=store, query_parser=parser, top_k=5)
print(f"Loaded retriever with {len(entity_names)} known entities.")"""),
    md("""## 2. Natural Language Query Parsing"""),
    code("""test_queries = [
    "Who did the United States make statements to in March 2014?",
    "What events occurred between United States and Russia in 2014?",
    "What happened before June 2014 regarding China?",
    "What events occurred after October 2014 involving Germany?"
]

for query in test_queries:
    parsed = parser.parse(query)
    print(f"Query: {query}")
    print(f"  Query Type:  {parsed['query_type']}")
    print(f"  Entities:    {parsed['entities']}")
    print(f"  Date Range:  {parsed['date_range']}")
    print("-" * 60)"""),
    md("""## 3. End-to-End Temporal Fact Retrieval"""),
    code("""sample_q = "What events occurred between United States and Russia in March 2014?"
facts = retriever.retrieve(sample_q, top_k=5)

print(f"Retrieved {len(facts)} facts for: '{sample_q}':")
for i, f in enumerate(facts, 1):
    print(f"{i}. [{f['timestamp']}] ({f['subject']}) --[{f['predicate']}]--> ({f['object']}) [Relevance: {f['relevance_score']:.2f}]")""")
]

# ---------------------------------------------------------------------------
# 04_verification_demo.ipynb
# ---------------------------------------------------------------------------
nb4_cells = [
    md("""# 04. Multi-Stage Temporal Claim Verification & Hallucination Detection

Demonstration of the core verification engine:
1. **Multi-Stage Verification**: Entity existence, relation validation, timestamp distance calculation, and chronological ordering.
2. **Hallucination Detection**: Categorizing claim validity into SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, and TEMPORALLY_IMPOSSIBLE.
3. **Explanation Generation**: Automated human-readable justifications.
"""),
    code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from config import settings
from src.knowledge_graph.memory_store import InMemoryKGStore
from src.verification.temporal_verifier import TemporalVerifier
from src.verification.verification_result import TemporalClaim
from src.hallucination.detector import HallucinationDetector
from src.explanation.explainer import ExplanationGenerator"""),
    md("""## 1. Initializing Verification Components"""),
    code("""store = InMemoryKGStore()
store.load(settings.PROCESSED_DATA_DIR / "knowledge_graph.json")

verifier = TemporalVerifier(graph_store=store, timestamp_tolerance_days=0)
detector = HallucinationDetector()
explainer = ExplanationGenerator()
print("Verification engine ready.")"""),
    md("""## 2. Testing Genuine Fact vs Controlled Hallucinations"""),
    code("""# 1. Genuine Fact from ICEWS14
clean_claim = TemporalClaim(
    subject="South Korea",
    predicate="Make a visit",
    object="Mexico",
    timestamp="2014-03-16"
)

# 2. Timestamp Corruption (Temporal Hallucination)
corrupt_ts_claim = TemporalClaim(
    subject="South Korea",
    predicate="Make a visit",
    object="Mexico",
    timestamp="2014-05-01"
)

# 3. Entity Corruption (Factual Hallucination)
corrupt_ent_claim = TemporalClaim(
    subject="South Korea",
    predicate="Make a visit",
    object="FictionalAtlantis",
    timestamp="2014-03-16"
)

claims_to_test = [clean_claim, corrupt_ts_claim, corrupt_ent_claim]"""),
    code("""for idx, claim in enumerate(claims_to_test, 1):
    vr = verifier.verify(claim)
    report = detector.classify_claim(vr)
    explanation = explainer.explain(vr)
    
    print(f"--- Claim #{idx} ---")
    print(f"Triple: ({claim.subject}, {claim.predicate}, {claim.object}, {claim.timestamp})")
    print(f"Status:       {vr.status.value}")
    print(f"Confidence:   {vr.confidence}")
    print(f"Hallucinated: {report['is_hallucination']}")
    print(f"Explanation:  {explanation}")
    print()""")
]

# ---------------------------------------------------------------------------
# 05_evaluation_results.ipynb
# ---------------------------------------------------------------------------
nb5_cells = [
    md("""# 05. Benchmark Evaluation & Empirical Results

This notebook analyzes and visualizes the quantitative findings of **TKG-Verify**:
1. Synthetic Hallucination Benchmark results across corruption types (Timestamp, Entity, Relation, Ordering).
2. Comparative QA Evaluation on CronQuestions across all 4 experimental baselines.
3. Component Ablation Study results demonstrating the impact of each subsystem.
"""),
    code("""import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from config import settings"""),
    md("""## 1. Benchmark Detection Results"""),
    code("""bench_path = settings.PROJECT_ROOT / "evaluation" / "results" / "benchmark_results.json"
with open(bench_path, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

print(f"Overall Accuracy:  {bench_data['accuracy']:.4f}")
print(f"Overall Precision: {bench_data['precision']:.4f}")
print(f"Overall Recall:    {bench_data['recall']:.4f}")
print(f"Overall F1 Score:  {bench_data['f1']:.4f}")

per_type = bench_data["per_corruption_type"]
df_types = pd.DataFrame(per_type).T
df_types"""),
    md("""### Detection Accuracy by Corruption Mechanism"""),
    code("""plt.figure(figsize=(8, 4))
df_types["accuracy"].plot(kind="bar", color="cornflowerblue", edgecolor="black")
plt.title("Hallucination Detection Accuracy Across Corruption Types")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()"""),
    md("""## 2. CronQuestions Baseline Comparison"""),
    code("""cron_path = settings.PROJECT_ROOT / "evaluation" / "results" / "cronquestions_results.json"
with open(cron_path, "r", encoding="utf-8") as f:
    cron_data = json.load(f)

df_cron = pd.DataFrame(cron_data["comparison"]).T
df_cron"""),
    md("""## 3. Component Ablation Study"""),
    code("""ablation_path = settings.PROJECT_ROOT / "evaluation" / "results" / "ablation_results.json"
with open(ablation_path, "r", encoding="utf-8") as f:
    ablation_data = json.load(f)

df_ablation = pd.DataFrame(ablation_data.get("conditions", ablation_data)).T
df_ablation""")
]

# Write out all notebooks
notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_graph_analysis.ipynb": nb2_cells,
    "03_retrieval_demo.ipynb": nb3_cells,
    "04_verification_demo.ipynb": nb4_cells,
    "05_evaluation_results.ipynb": nb5_cells,
}

for name, cells in notebooks.items():
    path = NOTEBOOKS_DIR / name
    nb_json = make_notebook(cells)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)
    print(f"Created {path}")

print("All 5 Jupyter notebooks generated successfully.")
