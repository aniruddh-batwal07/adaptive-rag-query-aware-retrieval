import matplotlib.pyplot as plt
import numpy as np
import os
import json

os.makedirs("paper/figures", exist_ok=True)

# GRAPH 1: Classifier Performance (DistilBERT vs BERT)
with open("results/router_comparison/comparison.json", "r") as f:
    comp = json.load(f)

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
distilbert_scores = [
    comp['distilbert']['accuracy'] * 100,
    comp['distilbert']['precision'] * 100,
    comp['distilbert']['recall'] * 100,
    comp['distilbert']['f1'] * 100
]
bert_scores = [
    comp['bert']['accuracy'] * 100,
    comp['bert']['precision'] * 100,
    comp['bert']['recall'] * 100,
    comp['bert']['f1'] * 100
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
rects1 = ax.bar(x - width/2, distilbert_scores, width, label='DistilBERT', color='#1f77b4')
rects2 = ax.bar(x + width/2, bert_scores, width, label='BERT', color='#ff7f0e')

ax.set_ylabel('Percentage (%)')
ax.set_title('Router Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_ylim(90, 101)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig("paper/figures/classifier_performance.png", dpi=300)
plt.close()

# GRAPH 2: Classifier Inference Latency
labels = ['DistilBERT', 'BERT']
latencies = [
    comp['distilbert']['avg_inference_latency_ms'],
    comp['bert']['avg_inference_latency_ms']
]

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(labels, latencies, color=['#1f77b4', '#ff7f0e'], width=0.5)
ax.set_ylabel('Average Latency (ms)')
ax.set_title('Router Inference Latency')
for i, v in enumerate(latencies):
    ax.text(i, v + 0.5, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, 30)
fig.tight_layout()
plt.savefig("paper/figures/classifier_latency.png", dpi=300)
plt.close()

# GRAPH 3: Primary Benchmark Total Latency
baselines = ['LLM-only', 'Fixed RAG', 'Always-Compress', 'AdaptiveRAG']
# Extracted from refer_this.pdf / main.tex
total_lats = [11.877, 18.472, 19.510, 16.514]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(baselines, total_lats, color='#2ca02c')
ax.set_ylabel('Total Latency (seconds)')
ax.set_title('Primary Benchmark End-to-End Latency')
for i, v in enumerate(total_lats):
    ax.text(i, v + 0.5, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, 25)
fig.tight_layout()
plt.savefig("paper/figures/benchmark_latency.png", dpi=300)
plt.close()

# GRAPH 4: Compression token reduction
conditions = ['Always-Compress', 'AdaptiveRAG']
orig_tokens = [1234.0, 1244.6]
comp_tokens = [600.5, 605.0]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(5, 4))
rects1 = ax.bar(x - width/2, orig_tokens, width, label='Original Tokens', color='#d62728')
rects2 = ax.bar(x + width/2, comp_tokens, width, label='Compressed Tokens', color='#9467bd')

ax.set_ylabel('Average Tokens')
ax.set_title('Context Token Reduction')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend()
ax.set_ylim(0, 1500)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("paper/figures/token_reduction.png", dpi=300)
plt.close()

print("Graphs created successfully.")
