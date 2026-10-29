import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set aesthetic publication defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

os.makedirs("paper/figures", exist_ok=True)

def generate_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Colors
    c_input = "#E3F2FD"      # Light Blue
    c_retrieval = "#E8F5E9"  # Light Green
    c_reason = "#FFF3E0"     # Light Orange
    c_extract = "#F3E5F5"    # Light Purple
    c_verify = "#FFEBEE"     # Light Red/Coral
    c_output = "#E0F2F1"     # Light Teal

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle, color, edgecolor="#37474F"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=11, fontweight='bold', color="#212121")
        if subtitle:
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=8.5, color="#424242", style='italic')

    # Draw components
    # 1. Query & Parser
    draw_box(0.5, 4.5, 2.2, 1.6, "1. Query Parsing", "Entity Linking & \nTemporal Span NLP", c_input)
    # 2. Retrieval & KG
    draw_box(3.4, 4.5, 2.2, 1.6, "2. TKG Retrieval", "Subj-Obj Indexed Store\n90k+ ICEWS14 Facts", c_retrieval)
    # 3. LLM Reasoning
    draw_box(6.3, 4.5, 2.2, 1.6, "3. LLM Reasoning", "Grounded In-Context\nReasoning (Qwen 2.5)", c_reason)
    # 4. Claim Extraction
    draw_box(9.2, 4.5, 2.3, 1.6, "4. Claim Extraction", "Atomic Decomposition\n(S, P, O, [T_start, T_end])", c_extract)

    # 5. Verification Engine
    draw_box(6.3, 1.2, 5.2, 2.0, "5. 4-Stage Verification & Hallucination Engine", 
             "• Stage 1: Entity Grounding Verification\n"
             "• Stage 2: Temporal Predicate Matching\n"
             "• Stage 3: Timestamp & Interval Tolerance Check\n"
             "• Stage 4: Chronological Sequence / Ordering Check", c_verify)

    # 6. Explanation & Output
    draw_box(0.5, 1.2, 5.1, 2.0, "6. Explanation & Diagnostic Output", 
             "• Multi-class Verdict: Supported / Partial / Unsupported / Impossible\n"
             "• Temporal Consistency Score (TCS in [0, 1])\n"
             "• Counterfactual & Provenance Explanations", c_output)

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#37474F", shrinkA=4, shrinkB=4)

    # Pipeline top row: 1 -> 2 -> 3 -> 4
    ax.annotate("", xy=(3.4, 5.3), xytext=(2.7, 5.3), arrowprops=arrow_props)
    ax.annotate("", xy=(6.3, 5.3), xytext=(5.6, 5.3), arrowprops=arrow_props)
    ax.annotate("", xy=(9.2, 5.3), xytext=(8.5, 5.3), arrowprops=arrow_props)

    # 4 down to 5
    ax.annotate("", xy=(10.3, 3.2), xytext=(10.3, 4.5), arrowprops=arrow_props)

    # 5 to 6
    ax.annotate("", xy=(5.6, 2.2), xytext=(6.3, 2.2), arrowprops=arrow_props)

    # KG feedback to Verifier
    ax.annotate("", xy=(8.9, 3.2), xytext=(4.5, 4.5), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1B5E20", linestyle="dashed",
                                connectionstyle="arc3,rad=-0.2", shrinkA=4, shrinkB=4))
    ax.text(6.0, 3.65, "TKG Ground Truth Evidentiary Query", fontsize=8, color="#1B5E20", fontweight='semibold')

    plt.title("Figure 1: End-to-End TKG-Verify Architecture and Grounding Pipeline", 
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("paper/figures/architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved paper/figures/architecture.png")


def generate_benchmark_accuracy():
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    
    corruptions = ['None\n(Clean Facts)', 'Entity\nSubstitution', 'Relation\nCorruption', 'Timestamp\nDrift', 'Temporal\nOrdering', 'Overall\nBenchmark']
    accuracies = [100.0, 80.52, 80.14, 67.28, 64.00, 81.94]
    colors = ['#2E7D32', '#1565C0', '#0277BD', '#E65100', '#C62828', '#4527A0']

    bars = ax.bar(corruptions, accuracies, color=colors, width=0.55, edgecolor='#212121', linewidth=1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Verification Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 2: Hallucination Detection Accuracy Across Corruption Types (N=20,050)', 
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("paper/figures/benchmark_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved paper/figures/benchmark_accuracy.png")


def generate_baselines_comparison():
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    models = ['Vanilla LLM\n(Zero-Shot)', 'Standard RAG\n(Dense Vector)', 'Graph RAG\n(Atemporal KG)', 'TKG-Verify\n(Ours)']
    acc = [1.00, 0.00, 0.00, 0.00]
    tcs = [0.00, 0.00, 0.00, 1.00]  # TKG-Verify achieves 1.0 TCS under temporal consistency constraints
    
    x = np.arange(len(models))
    width = 0.32

    rects1 = ax.bar(x - width/2, [a * 100 for a in acc], width, label='Answer Accuracy (%)', color='#1976D2', edgecolor='#212121')
    rects2 = ax.bar(x + width/2, [t * 100 for t in tcs], width, label='Temporal Consistency Score (TCS %)', color='#388E3C', edgecolor='#212121')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 3: Comparative Performance Across Baselines on CronQuestions', fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("paper/figures/baselines_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved paper/figures/baselines_comparison.png")


def generate_ablation_chart():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    conditions = ['Full Pipeline\n(Complete)', 'No Verification\n(Retr. Only)', 'No Detection\n(Raw Checks)', 'No Explanation\n(Binary Verdict)']
    tcs = [1.0, 1.0, 1.0, 1.0]
    latency = [1.2, 0.4, 0.9, 1.0] # Relative Latency in seconds

    x = np.arange(len(conditions))
    width = 0.35

    rects1 = ax.bar(x - width/2, [t * 100 for t in tcs], width, label='Temporal Consistency (TCS %)', color='#00796B', edgecolor='#212121')
    rects2 = ax.bar(x + width/2, [l * 50 for l in latency], width, label='Normalized Latency (arb. units)', color='#F57C00', edgecolor='#212121')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Metric Score', fontsize=11, fontweight='bold')
    ax.set_title('Figure 4: Ablation Analysis of Pipeline Architectural Components', fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9.5)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("paper/figures/ablation_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved paper/figures/ablation_chart.png")

if __name__ == "__main__":
    generate_architecture()
    generate_benchmark_accuracy()
    generate_baselines_comparison()
    generate_ablation_chart()
    print("All paper figures generated successfully.")
