import os
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # 1. Locate the latest benchmark JSON file
    json_files = sorted(glob.glob("benchmark_results/layer_benchmark_*.json"))
    if not json_files:
        print("No benchmark JSON files found in benchmark_results/")
        return
        
    latest_json = json_files[-1]
    print(f"Loading benchmark data from: {latest_json}")
    
    with open(latest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = [item["layer"] for item in data["layers"]]
    prob_gains = [item["probability_gain"] * 100 for item in data["layers"]]
    rank_improvements = [item["rank_improvement"] for item in data["layers"]]
    runtimes = [item["runtime_ms"] / 1000 for item in data["layers"]] # convert to seconds

    # Ensure output directory for journal images exists
    img_dir = "docs/Research_Journal/images"
    os.makedirs(img_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Chart 1: Probability Gain vs Layer
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar([f"Layer {l}" for l in layers], prob_gains, color="#4C72B0", width=0.5)
    ax.set_title("Target Probability Gain (%) vs Transformer Layer Depth", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Probability Gain (%)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"+{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    prob_chart_path = os.path.join(img_dir, "probability_gain_vs_layer.png")
    plt.savefig(prob_chart_path, dpi=300)
    plt.close()

    # Chart 2: Rank Improvement vs Layer
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar([f"Layer {l}" for l in layers], rank_improvements, color="#55A868", width=0.5)
    ax.set_title("Target Token Rank Improvement vs Transformer Layer Depth", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Rank Improvement (Positions Cleared)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"+{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    rank_chart_path = os.path.join(img_dir, "rank_improvement_vs_layer.png")
    plt.savefig(rank_chart_path, dpi=300)
    plt.close()

    # Chart 3: Runtime vs Layer
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar([f"Layer {l}" for l in layers], runtimes, color="#C44E52", width=0.5)
    ax.set_title("Benchmark Execution Runtime (seconds) vs Layer", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    runtime_chart_path = os.path.join(img_dir, "runtime_vs_layer.png")
    plt.savefig(runtime_chart_path, dpi=300)
    plt.close()

    print(f"Generated benchmark charts in {img_dir}:")
    print(f" - {prob_chart_path}")
    print(f" - {rank_chart_path}")
    print(f" - {runtime_chart_path}")

if __name__ == "__main__":
    main()
