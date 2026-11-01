"""
Generate publication-quality Gantt chart and 3x3 Risk Matrix for Assignment 2.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np

# Set font and style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def generate_gantt_chart():
    tasks = [
        {"id": "T1", "name": "1. Repository Audit & Git Isolation (se_assignment)", "start": "2026-10-14", "end": "2026-10-18", "owner": "Alvin", "color": "#4A90E2"},
        {"id": "T2", "name": "2. Streamlit UI Input Validation & Odds Formatters", "start": "2026-10-18", "end": "2026-10-24", "owner": "Alvin", "color": "#4A90E2"},
        {"id": "T3", "name": "3. Causal Feature Selector & Safety Filter Review", "start": "2026-10-22", "end": "2026-10-29", "owner": "Adithyan", "color": "#50E3C2"},
        {"id": "T4", "name": "4. Deterministic PyTest Unit Test Suite (BVA & EP)", "start": "2026-10-26", "end": "2026-11-01", "owner": "Alvin", "color": "#4A90E2"},
        {"id": "T5", "name": "5. CounterFact Benchmark Validation & Metrics", "start": "2026-10-28", "end": "2026-11-04", "owner": "Alona", "color": "#F5A623"},
        {"id": "T6", "name": "6. GitHub Actions Multi-Version CI/CD Pipeline", "start": "2026-10-30", "end": "2026-11-03", "owner": "Alvin", "color": "#4A90E2"},
        {"id": "T7", "name": "7. Cross-Model Ridge Translator & Cloud Deployment Spec", "start": "2026-11-02", "end": "2026-11-08", "owner": "Darsan", "color": "#BD10E0"},
        {"id": "T8", "name": "8. PR Peer Review, Merge & Assignment Report Finalization", "start": "2026-11-05", "end": "2026-11-12", "owner": "All Team", "color": "#9013FE"},
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    y_pos = np.arange(len(tasks))
    start_dates = [datetime.strptime(t["start"], "%Y-%m-%d") for t in tasks]
    end_dates = [datetime.strptime(t["end"], "%Y-%m-%d") for t in tasks]
    durations = [(end - start).days for start, end in zip(start_dates, end_dates)]

    # Plot bars
    for i, t in enumerate(tasks):
        start = mdates.date2num(start_dates[i])
        dur = durations[i]
        bar = ax.barh(y_pos[i], dur, left=start, height=0.55, align='center',
                      color=t["color"], edgecolor='#333333', linewidth=0.8, alpha=0.9)
        # Add owner text inside or next to the bar
        ax.text(start + dur / 2, y_pos[i], f"{t['owner']} ({dur}d)",
                ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t["name"] for t in tasks], fontsize=9.5, fontweight='medium')
    ax.invert_yaxis()  # top-down

    # Format date axis
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=2))

    # Reference today line
    today = datetime(2026, 11, 1)
    ax.axvline(mdates.date2num(today), color='#D0021B', linestyle='--', linewidth=1.5, label='Project Milestone (Today: 1 Nov)')

    ax.set_title("FeatureScalpel: Project Work Breakdown and Schedule (14 Oct – 15 Nov 2026)",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Timeline", fontsize=10, labelpad=8)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='lower right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("docs/assignment-2/evidence/gantt_chart.png", dpi=300)
    plt.close()
    print("Saved docs/assignment-2/evidence/gantt_chart.png")

def generate_risk_matrix():
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    # 3x3 matrix colors
    # Impact (rows from bottom: Low=0, Medium=1, High=2)
    # Likelihood (cols from left: Low=0, Medium=1, High=2)
    colors = [
        ["#C8E6C9", "#FFF59D", "#FFE082"],  # Low Impact: Green, Light Yellow, Amber
        ["#C8E6C9", "#FFF59D", "#FFAB91"],  # Med Impact: Green, Yellow, Light Red
        ["#FFF59D", "#FFAB91", "#FF8A80"]   # High Impact: Yellow, Light Red, Red
    ]

    for r in range(3):
        for c in range(3):
            rect = patches.Rectangle((c, r), 1, 1, facecolor=colors[r][c], edgecolor='#CCCCCC', linewidth=1.5)
            ax.add_patch(rect)
            # Add risk score
            score = (r + 1) * (c + 1)
            ax.text(c + 0.92, r + 0.08, str(score), ha='right', va='bottom', color='#777777', fontsize=9, fontweight='bold')

    # Add risks
    # R1: Out-of-Memory (High Likelihood=2, High Impact=2) -> cell (2, 2)
    # R2: Dependency Drift (Medium Likelihood=1, High Impact=2) -> cell (1, 2)
    # R3: Non-deterministic logit variance (Medium Likelihood=1, Medium Impact=1) -> cell (1, 1)
    risks = [
        {"id": "R1", "x": 2.5, "y": 2.5, "title": "R1: GPU VRAM OOM\n(Score: 9, High/High)"},
        {"id": "R2", "x": 1.5, "y": 2.5, "title": "R2: Library API Drift\n(Score: 6, Med/High)"},
        {"id": "R3", "x": 1.5, "y": 1.5, "title": "R3: Hardware Variance\n(Score: 4, Med/Med)"},
    ]

    for rk in risks:
        ax.plot(rk["x"], rk["y"], marker='o', markersize=14, color='#D0021B', markeredgecolor='black', markeredgewidth=1.2)
        ax.text(rk["x"], rk["y"] - 0.22, rk["title"], ha='center', va='top', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Low", "Medium", "High"], fontsize=11, fontweight='bold')
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Low", "Medium", "High"], fontsize=11, fontweight='bold')

    ax.set_xlabel("Likelihood / Probability", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Impact / Severity", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title("Risk Matrix: FeatureScalpel Engineering & Deployment (Score = Likelihood × Impact)",
                 fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig("docs/assignment-2/evidence/risk_matrix.png", dpi=300)
    plt.close()
    print("Saved docs/assignment-2/evidence/risk_matrix.png")

if __name__ == "__main__":
    generate_gantt_chart()
    generate_risk_matrix()
