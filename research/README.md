# FeatureScalpel Research Directory

This directory houses the scientific infrastructure for the FeatureScalpel research project, focusing on transient activation steering in Large Language Models using Sparse Autoencoders (SAEs) for factual correction.

## Purpose of this Directory

We transition from an implementation-only repository to a structured research repository. This directory isolates our theoretical formulations, experimental logs, and paper manuscripts from the execution code.

## Directory Structure

* **`research/timeline.md`**: Chronological trace of milestones, pivots, and results.
* **`research/hypothesis_log.md`**: Structured record of hypotheses tested, their outcomes, and evidence.
* **`research/experiment_index.md`**: Master list of all experiments.
* **`research/notebook/`**: Chronological lab notebooks detailing specific experiments (use `experiment_template.md` to create new entries).
* **`research/paper/`**: Modular sections of our working manuscript organized by topic.
* **`research/figures/`**: Visual assets, plots, and architecture diagrams.

## Distinction of Concepts

| Category | Scope & Purpose | Target Audience | Chronology |
| :--- | :--- | :--- | :--- |
| **Implementation** | Source code (`src/`), Streamlit UI (`experiment_app.py`), hooks, and utilities. | Engineers & Developers | Version-controlled / Git |
| **Engineering Docs** | System configuration, UI layouts, and walkthroughs. | Users of the tool | Sequential |
| **Experiment Notebook** | Detailed logs, raw inputs/outputs, failures, and diagnostics. | Researchers replicating results | Chronological |
| **Research Paper** | Abstracted, formal presentation of methodology, results, and discussion. | Academic community | Topical / Structured |

---
*Note: Some historical contexts, design discussions, and rejected alternatives are currently **pending reconstruction from Claude conversation history**.*
