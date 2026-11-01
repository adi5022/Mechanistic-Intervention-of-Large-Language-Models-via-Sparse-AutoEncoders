# FeatureScalpel — Software Engineering Assignment 2
## Team Execution Guide & Submission Instructions

**Repository:** [adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders](https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders)  
**Base Integration Branch:** `se_assignment`  
**Max Marks:** 50 (Q1: 10, Q2: 10, Q3: 15, Q4: 15)

---

### Team Contribution Mapping

| Member | Roll No | KTU Reg. | GitHub Username | Assigned Task / Issue | Feature Branch | Reviewer | Status |
|---|---|---|---|---|---|---|---|
| **Alvin Binu** | B23CS2114 | MBT23CA014 | `Alvinbinu` | Issue #3: Automated Unit Test Suite (14 TCs with BVA/EP) & CI Pipeline | `feature/alvin-app-testing` | `adi5022` | **Merged & Verified** (CI Green) |
| **Adithyan R S** | B23CS2109 | MBT23CA009 | `adi5022` | Issue #1: Causal Feature Selector & Layer 8 Ablation Stability | `feature/causal-intervention` | `Alvinbinu` | Pending Teammate Execution |
| **Alona Biju** | B23CS2113 | MBT23CA013 | `Alonabiju` | Issue #2: CounterFact Benchmark Verification & Reproducibility | `feature/benchmark-eval` | `DarsanM` | Pending Teammate Execution |
| **Darsan M** | B23CS2125 | MBT23CA025 | `DarsanM` | Issue #4: Cross-Model Ridge Translator & Deployment Specification | `feature/cross-model-deploy` | `Alonabiju` | Pending Teammate Execution |

---

### Instructions for Teammates to Complete Their Genuine Contributions

Each member must execute their own genuine commits from their authenticated GitHub account. Never commit under another person's name.

#### 1. Adithyan R S (`adi5022`)
```bash
git checkout se_assignment
git pull origin se_assignment
git checkout -b feature/causal-intervention

# Commit 1: Causal ranking validation
git add src/editing.py
git commit -m "Validate causal feature ranking and batch dimension candidate ablation"

# Commit 2: Hook safety checks
git add src/hooks.py
git commit -m "Ensure non-negative probability delta assertions in signed ablation hooks"

git push -u origin feature/causal-intervention
```
* Then open PR #2 targeting `se_assignment`, assign `Alvinbinu` as reviewer. Once reviewed and approved, merge PR.

#### 2. Alona Biju (`Alonabiju`)
```bash
git checkout se_assignment
git pull origin se_assignment
git checkout -b feature/benchmark-eval

# Commit 1: CounterFact verification
git add tools/build_counterfact_set.py
git commit -m "Verify CounterFact hard set split consistency and relation holdout"

# Commit 2: Benchmark assertions
git add src/benchmark/
git commit -m "Add integrity checks for candidate source evaluation tables"

git push -u origin feature/benchmark-eval
```
* Then open PR #3 targeting `se_assignment`, assign `DarsanM` as reviewer. Once reviewed and approved, merge PR.

#### 3. Darsan M (`DarsanM`)
```bash
git checkout se_assignment
git pull origin se_assignment
git checkout -b feature/cross-model-deploy

# Commit 1: Translator validation
git add src/transfer/
git commit -m "Add unit tests for ridge regression translation matrix dimensions"

# Commit 2: Deployment documentation
git add docs/deployment/
git commit -m "Add Dockerfile specification and IaaS deployment runbook for FeatureScalpel"

git push -u origin feature/cross-model-deploy
```
* Then open PR #4 targeting `se_assignment`, assign `Alonabiju` as reviewer. Once reviewed and approved, merge PR.

---

### PDF Report Filenames for Submission
- Alvin Binu: `A2_B23CS2114_AlvinBinu.pdf` (Ready)
- Adithyan R S: `A2_B23CS2109_Adithyan_R_S.pdf` (Generated from template)
- Alona Biju: `A2_B23CS2113_Alona_Biju.pdf` (Generated from template)
- Darsan M: `A2_B23CS2125_Darsan_M.pdf` (Generated from template)
