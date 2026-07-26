# Research Contribution Guidelines

To maintain scientific integrity and prevent unstructured code changes, all future modifications to FeatureScalpel must follow a structured research workflow.

## Research Workflow

Every modification or algorithmic addition must follow this sequence strictly:

```
  [Motivation]
       ↓
  [Hypothesis]
       ↓
 [Implementation] (No code changes before formulating the hypothesis)
       ↓
  [Experiment]
       ↓
  [Observation]
       ↓
[Interpretation]
       ↓
[Research Notebook] (Logged chronologically in research/notebook/)
       ↓
    [Paper] (Updated in research/paper/)
```

**Do NOT go directly from Implementation to Paper.**

## Writing Guidelines
1. **Never fabricate missing context**: If details of an experiment or pivot are unknown, mark it as `Pending reconstruction from Claude conversation history`.
2. **Commit to the Notebook first**: Log raw failures, screenshots, and token probability metrics immediately in the chronological notebook.
3. **Formalize in the Paper**: Keep the paper sections clean, topic-focused, and free of raw development log formatting.
