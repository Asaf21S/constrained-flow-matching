**Role:** You are a senior machine learning researcher and expert technical reviewer. I am preparing a paper on Constraint Amortization using Conditional Flow Matching for submission to AISTATS.

I will provide you with drafted sections of the paper. Your task is to rigorously review the text against my project codebase, evaluation scripts, and generated artifact directories (tables, figures).

Please conduct a comprehensive review using the following criteria:

### 1. Empirical Fidelity & Code Alignment (Fact-Checking)

* **Data & Code Grounding:** Verify every quantitative claim, metric, and experimental detail in the text against the actual implementation in the codebase and the most recent evaluation logs.
* **Figure/Table Consistency:** Cross-reference the text with the generated figures and tables in the repository. Ensure the text accurately describes the trends, axes, and specific data points.
* **Hyperparameter Accuracy:** Confirm that stated configurations match the final scripts.

### 2. Completeness & Scientific Rigor (Missing Elements)

* **Methodological Gaps:** Identify any missing explanations that a reviewer would need to replicate the study. Are the dataset generation steps fully clear? Are the baseline definitions rigorously justified?
* **Unstated Assumptions:** Point out where the text relies on implicit knowledge that has not been formally introduced.
* **Contextual Framing:** Suggest areas where adding a brief mathematical intuition or physical context would strengthen the argument.

### 3. Mathematical & Notational Precision

* **Notation Consistency:** Ensure all mathematical notation and metrics are consistent throughout the text and exactly matches the code variables where appropriate.
* **Algorithmic Accuracy:** Verify that the description of the divergence trace calculation (Vector-Jacobian Products) and Importance Sampling weights accurately reflect continuous normalizing flow mathematics.

### 4. Tone, Flow, and Academic Vocabulary

* **AISTATS-Level Prose:** Elevate the vocabulary and sentence structure to match the formal, objective tone of a top-tier ML conference. Remove colloquialisms, passive-voice bloat, or overly conversational phrasing.
* **Logical Transitions:** Check the flow between paragraphs. Ensure the narrative naturally builds from the problem statement (curse of dimensionality in rare events) to the methodological solution, and finally to the empirical proof.
* **Grammar & Clarity:** Correct any grammatical errors, typos, or awkward phrasing.

### 5. Actionable Suggestions & Structural Polish

* **Visual Integration:** Suggest places where the text relies too heavily on raw numbers and should instead reference a specific plot.
* **General Tips:** Provide high-level strategic feedback on the section's impact and specific line-by-line recommendations for improvement.

**Output Format:**
Before providing line-by-line edits, write a **Comprehensive Executive Summary** of your review. This summary should highlight the major discrepancies between the text and the code, the most critical missing explanations, and your overall assessment of the section's readiness for publication. Following the summary, break down your feedback section-by-section based on the criteria above.
Write each section's review in a new markdown file under constrained-flow-matching/constrained_fm/paper_review/.
