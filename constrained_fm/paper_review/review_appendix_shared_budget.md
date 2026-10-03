# Review — Appendix: Shared-Budget Sample Efficiency

**Verdict:** every number in the table matches `tables/shared_budget.tex` (run `shared_budget_v1k-d0c34cd1`). The section's problems are in the setup and interpretation: it leaves out details a reviewer needs to judge whether the comparison is fair, and one sentence misdescribes the baseline.

## Must fix

- [ ] **1. "Without forcing the generative flow to relearn the target density from scratch" misdescribes FS-FT.**
  - FS-FT *fine-tunes the base unconditional model*, which already knows $p_{\text{data}}$. It does not learn from scratch. The few-shot model that *does* train from scratch is a different baseline that no longer appears in the paper.
  - Fix: the real difference is that FS-FT must adapt *network weights* from $N_{\text{in}}$ positive examples, while the amortized model only fits a 512-d latent through a shared, meta-learned encoder.

- [ ] **2. The two methods receive different information per point. State it.**
  - Implicit: all $N$ points, each labelled with the *continuous constraint value* $\tanh C_\phi(x)$. That includes the $N-N_{\text{in}}$ points outside the constraint.
  - FS-FT: only the $N_{\text{in}}$ inside points, with no labels.
  - "Strictly shared budget" is true for *draws*, not for *information*. One sentence acknowledging this protects against a "the comparison is unfair" review.

- [ ] **3. Explain why the implicit model plateaus, and why it is worse than in the main table.**
  - Here the queries are the $N$ GMM draws, *clamped* to the domain. The SIREN was meta-trained on **uniform** queries, so these queries are off-distribution for it.
  - This is why implicit SWD only falls from 0.114 to 0.085. With 1000 uniform queries in the main table it reaches 0.0706, which beats its own $N=2000$ result here.
  - Without this sentence, a reader comparing the main table with this appendix will see a contradiction.
  - (The repo's table caption already says "the SIREN was meta-trained on uniformly distributed query points". The paper caption dropped it.)

- [ ] **4. State the aggregation filter.** The medians are over **917 of 1000** constraints, those with at least 5 inside points at $N=50$, and the same subset is used for both methods. FS-FT is undefined when a constraint has no inside points. The repo caption says this; the paper caption dropped it.

## Should fix

- [ ] **5. The crossover claim is slightly off.**
  - "Parity at approximately $N\ge1000$" understates what happens.
  - At $N=1000$ the two tie on SWD and KLD, and FS-FT is already ahead on AR (98.06 vs 97.61).
  - At $N=2000$, FS-FT is ahead on AR, SWD and KLD.
  - Suggested wording: "FS-FT reaches parity at $N\approx1000$ ($N_{\text{in}}\approx540$) and is slightly ahead at $N=2000$".
- [ ] **6. Naming.** The table headers say "Functa" and the figure legend says "Functa (ours)". Change both to "Implicit" to match the main text. That needs a table regeneration (the `LABELS` in `shared_budget_results.py` / `table_shared_budget.py`) and a figure re-render.
- [ ] **7. "Zero-shot generalization"** → "amortized"; the implicit model fits its latent from the same $N$ points.
- [ ] **8. Notation.** $P(x)\le0$ → $C_\phi(x)\le0$.
- [ ] **9. Captions.** The table and figure captions are identical. Keep the full protocol in the table caption only: nested budgets, 917/1000 constraints, medians, uniform meta-training, the early-stopping set. Give the figure caption one line: "Median and interquartile range over constraints."
- [ ] **10. Wording.** Remove "rigorously", "strictly", "massive", "severe", "robust high-fidelity" and "reliably captures".

## Optional
- **The overlap is wide.** From $N=300$ upward the implicit IQR band (AR, SWD) overlaps FS-FT's. A paired win rate per constraint would be stronger evidence than medians alone.
- **Layout.** If the appendix stays two-column, `figure` with `width=\textwidth` and a 10-column `table` will overflow. Use `figure*`/`table*` (as in the repo table), or switch the appendix to `\onecolumn`.
- **Missing FS-FT detail.** Add one clause: lr $10^{-4}$, patience 12. The current text gives only the early-stopping set.

## Suggested replacement for paragraph 2
```latex
Table~\ref{tab:shared-budget} and Figure~\ref{fig:shared-budget-curves} show that the amortized model is markedly more sample-efficient at small budgets. At $N=50$, where FS-FT adapts on a median of $27$ feasible points, the implicit model attains $3\times$ lower SWD, $13\times$ lower MMD and $4.6\times$ lower KLD. Its advantage comes from adapting only a latent code through a meta-learned encoder, rather than the weights of the flow. FS-FT reaches parity at $N\approx1000$ ($N_{\mathrm{in}}\approx540$) and is slightly ahead at $N=2000$. The implicit model saturates early because the queries here follow the target distribution, whereas its encoder was meta-trained on uniform queries; with $1000$ uniform queries it reaches an SWD of $0.071$ (Table~\ref{tab:val1k-main}).
```
