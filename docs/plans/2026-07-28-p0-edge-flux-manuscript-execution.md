# P0 Conservative Edge-Flux Manuscript Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a self-contained Journal of Computational Physics-style manuscript on lowest-order conservative edge-flux remapping for nonmatching planar and spherical polygonal meshes, with a separate reproducibility companion and a compact planar preliminary high-order outlook.

**Architecture:** Create a new `elsarticle` manuscript rather than modifying either existing report. Build an internal evidence ledger first, generate P0-only publication figures from verified numerical data, draft each mathematical section against the approved scope, and keep all software-specific provenance in `REPRODUCIBILITY.md`. Quantitative high-order outlook claims are admitted only after an apples-to-apples planar P0-versus-patch-recovery experiment is available.

**Tech Stack:** LaTeX (`elsarticle`, BibTeX), C++14/MOAB/Eigen numerical executables, Python 3/Matplotlib for figure generation, Markdown for reproducibility and evidence traceability.

---

## Execution Rules

- Execute in an isolated worktree using `superpowers:using-git-worktrees`.
- Preserve `docs/mimetic_voronoi_report.tex` and `docs/voronoi_p0_report.tex` byte-for-byte.
- Do not mention C++, Python, MOAB, Eigen, source files, commands, tests, repositories, or generated artifacts in the manuscript.
- Do not include RLL round-trip results or analysis.
- Do not call the higher-order CSV `order=1` rows “P0” without first demonstrating mathematical equivalence to the lowest-order reconstruction used in this paper.
- A first-order claim requires at least three comparable refinement levels, a stated mesh-size proxy, and a defined norm.
- Every numerical value in the manuscript must appear in the evidence ledger when drafted and in the reproducibility companion before final acceptance.
- Run the manuscript hygiene check and a clean LaTeX build after every manuscript task.
- Use one atomic commit per task unless a task explicitly groups an inseparable figure/data pair.

## Planned Deliverables

- Create: `docs/p0_edge_flux_remap_jcp.tex`
- Create: `docs/references_p0.bib`
- Create: `docs/REPRODUCIBILITY.md`
- Create: `docs/p0_edge_flux_evidence.md`
- Create: `docs/p0_manuscript_review.md`
- Create: `docs/data/p0_edge_flux/raw/planar.txt`
- Create: `docs/data/p0_edge_flux/raw/spherical_quad.txt`
- Create: `docs/data/p0_edge_flux/raw/spherical_voronoi.txt`
- Create: `docs/p0_edge_flux_convergence.csv`
- Create: `docs/p0_edge_flux_spherical.csv`
- Create: `docs/p0_patch_extension.csv`
- Create: `docs/make_p0_edge_flux_figures.py`
- Create: `docs/figures/p0_edge_flux/planar_convergence.pdf`
- Conditionally create: `docs/figures/p0_edge_flux/spherical_results.pdf` only if a meaningful comparable spherical sequence exists; otherwise use a manuscript table generated from `docs/p0_edge_flux_spherical.csv`.
- Conditionally create: `docs/figures/p0_edge_flux/planar_patch_extension.pdf`
- Create: `scripts/check_p0_manuscript_hygiene.sh`
- Create: `scripts/build_p0_manuscript.sh`
- Optionally create or modify one planar numerical driver only if required to produce an apples-to-apples P0-versus-patch-recovery comparison.

## Task 1: Establish Baselines And Protect Existing Manuscripts

**Files:**
- Read: `docs/mimetic_voronoi_report.tex`
- Read: `docs/voronoi_p0_report.tex`
- Create: `/tmp/p0-manuscript-baseline.sha256`

**Step 1: Record immutable baselines**

Run:

```bash
shasum -a 256 docs/mimetic_voronoi_report.tex docs/voronoi_p0_report.tex \
  > /tmp/p0-manuscript-baseline.sha256
```

Expected: two SHA-256 records.

**Step 2: Verify the current numerical suite**

Run:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Expected: build exits 0 and all current tests pass. Record any pre-existing failure before proceeding.

**Step 3: Capture the exact current revision and environment**

Run:

```bash
git rev-parse HEAD
cmake --version
c++ --version
```

Expected: non-empty revision and tool versions for later entry in `REPRODUCIBILITY.md`.

**Step 4: Do not commit**

This task creates only a temporary baseline file.

## Task 2: Add Manuscript Hygiene Gate

**Files:**
- Create: `scripts/check_p0_manuscript_hygiene.sh`
- Create: `scripts/build_p0_manuscript.sh`

**Step 1: Write the failing gate before the manuscript exists**

The script must:

- accept an optional manuscript path, defaulting to `docs/p0_edge_flux_remap_jcp.tex`;
- fail if the file does not exist;
- remove LaTeX comments by stripping from the first unescaped `%` through end
  of line, while preserving escaped `\%`;
- scan only the manuscript `.tex` source, not the bibliography or generated
  `.bbl` file;
- use case-insensitive fixed-string matching, not regular expressions, for
  these exact forbidden tokens:
  `C++`, `Python`, `MOAB`, `Eigen`, `.cpp`, `.py`, `cmake`, `ctest`, `build/`,
  `repository`, `artifact`, `RLL`, `round-trip`, or `roundtrip`;
- exit 0 with `P0 manuscript hygiene: PASS` otherwise.

**Step 2: Run to verify the initial failure**

Run:

```bash
bash scripts/check_p0_manuscript_hygiene.sh
```

Expected: non-zero exit because the new manuscript does not exist yet.

**Step 3: Shell-check behavior with a temporary clean file**

Run:

```bash
tmp=$(mktemp)
printf '\\documentclass{article}\n' > "$tmp"
bash scripts/check_p0_manuscript_hygiene.sh "$tmp"
rm "$tmp"
```

Expected: `P0 manuscript hygiene: PASS`.

**Step 4: Add the standardized manuscript build gate**

Create `scripts/build_p0_manuscript.sh`. It must run from the repository root,
remove and recreate `/tmp/p0-jcp-build`, invoke:

```bash
(cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/p0-jcp-build p0_edge_flux_remap_jcp.tex)
```

and fail if the final log contains `undefined references`, `undefined
citations`, `Citation .* undefined`, or `Reference .* undefined`. Overfull and
underfull box warnings are permitted but must be printed for review. All other
LaTeX errors are blocking through `-halt-on-error`.

Before the manuscript exists, verify that this script fails with a clear
missing-file message.

**Step 5: Commit**

```bash
git add scripts/check_p0_manuscript_hygiene.sh scripts/build_p0_manuscript.sh
git commit -m "test: add P0 manuscript validation gates"
```

## Task 3: Build The Evidence Ledger Before Drafting Claims

**Files:**
- Create: `docs/p0_edge_flux_evidence.md`
- Create: `docs/REPRODUCIBILITY.md`
- Create: `docs/data/p0_edge_flux/raw/planar.txt`
- Create: `docs/data/p0_edge_flux/raw/spherical_quad.txt`
- Create: `docs/data/p0_edge_flux/raw/spherical_voronoi.txt`
- Read: `tests/convergence_validation_test.cpp`
- Read: `tests/spherical_quad_test.cpp`
- Read: `tests/spherical_voronoi_test.cpp`
- Read: `tests/patch_test.cpp`
- Read: `tests/conservative_intersection_test.cpp`
- Read: `tests/voronoi_intersection_test.cpp`
- Read: `tests/hdiv_conforming_projection_test.cpp`
- Read: `docs/figures/voronoi_p0/metrics_summary.csv`

**Step 1: Run the lowest-order planar studies and save raw output**

Run:

```bash
mkdir -p docs/data/p0_edge_flux/raw
./build/convergence_validation_test | tee docs/data/p0_edge_flux/raw/planar.txt
./build/patch_test | tee docs/data/p0_edge_flux/raw/patch.txt
./build/conservative_intersection_test | tee docs/data/p0_edge_flux/raw/quad_overlap.txt
./build/voronoi_intersection_test | tee docs/data/p0_edge_flux/raw/voronoi_overlap.txt
./build/hdiv_conforming_projection_test | tee docs/data/p0_edge_flux/raw/conforming.txt
```

Expected: each exits 0.

**Step 2: Run the lowest-order spherical studies and save raw output**

Run:

```bash
./build/spherical_quad_test | tee docs/data/p0_edge_flux/raw/spherical_quad.txt
./build/spherical_voronoi_test | tee docs/data/p0_edge_flux/raw/spherical_voronoi.txt
```

Expected: each exits 0.

**Step 3: Write the evidence ledger**

For every intended manuscript claim, record:

- mathematical claim wording;
- geometry and mesh family;
- field and exact divergence;
- raw or conforming output;
- error norm;
- mesh-size definition;
- refinement levels;
- measured errors and fitted/consecutive rates;
- conservation and compatibility residuals;
- primary internal evidence location;
- whether the claim is suitable for abstract, results, discussion, or only the reproducibility companion.

The ledger must explicitly classify:

- planar quad-to-quad sequences as observed higher-than-first-order behavior;
- planar Voronoi-to-Voronoi sequences by field, noting oscillatory consecutive rates;
- planar Voronoi-to-quad cases that are not suitable for a universal order claim;
- spherical two-level comparisons as error reduction only;
- any spherical three-level clean sequence that supports a fitted rate;
- the independent Halton-Voronoi study as corroborative rather than principal evidence.

Use this operational classification rubric:

- comparable sequence: identical mesh family, field, raw/conforming choice,
  norm, quadrature definition, and mesh-size proxy at every level;
- fitted-order eligible: at least three comparable levels with strictly
  decreasing `h`; report all consecutive rates and the least-squares slope of
  `log(error)` against `log(h)`;
- monotone-order claim: errors decrease at every level and the fitted slope is
  at least `0.8`; describe this as evidence consistent with at least first order;
- oscillatory sequence: any error increase or non-positive consecutive rate;
  retain every level in tables and make no blanket fitted-order claim;
- two-level sequence: report only the error-reduction factor and consecutive
  rate as a diagnostic, not an asymptotic order.

**Step 4: Audit terminology**

Add a short note resolving the project’s order nomenclature:

- “P0” in this manuscript means one integrated normal flux per edge and the
  lowest-order constant-divergence reconstruction;
- higher-order edge-moment order labels must not be silently equated with P0.

**Step 5: Create the reproducibility skeleton**

Create `docs/REPRODUCIBILITY.md` with the revision, environment placeholders,
raw-output paths, tolerance policy, and empty mapping tables for every planned
manuscript figure and table. Update this file whenever a later task introduces
or changes a quantitative result.

**Step 6: Commit**

```bash
git add docs/p0_edge_flux_evidence.md docs/REPRODUCIBILITY.md \
  docs/data/p0_edge_flux/raw
git commit -m "docs: establish P0 evidence and reproducibility ledger"
```

## Task 4: Export P0-Only Planar And Spherical Data

**Files:**
- Create: `docs/p0_edge_flux_convergence.csv`
- Create: `docs/p0_edge_flux_spherical.csv`
- Read: `docs/data/p0_edge_flux/raw/planar.txt`
- Read: `docs/data/p0_edge_flux/raw/spherical_quad.txt`
- Read: `docs/data/p0_edge_flux/raw/spherical_voronoi.txt`

**Step 1: Define the planar CSV schema**

Use these columns:

```text
mesh_family,field,level,h,l2_relative,linf,conservation,rate_l2,rate_linf
```

Populate only values from the actual lowest-order planar executable output.

**Step 2: Define the spherical CSV schema**

Use these columns:

```text
mesh_family,case,source_resolution,target_resolution,source_cells,target_cells,l2_relative_flux,linf_flux,global_conservation,direct_sparse_delta,reintegration_residual,conforming_cell_residual
```

Do not fabricate a mesh-size value or fitted order for non-comparable cases.

**Step 3: Verify numeric transcription**

Add an exact parsing/verification command to `docs/make_p0_edge_flux_figures.py`
in Task 7. Until then, manually cross-check every row and record the source-line
mapping in `docs/p0_edge_flux_evidence.md`. Task 7 must fail if a CSV value does
not match the persistent raw output to printed precision.

Expected: exact text-to-double agreement to the printed precision.

**Step 4: Commit the inseparable data pair**

```bash
git add docs/p0_edge_flux_convergence.csv docs/p0_edge_flux_spherical.csv \
  docs/p0_edge_flux_evidence.md docs/REPRODUCIBILITY.md
git commit -m "data: record P0 planar and spherical validation results"
```

Justification: both files are the P0 evidence dataset consumed by the same manuscript figure and table workflow.

## Task 5: Create A Focused Bibliography

**Files:**
- Create: `docs/references_p0.bib`
- Read: `docs/references.bib`

**Step 1: Copy and verify primary P0 references**

Include verified entries for:

- Perot and Chartrand’s level-2 mimetic reconstruction;
- Raviart and Thomas;
- Brezzi and Fortin;
- Arnold, Falk, and Winther;
- Bochev and Hyman;
- Lipnikov, Manzini, and Shashkov;
- Jones/SCRIP;
- Ullrich/Taylor and Ullrich/Devendran/Johansen conservative remapping;
- Sutherland and Hodgman clipping;
- authoritative gnomonic projection and Piola-transform sources already present in the bibliography.

Include future-work references only for:

- mixed virtual elements;
- polynomial or patch recovery;
- higher edge moments and high-order conservative remapping.

**Step 2: Verify bibliographic metadata**

Check authors, title, venue, year, volume, pages, DOI, and URL using this source
hierarchy: publisher or journal page, then DOI/Crossref record, then an
institutional archive. Do not introduce a citation solely from a secondary
survey. A missing DOI is acceptable only when the authoritative record has no
DOI; record the verification source in `REPRODUCIBILITY.md`. Resolve metadata
disagreements in favor of the publisher version of record.

**Step 3: Test BibTeX parsing**

Create `docs/p0-bib-check.tex` temporarily, citing every key, then run from the
repository root:

```bash
(cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error p0-bib-check.tex)
(cd docs && latexmk -C p0-bib-check.tex)
rm docs/p0-bib-check.tex
```

Expected: no missing-entry or undefined-citation warnings.

**Step 4: Commit**

```bash
git add docs/references_p0.bib
git commit -m "docs: add focused P0 remapping bibliography"
```

## Task 6: Scaffold A Compiling JCP Manuscript

**Files:**
- Create: `docs/p0_edge_flux_remap_jcp.tex`

**Step 1: Create the `elsarticle` skeleton**

Use:

```latex
\documentclass[preprint,12pt]{elsarticle}
\journal{Journal of Computational Physics}
```

Add only:

- title and placeholder author/affiliation fields;
- abstract placeholder;
- keywords;
- notation macros;
- the ten approved section headings;
- `\bibliographystyle{elsarticle-num}`;
- `\bibliography{references_p0}`.

Do not copy prose yet.

**Step 2: Run the hygiene gate**

```bash
bash scripts/check_p0_manuscript_hygiene.sh
```

Expected: PASS.

**Step 3: Compile from a clean temporary directory**

Run from the repository root:

```bash
bash scripts/build_p0_manuscript.sh
```

Expected: PDF produced and no undefined references after the final pass.

**Step 4: Commit**

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: scaffold P0 edge-flux JCP manuscript"
```

## Task 7: Generate P0-Only Publication Figures

**Files:**
- Create: `docs/make_p0_edge_flux_figures.py`
- Create: `docs/figures/p0_edge_flux/planar_convergence.pdf`
- Conditionally create: `docs/figures/p0_edge_flux/spherical_results.pdf`
- Read: `docs/p0_edge_flux_convergence.csv`
- Read: `docs/p0_edge_flux_spherical.csv`

**Step 1: Write figure-generator tests as numeric assertions**

Before plotting, implement functions that:

- load the two CSV files;
- reject duplicate `(mesh_family, field, level)` rows;
- verify positive `h` and non-negative errors;
- recompute every reported planar rate from adjacent rows;
- reject a fitted-order annotation when fewer than three comparable levels exist.

Run the script in validation-only mode and expect PASS.

**Step 2: Create the planar convergence figure**

Use log-log axes and separate panels or line styles for:

- structured quad-to-quad;
- Voronoi-to-Voronoi;
- mixed Voronoi-to-quad where useful.

Include an `O(h)` guide and optionally an `O(h^2)` guide. The caption—not the
plot title—will interpret structured superconvergence.

**Step 3: Select the spherical presentation using the evidence rubric**

Create `spherical_results.pdf` only if the evidence ledger identifies at least
one fitted-order-eligible spherical sequence. Otherwise do not create this PDF;
Task 12 must present the spherical results as a table of error reductions and
invariants. Include no RLL data.

**Step 4: Verify outputs**

Run:

```bash
env MPLCONFIGDIR=/tmp/mimetic-sphpoly-mpl \
  python3 docs/make_p0_edge_flux_figures.py
```

Expected: `planar_convergence.pdf` is produced; `spherical_results.pdf` is
produced only when the eligibility rule is met; validation assertions pass.

**Step 5: Visually inspect every generated figure**

Check labels, units, legend, color accessibility, non-overlapping annotations,
and consistency with the CSV values for every figure actually produced.

**Step 6: Commit each figure with its generator**

```bash
git add docs/make_p0_edge_flux_figures.py \
  docs/figures/p0_edge_flux/planar_convergence.pdf \
  docs/p0_edge_flux_evidence.md docs/REPRODUCIBILITY.md
# Add docs/figures/p0_edge_flux/spherical_results.pdf only if created.
git commit -m "figures: add P0 planar and spherical validation plots"
```

Justification: the script and generated PDFs are inseparable reproducible figure outputs.

## Task 8: Draft Abstract, Introduction, And Related Work

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`
- Read: `docs/p0_edge_flux_evidence.md`
- Read: `docs/references_p0.bib`

**Step 1: Draft an evidence-qualified abstract**

The abstract must state:

- the edge-flux remapping problem;
- the five-stage P0 method;
- exact conservation/closure properties;
- evidence-qualified language such as “at least first-order on qualifying
  tested sequences,” derived directly from the evidence ledger;
- planar and spherical polygon coverage;
- no high-order preliminary results.

**Step 2: Draft the introduction**

Explain the scientific need, distinguish edge-flux from scalar remapping, and
list contributions without implementation references or priority claims.

**Step 3: Draft related work**

Synthesize the focused bibliography. Distinguish the present construction from
mixed finite-element discretization, scalar conservative remapping, and
high-order methods.

**Step 4: Verify**

Run the hygiene gate and full clean LaTeX/BibTeX build. Search the log for
undefined citations.

Use exactly:

```bash
bash scripts/check_p0_manuscript_hygiene.sh
bash scripts/build_p0_manuscript.sh
```

**Step 5: Commit**

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: draft P0 manuscript introduction and related work"
```

## Task 9: Draft Discrete Data And Local Reconstruction

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`
- Read: `docs/mimetic_voronoi_report.tex` only as a source of verified mathematics
- Read: `docs/voronoi_p0_report.tex` only as a source of verified mathematics

**Step 1: Define orientation and flux data**

Introduce intrinsic edge orientation, cell-local signs, integrated normal flux,
and the discrete divergence theorem.

**Step 2: Derive the reconstruction**

Present the constant-divergence particular field plus harmonic gradients,
minimum-energy objective, edge constraints, dependent constraint removal, and
KKT system.

**Step 3: State propositions precisely**

Include propositions for:

- constant divergence;
- exact reproduction of prescribed edge fluxes under compatibility;
- divergence-free harmonic completion.

Do not claim a general uniqueness theorem unless assumptions are stated and
supported.

**Step 4: Explain scaling mathematically**

Describe nondimensional local coordinates and conditioning without mentioning
software or a specific solver.

**Step 5: Verify and commit**

Run `bash scripts/check_p0_manuscript_hygiene.sh` and
`bash scripts/build_p0_manuscript.sh`, then:

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: derive lowest-order harmonic reconstruction"
```

## Task 10: Draft Transfer And Conforming Projection

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`

**Step 1: Draft nonmatching edge transfer**

Define target-edge clipping, subsegment ownership, and source-cell contribution
summation. Add pseudocode independent of any implementation language.

**Step 2: Prove overlap conservation**

Apply the divergence theorem to a clipped overlap and show that harmonic terms
contribute zero net boundary flux.

**Step 3: Express the linear operator**

Define directed source and target vectors and the sparse transfer matrix.

**Step 4: Derive the target-skeleton projection**

Present the incidence matrix, constrained minimum-correction problem,
compatibility condition, nullspace handling, and Schur-complement solution.

**Step 5: State conservation consequences**

Separate overlap conservation, global conservation, unique-edge continuity,
and exact target-cell divergence closure.

**Step 6: Verify and commit**

Run `bash scripts/check_p0_manuscript_hygiene.sh` and
`bash scripts/build_p0_manuscript.sh`, then:

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: derive nonmatching transfer and conforming projection"
```

## Task 11: Draft The Spherical Extension

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`

**Step 1: Define the chart geometry**

Introduce the source-cell gnomonic frame, forward/inverse maps, and the
great-circle-to-line property.

**Step 2: Derive flux preservation**

Present the contravariant Piola transformation and show preservation of normal
flux integrals between sphere and chart.

**Step 3: State assumptions**

Require convex great-circle cells contained in the chart hemisphere. Do not
discuss RLL meshes or round-trip transfer; a general statement that the method
assumes great-circle boundaries is permitted.

**Step 4: Verify scope mechanically**

Run the hygiene gate; it already forbids `RLL`, `round-trip`, and `roundtrip`.

**Step 5: Compile and commit**

Run:

```bash
bash scripts/check_p0_manuscript_hygiene.sh
bash scripts/build_p0_manuscript.sh
```

Then commit:

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: add spherical gnomonic flux-transfer formulation"
```

## Task 12: Draft Numerical Experiments And Results

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`
- Read: `docs/p0_edge_flux_evidence.md`
- Read: `docs/p0_edge_flux_convergence.csv`
- Read: `docs/p0_edge_flux_spherical.csv`
- Read: `docs/figures/p0_edge_flux/planar_convergence.pdf`
- Conditionally read: `docs/figures/p0_edge_flux/spherical_results.pdf` if Task 7 created it

**Step 1: Define fields, meshes, and norms**

State all manufactured fields, exact divergence, mesh construction, mesh-size
proxy, edge-error norms, and conservation metrics mathematically.

**Step 2: Add exact-recovery and invariant results**

Create one table for compatible-field recovery and one consolidated residual
table covering overlap, global, reintegration, direct/operator equivalence, and
target-cell closure.

**Step 3: Add planar convergence results**

Insert `planar_convergence.pdf`. Report consecutive and fitted rates by field
and topology. Describe structured rates above one as observed superconvergence.
Discuss oscillatory irregular-mesh rates without averaging away unfavorable
levels.

**Step 4: Add spherical results**

Insert `spherical_results.pdf` only if Task 7 created it. Otherwise use a table.
Apply the Task 3 operational rubric: two-level cases receive error-reduction
statements only; only fitted-order-eligible sequences receive order claims.

**Step 5: Cross-check every number**

For every numeric literal in the results section, add or verify a matching row
in `docs/p0_edge_flux_evidence.md`.

**Step 6: Verify and commit**

Run `bash scripts/check_p0_manuscript_hygiene.sh` and
`bash scripts/build_p0_manuscript.sh`, then:

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: add evidence-qualified P0 validation results"
```

## Task 13: Produce Apples-To-Apples Planar Patch-Recovery Evidence

**Files:**
- Create: `docs/p0_patch_extension.csv`
- Modify: `tests/patch_recovery_test.cpp`
- Read: `tests/convergence_validation_test.cpp`
- Read: `tests/patch_recovery_test.cpp`

**Step 1: Audit whether comparable data already exist**

Require the same:

- manufactured field;
- source/target mesh sequence;
- error norm;
- conforming/raw choice;
- mesh-size proxy;
- quadrature accuracy

for P0 and patch-recovered higher-order paths.

If current outputs do not satisfy all six criteria, do not compare their raw
error values in the manuscript.

**Step 2: Write the failing comparison check**

Add an optional CLI mode to `patch_recovery_test`:

```text
patch_recovery_test --manuscript-csv <path>
```

The mode must emit P0 and the first enriched order on the same planar
quad-to-quad sequence. It may also emit a second enriched order when that path
is already supported and stable on the identical sequence. The executable must
reject any missing required method/level row or mismatched field, mesh, norm,
quadrature, or mesh-size definition. The no-argument regression behavior must
remain unchanged.

**Step 3: Implement the minimum experiment**

Reuse existing reconstruction and transfer paths. Do not refactor production
algorithms. Emit CSV columns:

```text
method,order,level,h,l2_relative,linf,conservation,rate_l2
```

At minimum include P0 and one patch-recovered enriched method. Include a second
enriched order only if it is already supported and stable on the same sequence.

**Step 4: Run and validate**

Run:

```bash
cmake --build build --target patch_recovery_test --parallel
./build/patch_recovery_test --manuscript-csv /tmp/p0_patch_extension_a.csv
./build/patch_recovery_test --manuscript-csv /tmp/p0_patch_extension_b.csv
cmp /tmp/p0_patch_extension_a.csv /tmp/p0_patch_extension_b.csv
cp /tmp/p0_patch_extension_a.csv docs/p0_patch_extension.csv
```

Expected:

- conservation remains below `5e-13`;
- no method is omitted because of an unfavorable result;
- the CSV is deterministic across two runs.

If enriched errors are not lower than P0 on at least the two finest levels,
retain the complete CSV but do not create a quantitative comparison figure or
claim improved errors. In that case, Task 14 becomes a prose-only future-work
section that reports the negative preliminary result and its validation gap.

**Step 5: Commit implementation and its direct evidence together**

If implementation changes were required:

```bash
git add tests/patch_recovery_test.cpp docs/p0_patch_extension.csv \
  docs/p0_edge_flux_evidence.md docs/REPRODUCIBILITY.md
git commit -m "tests: compare P0 and patch-recovered planar transfer"
```

## Task 14: Generate And Draft The Planar High-Order Outlook

**Files:**
- Modify: `docs/make_p0_edge_flux_figures.py`
- Conditionally create: `docs/figures/p0_edge_flux/planar_patch_extension.pdf`
- Modify: `docs/p0_edge_flux_remap_jcp.tex`
- Read: `docs/p0_patch_extension.csv`

**Step 1: Add validation for comparable rows**

The figure generator must assert that all plotted methods share the same field,
mesh family, levels, `h`, and norm.

**Step 2: Select the quantitative or negative-result presentation**

If Task 13 found lower enriched errors on at least the two finest levels, plot
P0 and patch-recovered enriched methods on the same log-log axes and create
`planar_patch_extension.pdf`. Include only planar results. Use conservative
labels such as “preliminary enriched reconstruction” and define moment order in
the caption.

Otherwise, do not create the PDF. Present the complete preliminary comparison
as a compact table or prose-only negative result, without claiming improved
errors.

**Step 3: Draft the future-work section**

Explain:

- why one integrated flux does not determine a higher-order normal trace;
- how neighboring-cell patch recovery synthesizes missing moments;
- how higher Legendre moments and enriched polygonal H(div) reconstruction can
  reduce error;
- what the preliminary planar data demonstrate and do not demonstrate;
- why irregular Voronoi and spherical high-order validation remain future work.

Do not include detailed VEM derivations or implementation descriptions.

**Step 4: Verify claims**

Every number must trace to `docs/p0_patch_extension.csv` and the evidence ledger.
Update `docs/REPRODUCIBILITY.md` with the comparison and figure workflows in the
same task.

**Step 5: Commit the figure and section together**

```bash
git add docs/make_p0_edge_flux_figures.py \
  docs/p0_edge_flux_remap_jcp.tex docs/p0_edge_flux_evidence.md \
  docs/REPRODUCIBILITY.md
# Add docs/figures/p0_edge_flux/planar_patch_extension.pdf only if created.
git commit -m "report: add preliminary planar high-order extension"
```

Justification: the manuscript section and evidence records must be committed
together with whichever presentation the comparison supports: a generated
figure for improved errors, or a prose/table negative-result presentation.

## Task 15: Draft Discussion And Conclusions

**Files:**
- Modify: `docs/p0_edge_flux_remap_jcp.tex`

**Step 1: Draft discussion**

Cover:

- nominal first-order consistency;
- structured superconvergence;
- irregular non-nested rate variation;
- distinction between conservation and approximation error;
- distinction between measured order and two-level error reduction;
- geometric and information-content assumptions.

**Step 2: Draft conclusions**

Keep conclusions P0-centered. Mention patch recovery only as a future extension.

**Step 3: Re-read abstract against conclusions**

Ensure every abstract claim is supported in the results and no preliminary
high-order claim appears as a primary contribution.

**Step 4: Verify and commit**

Run `bash scripts/check_p0_manuscript_hygiene.sh` and
`bash scripts/build_p0_manuscript.sh`, then:

```bash
git add docs/p0_edge_flux_remap_jcp.tex
git commit -m "report: complete P0 manuscript discussion and conclusions"
```

## Task 16: Write The Reproducibility Companion

**Files:**
- Modify: `docs/REPRODUCIBILITY.md`
- Read: `docs/p0_edge_flux_evidence.md`

**Step 1: Complete software provenance**

Include revision, operating system, compiler, CMake, MOAB, Eigen, Python, and
Matplotlib versions. This document may contain all implementation-specific
details excluded from the paper.

**Step 2: Map every manuscript result**

For each figure and table, record:

- source executable or generation workflow;
- exact command;
- input mesh levels and deterministic seeds;
- raw output/CSV;
- plotting command;
- expected checksum or key numeric values.

**Step 3: Separate the two realizations**

Clearly identify the principal lowest-order implementation and the independent
Halton-Voronoi reference realization. Document mathematical similarities and
differences without implying bitwise equivalence.

**Step 4: Add validation and tolerance table**

Include the `5e-13` conservation tolerance and the expected ranges for all
reported residuals and rates.

**Step 5: Verify completeness and commit**

Check that every manuscript figure/table label appears in `REPRODUCIBILITY.md`.

```bash
git add docs/REPRODUCIBILITY.md
git commit -m "docs: add P0 manuscript reproducibility companion"
```

## Task 17: Scientific And Editorial Review

**Files:**
- Modify as needed: `docs/p0_edge_flux_remap_jcp.tex`
- Modify as needed: `docs/references_p0.bib`
- Modify as needed: `docs/p0_edge_flux_evidence.md`

**Step 1: Run a scientific correctness review**

Use `research-code-review` to check:

- equation-to-method correspondence;
- orientation and sign consistency;
- conservation proofs;
- spherical Piola statements;
- convergence interpretation;
- distinction between facts, inferences, and preliminary evidence.

**Step 2: Run an adversarial manuscript review**

Ask an independent reviewer to identify unsupported novelty, overclaiming,
missing assumptions, inconsistent notation, and missing citations.

**Step 3: Triage every review finding**

Create `docs/p0_manuscript_review.md` with each finding marked `accepted`,
`rejected`, or `deferred`, including evidence and rationale. A correction is
verified only when supported by a derivation, cited primary source, numerical
evidence ledger entry, or reproducible test. Reviewer disagreement is resolved
by the manuscript’s evidence policy, not majority vote.

Any blocking contradiction in conservation, orientation, Piola mapping, or a
headline convergence claim stops execution; revise the manuscript scope and
design before committing.

**Step 4: Apply verified corrections**

Do not add speculative claims or unrelated high-order material.

**Step 5: Verify and commit**

Run `bash scripts/check_p0_manuscript_hygiene.sh` and
`bash scripts/build_p0_manuscript.sh`, then:

```bash
git add docs/p0_edge_flux_remap_jcp.tex docs/references_p0.bib \
  docs/p0_edge_flux_evidence.md docs/p0_manuscript_review.md \
  docs/REPRODUCIBILITY.md
git commit -m "report: address scientific review of P0 manuscript"
```

## Task 18: Final Acceptance And Clean Build

**Files:**
- Verify: all deliverables
- Verify unchanged: `docs/mimetic_voronoi_report.tex`
- Verify unchanged: `docs/voronoi_p0_report.tex`

**Step 1: Verify existing manuscripts are unchanged**

Run:

```bash
shasum -a 256 -c /tmp/p0-manuscript-baseline.sha256
```

Expected: both `OK`.

**Step 2: Run all numerical tests**

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Expected: all tests pass, including any new planar comparison test.

**Step 3: Regenerate all new figures**

```bash
env MPLCONFIGDIR=/tmp/mimetic-sphpoly-mpl \
  python3 docs/make_p0_edge_flux_figures.py
```

Expected: deterministic PDFs and all numeric assertions pass.

**Step 4: Run manuscript hygiene**

```bash
bash scripts/check_p0_manuscript_hygiene.sh
```

Expected: PASS.

**Step 5: Perform a clean LaTeX/BibTeX build**

```bash
bash scripts/build_p0_manuscript.sh
```

Expected:

- PDF exists;
- no undefined references or citations;
- no LaTeX errors;
- only explicitly accepted layout warnings remain.

**Step 6: Inspect the final PDF**

Check title page, equations, algorithms, tables, figure legibility, float order,
references, and page breaks.

**Step 7: Verify working tree scope**

```bash
git status --short
DESIGN_COMMIT=$(git log -1 --format=%H -- \
  docs/plans/2026-07-28-p0-edge-flux-manuscript-design.md)
git diff --stat "$DESIGN_COMMIT"..HEAD
```

Expected: only approved manuscript, data, figure, bibliography,
reproducibility, test, and generator files changed.

**Step 8: Final commit only if acceptance fixes remain**

```bash
git add <acceptance-fix-files>
git commit -m "report: finalize P0 edge-flux manuscript"
```

Do not create an empty final commit.

## Final Definition Of Done

- New JCP-style P0 manuscript compiles independently.
- Manuscript contains no implementation or repository references.
- Existing manuscripts are unchanged.
- P0 planar and spherical claims are evidence-qualified.
- RLL round-trip material is absent.
- Higher-order material is confined to a compact planar preliminary outlook.
- All manuscript numeric claims appear in the evidence ledger.
- All figures and tables are mapped in `REPRODUCIBILITY.md`.
- Conservation residuals satisfy `5e-13`.
- Full numerical test suite passes.
- Final PDF has resolved citations and references.
