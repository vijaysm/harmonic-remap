# P0 Conservative Edge-Flux Manuscript Design

Date: 2026-07-28

## Objective

Create a new, self-contained Journal of Computational Physics-style manuscript
focused on lowest-order conservative edge-flux remapping between nonmatching
planar and spherical polygonal meshes. The paper will be method-centric and
publication-facing. It will not mention programming languages, source files,
tests, repository artifacts, or implementation identifiers.

The existing broad manuscript and compact planar P0 report remain unchanged.
The new paper is authored from a clean structure rather than by deleting
high-order material from either source.

## Working Title

**Conservative Edge-Flux Remapping on Nonmatching Planar and Spherical
Polygonal Meshes**

## Scientific Thesis

The manuscript presents a complete lowest-order one-way remapping operator
composed of:

1. one signed integrated normal-flux degree of freedom per directed edge;
2. a cell-local constant-divergence harmonic reconstruction;
3. geometric clipping and integration over target-edge/source-cell
   subsegments;
4. assembly of a linear source-to-target edge operator; and
5. a global target-skeleton H(div)-conforming projection that enforces the
   source-determined divergence integral in every target cell.

The principal evidence-qualified claim is:

> The method preserves overlap and global conservation to roundoff, reproduces
> compatible low-degree fields, and exhibits at least first-order one-way
> edge-flux convergence on the tested nonmatching polygonal meshes, with
> higher observed rates on structured planar refinements.

The title, abstract, principal results, and conclusions remain P0-focused.

## Scope

### Included

- Planar quadrilateral, Voronoi, and mixed polygonal mesh pairs.
- Spherical cubed-sphere, spherical Voronoi, and mixed great-circle polygon
  patches.
- Directed edge orientation and one integrated normal-flux degree of freedom.
- Discrete divergence and constant-divergence harmonic reconstruction.
- Edge clipping, subsegment integration, overlap conservation, and operator
  assembly.
- Global target-skeleton H(div)-conforming projection.
- Conservation, exact recovery, reintegration, operator equivalence, and
  one-way convergence evidence.
- A compact planar-only preliminary outlook on higher-order edge moments and
  patch-recovered moments.

### Excluded

- Any mention of actual C++ or Python code in the manuscript body.
- Source paths, test names, generated artifacts, commands, implementation
  identifiers, or software architecture.
- RLL-to-cubed-sphere-to-RLL round-trip results and their geometric limitation.
- Detailed VEM derivations, spherical high-order results, Piola-RT fallback,
  and high-order round-trip analysis.
- Claims of universal first-order convergence where refinement evidence is
  insufficient or non-monotone.

## Manuscript Architecture

### 1. Introduction

- Motivate compatible vector-field transfer across nonmatching meshes.
- Distinguish edge-normal flux remapping from conventional scalar cell-average
  conservative remapping.
- Identify the need to preserve both integral conservation and target-cell
  divergence compatibility.
- State the contributions and evidence-qualified convergence claims.

### 2. Related Work

- Raviart-Thomas spaces and classical H(div) mixed methods.
- Mimetic finite differences and Perot-Chartrand polygon reconstruction.
- Conservative remapping, including SCRIP- and TempestRemap-style overlap
  methods.
- Polygon clipping and robust nonmatching geometry.
- Gnomonic coordinates and contravariant Piola transformations on the sphere.
- Delimit the paper from high-order and virtual-element developments without
  reviewing them in depth.

### 3. Discrete Data and Local Reconstruction

- Define directed edge orientation and integrated normal fluxes.
- Derive the discrete cell-divergence identity.
- Introduce the polynomial-divergence plus harmonic-gradient decomposition.
- Formulate the constrained minimum-energy/KKT reconstruction.
- Explain coordinate scaling and exact reproduction of prescribed edge fluxes.
- State the assumptions required for solvability and polygon quality.

### 4. Nonmatching Edge-Flux Transfer

- Clip each target edge against source cells.
- Integrate the source reconstruction over each retained subsegment.
- Explain coincident-boundary ownership and double-counting avoidance.
- Derive the source-target overlap conservation identity.
- Express the transfer as a linear edge operator.
- Discuss deterministic assembly and asymptotic computational cost without
  implementation-specific details.

### 5. Target-Skeleton H(div)-Conforming Projection

- Explain why independently transferred directed edges need not satisfy a
  single target-cell divergence constraint.
- Formulate the minimum-correction constrained least-squares problem.
- Introduce the signed cell-edge incidence matrix and target divergence
  right-hand side.
- Derive the Schur-complement or KKT solution.
- Address the compatibility condition and constant nullspace.
- Prove preservation of global conservation and exact target-cell closure.
- Distinguish cell-local directed values from a unique geometric-edge output.

### 6. Spherical Extension

- Define convex great-circle polygons contained in a source-cell gnomonic
  hemisphere.
- Show that great circles become straight segments in a gnomonic chart.
- Introduce the contravariant Piola map and normal-flux preservation.
- Explain reuse of planar clipping and reconstruction in source-cell charts.
- State chart, convexity, and geometric assumptions.

### 7. Numerical Experiments

Organize experiments by the invariant or convergence question they answer.

1. Constant and compatible-field exact recovery.
2. Planar quadrilateral-to-quadrilateral refinements.
3. Planar Halton Voronoi-to-Voronoi refinements.
4. Mixed planar Voronoi-to-quadrilateral refinements, with non-monotone cases
   reported rather than hidden.
5. Spherical cubed-sphere identity and refinement evidence.
6. Spherical Voronoi and mixed-patch error reduction.
7. Conservation, reintegration, raw-versus-projected behavior, and linear
   operator equivalence.

The principal tables must identify the mesh family, norm, refinement levels,
measured rate, and conservation residual. Spherical studies with only two
comparable levels will be described as demonstrating error reduction, not a
measured asymptotic order.

The planar Halton-Voronoi results from the independent reference realization
may be included as corroborating methodology, but the manuscript will not
discuss implementation language or repository provenance. Any differences in
mesh generation, projection, or norms must be stated mathematically.

### 8. Discussion

- Interpret first-order consistency separately from structured-mesh
  superconvergence.
- Explain sensitivity to norm, mesh topology, shape regularity, and non-nested
  irregular refinement sequences.
- Distinguish measured asymptotic order from monotone error reduction.
- Discuss the information available from one scalar flux per edge and the
  limitations this places on richer normal traces.

### 9. Extensions to Higher Order

This is a concise future-work section, not a second methods paper.

- Motivate higher Legendre moments of the edge-normal trace.
- Explain that a single P0 edge flux does not determine high-order edge traces.
- Introduce patch reconstruction as a practical synthesis mechanism for
  missing moments using neighboring cell data.
- Briefly connect the enriched reconstruction to polygonal H(div), mixed
  virtual-element, polynomial recovery, and high-order conservative remapping
  literature.
- Include one compact planar-only figure or table comparing P0 with selected
  higher-order results.
- Show representative error reduction and order separation on structured
  planar meshes.
- Clearly label all such results as preliminary extensions and avoid importing
  the full high-order/VEM methodology into this manuscript.
- Report degraded small irregular Voronoi behavior honestly if that case is
  shown; otherwise restrict the preliminary figure to the cleanest planar
  evidence and name broader irregular validation as future work.

### 10. Conclusions

- Restate the supported result: conservative one-way P0 edge-flux remapping on
  nonmatching planar and spherical polygon meshes.
- Summarize exact invariants and evidence-qualified convergence.
- Identify high-order moment recovery and broader spherical refinement studies
  as future directions.

## Evidence Policy

- Every quantitative claim must trace internally to a verified numerical
  result, although those internal traces do not appear in the paper.
- Conservation claims use the absolute tolerance of 5 x 10^-13.
- A first-order claim requires at least three suitably comparable refinement
  levels and a clearly defined norm.
- Two-level spherical comparisons support only an error-reduction statement.
- Non-monotone or sub-first-order sequences must be discussed, excluded from a
  fitted-order claim with justification, or replaced by a cleaner experiment.
- Structured-mesh rates above first order are reported as observed
  superconvergence, not as the nominal method order.
- Preliminary high-order evidence must be clearly separated from the validated
  P0 contribution.

## Figure and Table Plan

- Reuse order-independent geometry diagrams when they fit the new notation.
- Prepare P0-only plots rather than extracting curves from mixed-order figures.
- Use a planar convergence figure with structured and irregular mesh panels.
- Use a spherical results figure only if the compared cases form a meaningful
  sequence; otherwise prefer a table.
- Include a conservation/invariant table covering overlap, global, target-cell,
  reintegration, and operator-equivalence residuals.
- Include one compact planar preliminary-high-order figure or table in the
  extensions section.
- Captions must define the error norm, mesh-size proxy, refinement sequence,
  and whether values are raw or conforming.

## Literature Plan

Retain or add primary references in these categories:

- Raviart-Thomas and mixed H(div) finite elements.
- Mimetic finite differences and polygonal mimetic reconstruction.
- Perot-Chartrand level-2 reconstruction.
- Conservative remapping and overlap-based transfer.
- Polygon clipping and robust geometric predicates.
- Gnomonic projection and contravariant Piola mapping.
- For the future-work section only: mixed virtual elements, polynomial/patch
  recovery, higher edge moments, and high-order conservative remapping.

Avoid unsupported priority claims such as "first-ever." Position the work as a
conservative edge-flux remapping construction with verified planar and
spherical behavior.

## Reproducibility Companion

Create a separate `REPRODUCIBILITY.md` alongside the manuscript. It may contain
implementation-specific material that is intentionally absent from the paper:

- exact software revision;
- dependencies and build environment;
- commands for each numerical study;
- mapping from manuscript tables/figures to generating workflows;
- deterministic mesh seeds and refinement parameters;
- tolerances and acceptance checks;
- raw output and CSV provenance;
- known differences between the principal and independent reference
  realizations.

## Deliverables

1. A new JCP-style LaTeX manuscript, provisionally
   `docs/p0_edge_flux_remap_jcp.tex`.
2. A focused bibliography or verified subset of the existing bibliography.
3. P0-only figures and tables plus one planar preliminary-high-order comparison.
4. `docs/REPRODUCIBILITY.md` containing the technical provenance excluded from
   the manuscript.
5. A compiled PDF with resolved references and no new LaTeX errors.

## Acceptance Criteria

- The manuscript compiles independently in an Elsevier/JCP-compatible layout.
- The paper contains no code, repository, artifact, command, or programming-
  language references.
- No RLL round-trip analysis appears.
- The main method and conclusions remain P0-focused.
- Higher-order material is confined to the planar preliminary extension before
  the conclusion.
- Every numerical claim is internally traceable and evidence-qualified.
- Existing manuscripts are unchanged.
- The reproducibility companion contains the implementation details omitted
  from the paper.
