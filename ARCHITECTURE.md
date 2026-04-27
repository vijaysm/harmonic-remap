================================================================================
                    MIMETIC-SPHPOLY ARCHITECTURE
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC API LAYER                                   │
│                     (include/mimetic/mimetic.hpp)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Classes:                                                                    │
│    • MimeticInterpolator              ← Low-order level-2 reconstruction      │
│    • PlanarMomentInterpolator         ← High-order p=1,2,3 moments          │
│                                                                              │
│  Data Types:                                                                 │
│    • LocalPolygon, LocalEdge          ← Planar cell data                    │
│    • SphericalPolygon, SphericalEdge  ← Spherical cell data                 │
│    • ReconstructionCoeffs             ← Harmonic coefficients [d,a,b,...]   │
│    • DirectedEdgeDof, EdgeTransferResult, SparseEdgeProjection              │
│    • MomentReconstruction, EdgeMomentTransferResult                         │
│                                                                              │
│  Enums:                                                                      │
│    • GeometryMode { Planar, SphericalGnomonic }                             │
│    • CellAverageReductionMode { Harmonic }                                  │
│                                                                              │
│  Options:                                                                    │
│    • GeometryOptions (tolerances, mode, metric_weighted)                    │
│    • MomentMethodOptions (p, quadrature_points, regularization, ...)        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ includes
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                   IMPLEMENTATION & NUMERICS LAYER                            │
│                       (src/mimetic.cpp: 3680 lines)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GEOMETRY CONSTRUCTION                                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • local_polygon()           → extract cell, centroid-relative frame │   │
│  │ • spherical_polygon()       → normalize vertices, gnomonic chart    │   │
│  │ • local_edges()             → directed edges with outward normals   │   │
│  │ • make_gnomonic_frame()     → tangent space for spherical cell     │   │
│  │ • project_gnomonic()        → sphere → chart (straight lines!)     │   │
│  │ • inverse_gnomonic()        → chart → sphere                       │   │
│  │ • gnomonic_jacobian()       → differential geometry                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ QUADRATURE RULES                                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • integrate_edge_scalar()         ← 4-point Gauss-Legendre (deg 7) │   │
│  │ • integrate_triangle_scalar()     ← 7-point (deg 4)                │   │
│  │ • integrate_triangle_highorder()  ← 13-point Dunavant (deg 7)      │   │
│  │ • integrate_triangle_adaptive()   ← recursive subdivision           │   │
│  │ • integrate_polygon_adaptive()    ← fan-triangulation              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ HARMONIC BASIS & RECONSTRUCTION (Low-Order)                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • eval_harmonic_basis()           ← P_k, Q_k and gradients         │   │
│  │ • reconstruct_source_polygon()    ← KKT solver for [d, a, b]       │   │
│  │ • source_reconstruction_matrix()  ← local linear map               │   │
│  │ • velocity()                      ← evaluate u_h(x)                 │   │
│  │ • line_integral()                 ← exact edge integral            │   │
│  │ • edge_flux()                     ← normal flux integral           │   │
│  │ • polygon_boundary_flux()         ← flux on clipped polygon        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SPLIT BASIS & MOMENTS (High-Order p=1,2,3)                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • split_basis_modes()             ← poly-div + harmonic + bubble   │   │
│  │ • evaluate_split_basis()          ← evaluate basis at point        │   │
│  │ • moment_reconstruct()            ← build local system             │   │
│  │ • edge_moments()                  ← Legendre moment evaluation     │   │
│  │ • cell_vector_moments()           ← cell moment transfer           │   │
│  │ • local_length_scale()            ← numerical stability scaling    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CLIPPING & TRANSFER                                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • clip_segment_to_convex_polygon()     ← Sutherland-Hodgman        │   │
│  │ • transfer_source_to_target_edges()    ← direct transfer           │   │
│  │ • clipped_edge_transfer_contribution() ← single edge pair          │   │
│  │ • transfer_source_to_target_edge_moments() ← moment accumulation   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SPARSE OPERATOR ASSEMBLY                                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • assemble_edge_projection_operator()  ← U_t = P U_s               │   │
│  │ • assemble_conforming_constraints()    ← global divergence         │   │
│  │ • project_target_fluxes_to_hdiv_conforming() ← constrained LS      │   │
│  │ • project_target_edge_moments_to_hdiv_conforming() ← high-order    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SPATIAL ACCELERATION & I/O                                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • SpatialIndex (nanoflann k-d tree)    ← candidate acceleration    │   │
│  │ • find_overlap_candidates()            ← broad-phase search        │   │
│  │ • write_matrix_market()                ← sparse matrix export      │   │
│  │ • write_edge_map_csv()                 ← edge DOF mapping          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SPHERICAL-SPECIFIC                                                  │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ • lift_contravariant_piola()      ← chart vector → surface vector  │   │
│  │ • pullback_contravariant_piola()  ← surface vector → chart         │   │
│  │ • spherical_edges()                ← build edge records            │   │
│  │ • spherical_transfer_edge()        ← gnomonic-chart transfer       │   │
│  │ • chart_polygon_surface_area()     ← physical area on sphere       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ uses
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL DEPENDENCIES                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MOAB (Mesh & Geometry)          Eigen (Linear Algebra)   nanoflann (Search)│
│  ├─ Core                         ├─ Dense matrices        ├─ k-d tree       │
│  ├─ Entity handles               ├─ Sparse matrices       └─ radius search  │
│  ├─ Adjacency queries            ├─ Linear solvers (LDLT, QR)              │
│  └─ Tag storage/retrieval        └─ Vector/Matrix ops                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘


================================================================================
                          ALGORITHM DATA FLOW
================================================================================

LOW-ORDER RECONSTRUCTION & TRANSFER:
───────────────────────────────────

Input: Source cells with directed edge fluxes U_f

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Geometry Assembly                                            │
  ├─────────────────────────────────────────────────────────────────┤
  │    local_polygon(mesh, cell)                                    │
  │         ↓                                                       │
  │    [LocalPolygon: centroid, points, area, spherical frame]     │
  │         ↓                                                       │
  │    local_edges(mesh, polygon)                                   │
  │         ↓                                                       │
  │    [LocalEdge: directed edges, outward normals]                │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2. Reconstruction (Per Source Cell)                             │
  ├─────────────────────────────────────────────────────────────────┤
  │    read edge fluxes {U_f}                                       │
  │         ↓                                                       │
  │    compute d = Σ U_f / area  (constant divergence)             │
  │         ↓                                                       │
  │    assemble Gram matrix V for harmonics {P_k, Q_k}             │
  │         ↓                                                       │
  │    solve KKT: [a₁, b₁, a₂, b₂] = V⁻¹ × moments                │
  │         ↓                                                       │
  │    store ReconstructionCoeffs [d, a₁, b₁, a₂, b₂]            │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 3. Direct Edge-Wise Transfer (Per Target Cell)                  │
  ├─────────────────────────────────────────────────────────────────┤
  │    For each target edge e_t:                                    │
  │      For each candidate source cell K_s:                        │
  │        project(e_t endpoints) → e'_t in source chart           │
  │        clip(e'_t) ∩ (source polygon) → segment(s)             │
  │        For each clipped segment:                               │
  │          integrate u_h^s · n_t ds on segment                   │
  │          accumulate into U_t[e_t]                              │
  │         ↓                                                       │
  │    [EdgeTransferResult: target fluxes + contributions]         │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 4. Optional Global Conforming Projection                        │
  ├─────────────────────────────────────────────────────────────────┤
  │    collapse directed edges → unique geometric edges             │
  │         ↓                                                       │
  │    assemble exact divergence constraints (per target cell)     │
  │         ↓                                                       │
  │    constrained least-squares solve:                            │
  │      min ‖unique_fluxes - directed_fluxes‖²                   │
  │      s.t. div_i = d_i area_i  (exact per cell)                │
  │         ↓                                                       │
  │    [ConformingEdgeTransferResult: globally valid fluxes]       │
  └─────────────────────────────────────────────────────────────────┘

Output: Target cell edge fluxes (directed or global) with conservation


HIGH-ORDER MOMENT TRANSFER (p=1,2,3):
──────────────────────────────────────

Input: Source cells with edge Legendre moments {m_k}_{k=0..p}

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Split Basis Construction                                     │
  ├─────────────────────────────────────────────────────────────────┤
  │    Basis modes:                                                 │
  │      • Polynomial divergence:  ∇·u = p_k(x,y)  [p+1 modes]     │
  │      • Harmonic gradients:     ∇P_k, ∇Q_k      [divergence-free]
  │      • Bubble modes:           interior          [vanish on ∂]  │
  │         ↓                                                       │
  │    assemble local constraint matrix:                           │
  │      - m₀ edge flux: EXACT hard constraint                     │
  │      - higher moments: soft weighted constraints               │
  │      - optional cell moments: soft constraints                 │
  │         ↓                                                       │
  │    solve local least-squares system (scaled by length scale)   │
  │         ↓                                                       │
  │    [MomentReconstruction: basis coefficients]                  │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2. Moment Transfer (Same Clipping as Low-Order)                 │
  ├─────────────────────────────────────────────────────────────────┤
  │    For each target edge e_t:                                    │
  │      For each source cell K_s:                                 │
  │        clip e_t against K_s                                    │
  │        For each clipped segment:                               │
  │          evaluate Legendre moments [m₀, m₁, ..., m_p]         │
  │          accumulate into target moment vector                  │
  │         ↓                                                       │
  │    [EdgeMomentTransferResult: target edge moments]             │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 3. Global Skeleton Conforming Projection                        │
  ├─────────────────────────────────────────────────────────────────┤
  │    Similar to flux projection: collapse directed → unique       │
  │    Enforce global divergence from reconstructed moments         │
  │    Constrained LS on moment-0 fluxes                           │
  │         ↓                                                       │
  │    [ConformingEdgeMomentTransferResult: global moments]         │
  └─────────────────────────────────────────────────────────────────┘

Output: Target cell edge moments (directed or global) with conservation


SPHERICAL GNOMONIC PATH:
────────────────────────

Special feature: All numerics in source-cell tangent-plane charts!

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Spherical Polygon & Frame                                    │
  ├─────────────────────────────────────────────────────────────────┤
  │    spherical_polygon(mesh, cell) →                              │
  │      normalize 3D vertices                                      │
  │      construct GnomonicFrame (center, e_x, e_y tangent vectors)│
  │      project vertices to chart: xi = project_gnomonic(p)       │
  │         ↓                                                       │
  │    [SphericalPolygon: 3D + chart coordinates]                  │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2. Reconstruction in Chart                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │    Same KKT solver as planar, but:                              │
  │      • Gram matrix weighted by Hodge metric J^T J / |J|        │
  │      • operates on chart edge fluxes (Piola pullback)          │
  │      • evaluates reconstructed field in chart                  │
  └─────────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 3. Transfer: Great Circles → Straight Lines                     │
  ├─────────────────────────────────────────────────────────────────┤
  │    For each target edge e_t (great circle on sphere):          │
  │      For each source cell K_s:                                 │
  │        project e_t to source chart: e'_t                       │
  │          (great circle becomes straight line under gnomonic!)  │
  │        clip e'_t against projected polygon                     │
  │        evaluate reconstruction in chart (plain 2D evaluation)  │
  │        accumulate                                              │
  │      lift result to surface via Piola for diagnostics          │
  │         ↓                                                       │
  │    [EdgeTransferResult in surface normal flux coordinates]     │
  └─────────────────────────────────────────────────────────────────┘

Output: Target edge fluxes on sphere, with global conservation


================================================================================
                              DATA STRUCTURES
================================================================================

LocalPolygon (planar cell):
  • vertices: EntityHandle[]
  • points: Vector2d[] (centroid-relative)
  • centroid: Vector2d
  • area: double
  • spherical_area: double (for tracking)
  • points_3d: Vector3d[] (ambient coordinates, if spherical)
  • centroid_3d: Vector3d
  • e_x, e_y, n: Vector3d (tangent frame for spherical)

ReconstructionCoeffs (low-order):
  • d: double (divergence)
  • harmonic: vector<double> [a1, b1, a2, b2, ...]

MomentReconstruction (high-order):
  • options: MomentMethodOptions
  • vector_polynomial_degree: int
  • harmonic_degree: int
  • divergence_mode_count, harmonic_mode_count, bubble_mode_count
  • length_scale: double (numerical conditioning)
  • coefficients: vector<double> (basis expansion)

EdgeTransferResult (direct transfer):
  • target_edges: vector<DirectedEdgeDof>
  • target_fluxes: vector<double>
  • contributions: vector<EdgeTransferContribution>
    - target_dof_index, source_polygon, segment_a/b, flux

SparseEdgeProjection (sparse operator):
  • matrix: SparseMatrix<double> (rows=target, cols=source)
  • source_edges: vector<DirectedEdgeDof>
  • target_edges: vector<DirectedEdgeDof>

ConformingEdgeTransferResult (global projection):
  • target_edges: vector<DirectedEdgeDof>
  • target_fluxes: vector<double> (directed)
  • target_cells: vector<EntityHandle>
  • target_divergence_integrals: vector<double>
  • target_edge_to_unique: vector<size_t>
  • target_edge_signs: vector<int>
  • unique_edge_fluxes: vector<double> (global)


================================================================================
                           TEST ORGANIZATION
================================================================================

Test Hierarchy:

  Planar
    ├─ Low-order
    │  ├─ patch_test                    (constant field recovery)
    │  ├─ conservative_intersection_test (rectangular grid overlap)
    │  ├─ voronoi_intersection_test     (n-sided Voronoi)
    │  └─ convergence_validation_test   (h-refinement, 4 fields)
    │
    └─ High-order
       ├─ high_order_edge_moment_test         (exact recovery p=1,2)
       ├─ high_order_hdiv_convergence_test    (convergence p=1,2,3, CSV output)
       └─ hdiv_conforming_projection_test     (global skeleton solve)

  Spherical
    ├─ Low-order
    │  ├─ spherical_geometry_test      (primitives, round-trip)
    │  ├─ spherical_quad_test          (cubed-sphere, VTK output)
    │  ├─ spherical_voronoi_test       (Voronoi transfer)
    │  └─ spherical_scalar_test        (scalar conservation)
    │
    └─ High-order
       ├─ spherical_high_order_moment_test    (regression)
       └─ spherical_high_order_hdiv_convergence_test (p=1,2,3 study)

  Utilities
    ├─ cell_average_transfer_test    (cell-average reduction mode)
    ├─ dump_visuals                  (visualization helper)
    ├─ roundtrip_analysis            (diagnostics)
    └─ p3_roundtrip_study            (p=3 round-trip investigation)


================================================================================
