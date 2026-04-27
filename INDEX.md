# Mimetic-SphPoly Documentation Index

## 📚 Main Documentation Files (NEW)

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
**For:** Quick overview, common tasks, cheat sheet  
**Size:** 303 lines, 10.5 KB  
**Contains:**
- One-line project summary
- Quick start (build/test commands)
- Core components overview
- Convergence rates
- Common tasks with examples
- Known limitations
- Links to other resources

**Read this first!** It's concise and tells you everything you need to get started.

---

### 2. **EXPLORATION.md** 📖 COMPREHENSIVE GUIDE
**For:** Complete technical understanding, all details  
**Size:** 535 lines, 22.9 KB  
**Contains:**
1. Project Overview
2. Directory Structure & Organization
3. Key Source Files & Their Roles
4. Build System (CMake)
5. Test Files & Validation
6. Documentation & Resources
7. Algorithm Pipeline (low/high/spherical paths)
8. Technical Design Decisions
9. Convergence Performance
10. Known Limitations
11. Working with the Code
12. Code Statistics
13. Key Dependencies

**Read this for:** Deep understanding of structure, algorithms, and design choices.

---

### 3. **ARCHITECTURE.md** 🏗️ SYSTEM DESIGN
**For:** Implementation details, data structures, function organization  
**Size:** 378 lines, 32.7 KB  
**Contains:**
- API Layer diagram (classes, types, enums)
- Implementation Layer subsystems (geometry, quadrature, reconstruction, etc.)
- External Dependencies overview
- Algorithm Data Flows (3 detailed pipelines)
- Data Structures (LocalPolygon, ReconstructionCoeffs, etc.)
- Test Organization hierarchy

**Read this for:** Understanding how code is organized, what functions do what, data flow.

---

## 📖 Existing Documentation Files

### README.md
- Full project description
- Mathematical background (Perot-Chartrand method)
- Geometry backends explanation
- Conservative transfer description
- Higher-order edge moments (p=1,2,3)
- Spherical structured/unstructured drivers
- Build and test instructions
- Current limitations

### CLAUDE.md
- Guidance for Claude Code (claude.ai/code)
- Build and test quick commands
- Architecture overview
- Algorithm pipeline
- Key design decisions
- Test organization
- Manufactured test fields
- Convergence rates
- Working conventions

### docs/code_documentation.md
- Code-to-manuscript traceability
- Source layout and file roles
- Manuscript-to-code mapping
- Directed edge convention
- Reconstruction, transfer, and conforming paths
- High-order moment path
- Sparse projection path
- Global target-edge conforming projection

### docs/mimetic_voronoi_report.tex/pdf
- Full technical manuscript
- Background and method
- Algorithm descriptions
- Validation results
- Convergence studies
- Discussion and future work

---

## 🔍 Quick Navigation

**I want to...**

### ...understand what the project does
→ **QUICK_REFERENCE.md** (What is this project section)  
→ **README.md** (overview section)

### ...build and run tests
→ **QUICK_REFERENCE.md** (Quick Start section)  
→ **CMakeLists.txt** (build configuration)

### ...understand the directory structure
→ **EXPLORATION.md** (section 2: Directory Structure)  
→ **ARCHITECTURE.md** (directory tree)

### ...understand the algorithm
→ **QUICK_REFERENCE.md** (Algorithm Pipeline section)  
→ **EXPLORATION.md** (section 7: Algorithm Pipeline)  
→ **ARCHITECTURE.md** (Algorithm Data Flow section)  
→ **docs/mimetic_voronoi_report.pdf** (full details)

### ...understand the code architecture
→ **ARCHITECTURE.md** (entire document)  
→ **EXPLORATION.md** (section 3: Key Source Files)

### ...understand how tests work
→ **EXPLORATION.md** (section 5: Test Files)  
→ **QUICK_REFERENCE.md** (Test Categories table)  
→ Read individual test files in `tests/`

### ...add a new test
→ **QUICK_REFERENCE.md** (Common Tasks → Add a New Test)  
→ **CMakeLists.txt** (see add_test pattern)

### ...understand convergence behavior
→ **EXPLORATION.md** (section 9: Convergence Performance)  
→ **QUICK_REFERENCE.md** (Convergence Rates section)  
→ **docs/high_order_hdiv_convergence.csv** (data)

### ...understand the dependencies
→ **EXPLORATION.md** (section 13: Key Dependencies)  
→ **QUICK_REFERENCE.md** (Dependencies table)  
→ **CMakeLists.txt** (find_package/find_library commands)

### ...trace code to manuscript
→ **docs/code_documentation.md** (Manuscript-to-Code Map)

### ...work with spherical geometry
→ **EXPLORATION.md** (section 7: Spherical Gnomonic Path)  
→ **ARCHITECTURE.md** (Spherical Gnomonic Path section)  
→ **README.md** (Geometry Backends section)

### ...work with high-order moments
→ **EXPLORATION.md** (section 7: High-Order Path)  
→ **README.md** (Higher-Order Edge Moments section)  
→ **ARCHITECTURE.md** (Split Basis & Moments section)

### ...troubleshoot a failing test
→ **QUICK_REFERENCE.md** (Common Tasks → Debug a Test)  
→ Test source code in `tests/`  
→ Run with `--verbose` flag

### ...generate convergence plots
→ **QUICK_REFERENCE.md** (Common Tasks → Generate Convergence Plots)  
→ `scripts/plot_high_order_hdiv_convergence.py`

### ...understand conservative edge transfer
→ **QUICK_REFERENCE.md** (Algorithm Pipeline section)  
→ **EXPLORATION.md** (section 7: Low-Order Path, section 7: Spherical Path)  
→ **ARCHITECTURE.md** (Low-Order Reconstruction & Transfer diagram)

### ...understand directed edge DOFs
→ **QUICK_REFERENCE.md** (Key Design Decisions table)  
→ **docs/code_documentation.md** (Directed Edge Convention section)  
→ **EXPLORATION.md** (section 8: Directed Cell-Local Edge DOFs)

---

## 📊 File Size & Complexity

| Document | Lines | Size | Complexity | Target Reader |
|----------|-------|------|-----------|---|
| QUICK_REFERENCE.md | 303 | 10.5 KB | Low | Anyone |
| EXPLORATION.md | 535 | 22.9 KB | Medium-High | Developers |
| ARCHITECTURE.md | 378 | 32.7 KB | High | Implementers |
| README.md | 440 | 15.2 KB | Medium | Mathematicians |
| CLAUDE.md | 142 | 7.2 KB | Low-Medium | AI assistants |
| code_documentation.md | 409 | Large | High | Deep dives |
| mimetic_voronoi_report.pdf | - | Large | Very High | Researchers |

---

## 🎯 Reading Paths

### Path 1: "I just want to build and run tests"
1. QUICK_REFERENCE.md (Quick Start section)
2. CMakeLists.txt (understand build)
3. Done!

### Path 2: "I want to understand the project"
1. QUICK_REFERENCE.md (entire document)
2. README.md (sections 1-3)
3. EXPLORATION.md (sections 1-2, 5)
4. Done!

### Path 3: "I want to understand the code"
1. QUICK_REFERENCE.md (Project Structure + Core Components)
2. ARCHITECTURE.md (entire document)
3. EXPLORATION.md (sections 3, 7, 8)
4. Source code in `include/` and `src/`
5. Done!

### Path 4: "I want to understand the algorithm"
1. QUICK_REFERENCE.md (Algorithm Pipeline)
2. EXPLORATION.md (sections 1, 7, 8)
3. ARCHITECTURE.md (Algorithm Data Flow section)
4. README.md (sections on geometry, reconstruction, transfer)
5. docs/mimetic_voronoi_report.pdf (full details)
6. Done!

### Path 5: "I want to modify the code"
1. QUICK_REFERENCE.md (entire document)
2. ARCHITECTURE.md (entire document)
3. EXPLORATION.md (sections 3, 8, 11)
4. docs/code_documentation.md (Reconstruction Path, Transfer Path, etc.)
5. Read relevant test files
6. MOAB documentation (if mesh operations)
7. Eigen documentation (if linear algebra)
8. Ready to code!

### Path 6: "I want to understand convergence/validation"
1. QUICK_REFERENCE.md (Convergence Rates, Test Categories)
2. EXPLORATION.md (sections 5, 9)
3. Run tests: `ctest --test-dir build --output-on-failure`
4. Check CSV files: docs/high_order_hdiv_convergence.csv
5. Run convergence study: `./build/high_order_hdiv_convergence_test output.csv`
6. Plot results: `python3 scripts/plot_high_order_hdiv_convergence.py output.csv figure.png`
7. Done!

---

## 🏗️ Code Organization Summary

```
Mimetic-SphPoly/
├── Core Library
│   ├── include/mimetic/mimetic.hpp (634 lines) ← PUBLIC API
│   └── src/mimetic.cpp (3680 lines) ← IMPLEMENTATION
│
├── Tests
│   ├── tests/*.cpp (13 test executables)
│   ├── tests/test_utils.hpp
│   └── tests/spherical_transfer_test_utils.hpp
│
├── Build
│   └── CMakeLists.txt
│
├── Documentation ⭐ NEW
│   ├── QUICK_REFERENCE.md ← START HERE
│   ├── EXPLORATION.md ← COMPREHENSIVE
│   ├── ARCHITECTURE.md ← IMPLEMENTATION DETAILS
│   ├── INDEX.md (this file) ← NAVIGATION
│   └── (existing docs below)
│
├── Original Documentation
│   ├── README.md (project overview)
│   ├── CLAUDE.md (working conventions)
│   ├── docs/code_documentation.md (traceability)
│   ├── docs/mimetic_voronoi_report.tex/pdf (manuscript)
│   └── docs/references.bib
│
├── Data & Scripts
│   ├── docs/high_order_hdiv_convergence.csv
│   ├── docs/spherical_high_order_hdiv_convergence.csv
│   ├── scripts/convergence_study.sh
│   └── scripts/plot_*.py
│
└── Build Artifacts
    └── build/
```

---

## ✅ What Was Explored

### Project Definition
✅ What it does (conservative mimetic polygon interpolation)  
✅ Scientific domain (climate/atmospheric modeling)  
✅ Core algorithm (Perot-Chartrand level-2 reconstruction)  
✅ Application context (mesh remapping)  

### Directory & Files
✅ Complete directory tree with descriptions  
✅ All 4 major directories (include, src, tests, scripts)  
✅ Documentation hierarchy  
✅ Build configuration  

### Source Code
✅ Header file (634 lines): API, types, classes  
✅ Implementation (3680 lines): all numerics  
✅ 13 test files: both low-order and high-order  
✅ Test utilities: planar and spherical  

### Build System
✅ CMake configuration  
✅ Dependency resolution  
✅ Test registration  
✅ Optional OpenMP support  

### Tests & Validation
✅ 13 CTest targets  
✅ Manufactured test fields (4 types)  
✅ Conservation tolerance (5.0e-13)  
✅ Convergence rates and benchmarks  

### Documentation
✅ README (mathematical background)  
✅ Technical manuscript (LaTeX)  
✅ Code traceability (manuscript-to-code)  
✅ Convergence data (CSV files)  

### Algorithms
✅ Low-order reconstruction (level-2 harmonic)  
✅ High-order moments (p=1,2,3)  
✅ Planar geometry  
✅ Spherical gnomonic charts  
✅ Edge transfer (direct and sparse)  
✅ Global conforming projection  

### Design Decisions
✅ Directed edge DOFs (cell-local)  
✅ Single translation unit  
✅ Centroid-relative coordinates  
✅ Basis scaling and conditioning  
✅ Solver philosophy (KKT minimum-energy)  
✅ Conservation tolerance strategy  

### Dependencies
✅ Eigen3 (linear algebra)  
✅ MOAB (mesh framework)  
✅ nanoflann (spatial indexing)  
✅ C++14 (language)  

### Limitations & Future Work
✅ Spherical constraints  
✅ Spatial search optimization  
✅ Directed DOF collapsing  
✅ Parallelism potential  
✅ Research vs. production status  

---

## 💡 Pro Tips

1. **Start small**: Begin with QUICK_REFERENCE.md, not the full manuscript
2. **Learn by doing**: Build the code, run tests, see what happens
3. **Use grep**: Find functions by name: `grep -r "function_name" src/`
4. **Read tests**: Test files show real usage patterns
5. **Check git log**: Recent commits show what changed and why
6. **Explore incrementally**: Don't try to understand everything at once
7. **Use convergence data**: CSV files show empirical behavior
8. **Study the header**: mimetic.hpp is the contract with users

---

## 📞 Questions?

- **"What does X do?"** → Search ARCHITECTURE.md for function name
- **"Where is X implemented?"** → Search src/mimetic.cpp
- **"How does X work?"** → Read EXPLORATION.md section 7
- **"Why does X?"** → Check EXPLORATION.md section 8 (Design Decisions)
- **"How do I X?"** → Check QUICK_REFERENCE.md (Common Tasks)
- **"What's X tolerance?"** → QUICK_REFERENCE.md or EXPLORATION.md
- **"How's the convergence?"** → EXPLORATION.md section 9 or QUICK_REFERENCE.md table

---

Generated: 2024  
Documentation created by AI-assisted exploration  
Ready for developers, researchers, and maintainers! 🚀
