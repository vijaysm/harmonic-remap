#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Test fields (mathematical formulas — essential for verifying correctness):
//   A: u=(1+2x+2y, 1-2y+2x), div=0, exercises all harmonic coefficients
//   B: u=(sin(pi*x)cos(pi*y), -cos(pi*x)sin(pi*y)), div=0, non-polynomial
//   C: u=(x^2, -y^2), div=2x-2y, spatially varying divergence
//   D: u=(exp(x)sin(y), exp(x)cos(y)), div=0, exponential non-polynomial

constexpr double PI = 3.14159265358979323846;

// 10-point Gauss-Legendre nodes/weights on [-1,1] (Abramowitz & Stegun 25.4)
struct GaussPoint { double x; double w; };

static const GaussPoint gauss10_table[10] = {
    {-0.9739065285171717, 0.0666713443086881},
    {-0.8650633666889845, 0.1494513491505806},
    {-0.6794095682990244, 0.2190863625159820},
    {-0.4333953941292472, 0.2692667193099963},
    {-0.1488743389816312, 0.2955242247147529},
    { 0.1488743389816312, 0.2955242247147529},
    { 0.4333953941292472, 0.2692667193099963},
    { 0.6794095682990244, 0.2190863625159820},
    { 0.8650633666889845, 0.1494513491505806},
    { 0.9739065285171717, 0.0666713443086881},
};

template <typename Func>
double integrate_edge_highorder(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& func)
{
    const double length = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        sum += gauss10_table[i].w * func(mid + gauss10_table[i].x * half_delta);
    }
    return 0.5 * length * sum;
}

namespace {

// u_A = (1+2x+2y, 1-2y+2x): d=0, a1=1, b1=1, a2=1, b2=1
Eigen::Vector2d field_A(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(1.0 + 2.0 * p.x() + 2.0 * p.y(),
                           1.0 - 2.0 * p.y() + 2.0 * p.x());
}

// u_B = (sin(pi*x)cos(pi*y), -cos(pi*x)sin(pi*y)): div=0
Eigen::Vector2d field_B(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(std::sin(PI * p.x()) * std::cos(PI * p.y()),
                          -std::cos(PI * p.x()) * std::sin(PI * p.y()));
}

// u_C = (x^2, -y^2): div=2x-2y
Eigen::Vector2d field_C(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(p.x() * p.x(), -p.y() * p.y());
}

// u_D = (exp(x)sin(y), exp(x)cos(y)): div=0
Eigen::Vector2d field_D(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(std::exp(p.x()) * std::sin(p.y()),
                           std::exp(p.x()) * std::cos(p.y()));
}

using FieldFunc = Eigen::Vector2d(*)(const Eigen::Vector2d&);

struct TestField {
    std::string name;
    FieldFunc func;
    bool in_space;
};

static const TestField all_test_fields[] = {
    {"A: harmonic_exact",   field_A, true},
    {"B: sincos_divfree",   field_B, false},
    {"C: quad_vardiv",      field_C, false},
    {"D: exp_divfree",      field_D, false},
};

double exact_directed_edge_flux(const Eigen::Vector2d& a, const Eigen::Vector2d& b, FieldFunc field)
{
    const Eigen::Vector2d delta = b - a;
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    const double length = delta.norm();
    return integrate_edge_highorder(a, b, [&](const Eigen::Vector2d& p) {
        return field(p).dot(normal / length);
    });
}

void set_source_fluxes_highorder(moab::Core& mb,
                                 const moab::Tag source_flux_tag,
                                 const moab::EntityHandle polygon,
                                 FieldFunc field)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (const mimetic::LocalEdge& edge : edges) {
        const double flux = integrate_edge_highorder(edge.a, edge.b, [&](const Eigen::Vector2d& p) {
            return field(p + poly.centroid).dot(edge.outward_normal);
        });
        mimetic::check_moab(mb.tag_set_data(source_flux_tag, &edge.handle, 1, &flux),
                            "Failed to set source flux");
    }
}

double halton(int index, int base)
{
    double result = 0.0;
    double f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i /= base;
        f /= base;
    }
    return result;
}

std::vector<Eigen::Vector2d> halton_seeds(int n)
{
    std::vector<Eigen::Vector2d> seeds;
    seeds.reserve(n);
    for (int i = 1; i <= n; ++i) {
        seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                           0.05 + 0.90 * halton(i, 3));
    }
    return seeds;
}

std::vector<Eigen::Vector2d> clip_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                                const Eigen::Vector2d& normal,
                                                const double offset)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) return output;

    auto inside = [&](const Eigen::Vector2d& p) { return normal.dot(p) <= offset + 1.0e-13; };
    auto intersect = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        const double da = normal.dot(a) - offset;
        const double db = normal.dot(b) - offset;
        return a + (da / (da - db)) * (b - a);
    };

    Eigen::Vector2d previous = polygon.back();
    bool previous_inside = inside(previous);
    for (const Eigen::Vector2d& current : polygon) {
        const bool current_inside = inside(current);
        if (current_inside) {
            if (!previous_inside) output.push_back(intersect(previous, current));
            output.push_back(current);
        } else if (previous_inside) {
            output.push_back(intersect(previous, current));
        }
        previous = current;
        previous_inside = current_inside;
    }
    return output;
}

std::vector<Eigen::Vector2d> voronoi_cell_polygon(const Eigen::Vector2d& seed,
                                                   const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<Eigen::Vector2d> cell = {
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
        Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(0.0, 1.0),
    };
    for (const Eigen::Vector2d& other : seeds) {
        if ((other - seed).squaredNorm() < 1.0e-24) continue;
        const Eigen::Vector2d normal = 2.0 * (other - seed);
        const double offset = other.squaredNorm() - seed.squaredNorm();
        cell = clip_by_halfplane(cell, normal, offset);
        if (cell.size() < 3) return {};
    }
    return cell;
}

double polygon_area(const std::vector<Eigen::Vector2d>& points)
{
    return std::abs(mimetic::signed_area(points));
}

struct CellInfo {
    moab::EntityHandle handle;
    std::vector<Eigen::Vector2d> points;
};

std::vector<CellInfo> create_quad_mesh(moab::Core& mb, int nx, int ny)
{
    std::vector<CellInfo> cells;
    const double dx = 1.0 / nx;
    const double dy = 1.0 / ny;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x0 = i * dx;
            const double y0 = j * dy;
            const std::vector<Eigen::Vector2d> pts = {
                Eigen::Vector2d(x0, y0),
                Eigen::Vector2d(x0 + dx, y0),
                Eigen::Vector2d(x0 + dx, y0 + dy),
                Eigen::Vector2d(x0, y0 + dy),
            };
            cells.push_back(CellInfo{mimetic::create_polygon(mb, pts), pts});
        }
    }
    return cells;
}

std::vector<CellInfo> create_voronoi_mesh(moab::Core& mb, const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<CellInfo> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        std::vector<Eigen::Vector2d> pts = voronoi_cell_polygon(seed, seeds);
        if (pts.size() < 3 || polygon_area(pts) < 1.0e-12) continue;
        cells.push_back(CellInfo{mimetic::create_polygon(mb, pts), pts});
    }
    return cells;
}

std::vector<moab::EntityHandle> get_handles(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> handles;
    handles.reserve(cells.size());
    for (const CellInfo& c : cells) handles.push_back(c.handle);
    return handles;
}

struct ErrorMetrics {
    double l2_rel;
    double linf;
    double conservation;
    int n_target_edges;
    double h_effective;
};

// ∫_Ω div(u) dA via Gauss divergence theorem: equals ∮_∂Ω u·n ds on [0,1]²
double exact_divergence_integral(FieldFunc field)
{
    double flux = 0.0;
    // Bottom edge y=0: n=(0,-1), from (0,0) to (1,0)
    flux += integrate_edge_highorder(Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 0),
        [&](const Eigen::Vector2d& p) { return -field(p).y(); });
    // Right edge x=1: n=(1,0), from (1,0) to (1,1)
    flux += integrate_edge_highorder(Eigen::Vector2d(1, 0), Eigen::Vector2d(1, 1),
        [&](const Eigen::Vector2d& p) { return field(p).x(); });
    // Top edge y=1: n=(0,1), from (1,1) to (0,1)
    flux += integrate_edge_highorder(Eigen::Vector2d(1, 1), Eigen::Vector2d(0, 1),
        [&](const Eigen::Vector2d& p) { return field(p).y(); });
    // Left edge x=0: n=(-1,0), from (0,1) to (0,0)
    flux += integrate_edge_highorder(Eigen::Vector2d(0, 1), Eigen::Vector2d(0, 0),
        [&](const Eigen::Vector2d& p) { return -field(p).x(); });
    return flux;
}

ErrorMetrics compute_transfer_errors(
    moab::Core& mb,
    mimetic::MimeticInterpolator& interpolator,
    const std::vector<CellInfo>& source_cells,
    const std::vector<CellInfo>& target_cells,
    FieldFunc field,
    double h_eff)
{
    for (const CellInfo& src : source_cells) {
        set_source_fluxes_highorder(mb, interpolator.source_flux_tag(), src.handle, field);
        interpolator.reconstruct_source_polygon(src.handle);
    }

    const std::vector<moab::EntityHandle> src_handles = get_handles(source_cells);
    const std::vector<moab::EntityHandle> tgt_handles = get_handles(target_cells);
    const mimetic::EdgeTransferResult transfer =
        interpolator.transfer_source_to_target_edges(src_handles, tgt_handles);

    double l2_num = 0.0, l2_den = 0.0, linf = 0.0;
    double total_cell_divergence = 0.0;
    std::size_t dof = 0;

    for (const CellInfo& tgt : target_cells) {
        const mimetic::LocalPolygon tgt_poly = mimetic::local_polygon(mb, tgt.handle);
        const std::vector<mimetic::LocalEdge> tgt_edges = mimetic::local_edges(mb, tgt_poly);
        double cell_flux_sum = 0.0;

        for (const mimetic::LocalEdge& edge : tgt_edges) {
            const Eigen::Vector2d a = tgt_poly.centroid + edge.a;
            const Eigen::Vector2d b = tgt_poly.centroid + edge.b;

            const double exact = exact_directed_edge_flux(a, b, field);
            const double computed = transfer.target_fluxes[dof];
            const double err = std::abs(computed - exact);

            l2_num += err * err;
            l2_den += exact * exact;
            if (err > linf) linf = err;

            cell_flux_sum += computed;
            ++dof;
        }
        total_cell_divergence += cell_flux_sum;
    }

    const double exact_div_integral = exact_divergence_integral(field);
    const double conservation_err = std::abs(total_cell_divergence - exact_div_integral);
    const double l2_rel = (l2_den > 1.0e-30) ? std::sqrt(l2_num / l2_den) : std::sqrt(l2_num);

    return ErrorMetrics{l2_rel, linf, conservation_err,
                        static_cast<int>(transfer.target_edges.size()), h_eff};
}

double convergence_rate(double e_coarse, double e_fine, double h_coarse, double h_fine)
{
    if (e_coarse < 1.0e-15 || e_fine < 1.0e-15) return 0.0;
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

void print_table_header(const std::string& domain_label)
{
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  Domain: " << domain_label << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left << std::setw(24) << "Field"
              << std::right
              << std::setw(10) << "h"
              << std::setw(14) << "L2_rel"
              << std::setw(14) << "Linf"
              << std::setw(14) << "Conserv"
              << std::setw(10) << "Rate_L2"
              << std::setw(10) << "Rate_Linf"
              << std::setw(8) << "Edges"
              << "\n";
    std::cout << std::string(100, '-') << "\n";
}

void print_table_row(const std::string& field_name, const ErrorMetrics& m,
                     double rate_l2, double rate_linf)
{
    std::cout << std::left << std::setw(24) << field_name
              << std::right << std::scientific << std::setprecision(2)
              << std::setw(10) << m.h_effective
              << std::setw(14) << m.l2_rel
              << std::setw(14) << m.linf
              << std::setw(14) << m.conservation;
    if (rate_l2 > 0.01) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rate_l2;
    } else {
        std::cout << std::setw(10) << "---";
    }
    if (rate_linf > 0.01) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rate_linf;
    } else {
        std::cout << std::setw(10) << "---";
    }
    std::cout << std::setw(8) << m.n_target_edges << "\n";
}

enum class DomainType { QuadQuad, VoronoiVoronoi, VoronoiQuad };

struct RefinementLevel {
    int n_source;
    int n_target;
};

bool run_refinement_study(const std::string& domain_label,
                          DomainType domain_type,
                          const std::vector<RefinementLevel>& levels)
{
    bool all_ok = true;

    for (const TestField& tf : all_test_fields) {
        std::vector<ErrorMetrics> metrics;

        for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
            moab::Core mb;
            mimetic::MimeticInterpolator interpolator(mb);

            const int ns = levels[lvl].n_source;
            const int nt = levels[lvl].n_target;

            std::vector<CellInfo> source_cells, target_cells;
            double h_eff = 0.0;

            switch (domain_type) {
            case DomainType::QuadQuad: {
                source_cells = create_quad_mesh(mb, ns, ns);
                target_cells = create_quad_mesh(mb, nt, nt);
                h_eff = 1.0 / std::max(ns, nt);
                break;
            }
            case DomainType::VoronoiVoronoi: {
                std::vector<Eigen::Vector2d> src_seeds = halton_seeds(ns);
                std::vector<Eigen::Vector2d> tgt_seeds;
                for (int i = ns + 1; i <= ns + nt; ++i) {
                    tgt_seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                                           0.05 + 0.90 * halton(i, 3));
                }
                source_cells = create_voronoi_mesh(mb, src_seeds);
                target_cells = create_voronoi_mesh(mb, tgt_seeds);
                h_eff = 1.0 / std::sqrt(static_cast<double>(std::max(ns, nt)));
                break;
            }
            case DomainType::VoronoiQuad: {
                std::vector<Eigen::Vector2d> src_seeds = halton_seeds(ns);
                source_cells = create_voronoi_mesh(mb, src_seeds);
                target_cells = create_quad_mesh(mb, nt, nt);
                h_eff = 1.0 / std::sqrt(static_cast<double>(std::max(ns, nt)));
                break;
            }
            }

            if (source_cells.empty() || target_cells.empty()) {
                std::cerr << "  [SKIP] Empty mesh at level " << lvl << "\n";
                continue;
            }

            metrics.push_back(compute_transfer_errors(mb, interpolator,
                                                      source_cells, target_cells,
                                                      tf.func, h_eff));
        }

        if (metrics.empty()) continue;

        print_table_row(tf.name, metrics[0], 0.0, 0.0);
        for (std::size_t i = 1; i < metrics.size(); ++i) {
            double rate_l2 = convergence_rate(metrics[i-1].l2_rel, metrics[i].l2_rel,
                                              metrics[i-1].h_effective, metrics[i].h_effective);
            double rate_linf = convergence_rate(metrics[i-1].linf, metrics[i].linf,
                                                metrics[i-1].h_effective, metrics[i].h_effective);
            print_table_row(tf.name, metrics[i], rate_l2, rate_linf);
        }

        if (metrics.size() >= 3) {
            double avg_rate = 0.0;
            int rate_count = 0;
            for (std::size_t i = 1; i < metrics.size(); ++i) {
                double r = convergence_rate(metrics[i-1].l2_rel, metrics[i].l2_rel,
                                            metrics[i-1].h_effective, metrics[i].h_effective);
                if (r > 0.01) { avg_rate += r; ++rate_count; }
            }
            if (rate_count > 0) avg_rate /= rate_count;

            // Quad->quad should achieve O(h^2); Voronoi cases O(h^1) minimum
            double min_expected_rate = (domain_type == DomainType::QuadQuad) ? 1.8 : 0.8;
            if (avg_rate < min_expected_rate && metrics.back().l2_rel > 1.0e-13) {
                std::cout << "  [FAIL] " << tf.name
                          << " avg L2 rate = " << std::fixed << std::setprecision(2) << avg_rate
                          << " (expected >= " << min_expected_rate
                          << ") on " << domain_label << "\n";
                all_ok = false;
            }
        }

        std::cout << "\n";
    }

    return all_ok;
}

}  // namespace

int main()
{
    try {
        std::cout << "=== Mimetic Remapping Convergence Validation Test ===\n";
        std::cout << "Fields: A (harmonic exact), B (sincos divfree), "
                  << "C (quad vardiv), D (exp divfree)\n";
        std::cout << "Domains: quad->quad, voronoi->voronoi, voronoi->quad\n\n";

        bool ok = true;

        {
            print_table_header("quad -> quad");
            std::vector<RefinementLevel> levels = {
                {4, 5}, {8, 9}, {16, 17}, {32, 33}
            };
            ok = run_refinement_study("quad->quad", DomainType::QuadQuad, levels) && ok;
        }

        {
            print_table_header("voronoi -> voronoi");
            // h ~ 1/sqrt(N), so doubling N halves h roughly by sqrt(2)
            // Use (src, tgt) pairs where both refine and src != tgt
            std::vector<RefinementLevel> levels = {
                {25, 30}, {64, 72}, {144, 160}, {289, 306}
            };
            ok = run_refinement_study("voronoi->voronoi", DomainType::VoronoiVoronoi, levels) && ok;
        }

        {
            print_table_header("voronoi -> quad");
            // Source Voronoi and target quad both refine; h ~ 1/sqrt(N_src) ~ 1/N_tgt
            std::vector<RefinementLevel> levels = {
                {25, 5}, {100, 10}, {225, 15}, {400, 20}
            };
            ok = run_refinement_study("voronoi->quad", DomainType::VoronoiQuad, levels) && ok;
        }

        if (!ok) {
            std::cout << "\n[FAILED] Some convergence checks did not pass.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] All convergence and exact recovery checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
