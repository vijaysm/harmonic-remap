#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
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

// u_E = (sin(2*pi*x)*sin(pi*y), cos(pi*x)*sin(2*pi*y)):
//   div = 2*pi*cos(2*pi*x)*sin(pi*y) + 2*pi*cos(pi*x)*cos(2*pi*y)
// Highly oscillating divergence with O(1) magnitude variations.
Eigen::Vector2d field_E(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(std::sin(2 * PI * p.x()) * std::sin(PI * p.y()),
                           std::cos(PI * p.x()) * std::sin(2 * PI * p.y()));
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
                                 mimetic::MimeticInterpolator& interpolator,
                                 const moab::EntityHandle polygon,
                                 FieldFunc field)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const mimetic::LocalEdge& edge = edges[edge_index];
        const double flux = integrate_edge_highorder(edge.a, edge.b, [&](const Eigen::Vector2d& p) {
            return field(p + poly.centroid).dot(edge.outward_normal);
        });
        interpolator.set_source_edge_flux(polygon, edge_index, flux);
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
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        polygons.push_back(cell.handle);
    }
    mimetic::merge_polygon_vertices(mb, polygons);
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
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        polygons.push_back(cell.handle);
    }
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<moab::EntityHandle> get_handles(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> handles;
    handles.reserve(cells.size());
    for (const CellInfo& c : cells) handles.push_back(c.handle);
    return handles;
}


void dump_mesh(const std::string& filename, moab::Core& mb, const std::vector<CellInfo>& cells, const std::vector<double>& values) {
    std::ofstream out(filename);
    if (!out) return;
    for (size_t i = 0; i < cells.size(); ++i) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cells[i].handle);
        out << poly.centroid.x() << " " << poly.centroid.y() << " " << values[i] << " " << poly.points.size();
        for (const auto& v : poly.points) {
            out << " " << (poly.centroid.x() + v.x()) << " " << (poly.centroid.y() + v.y());
        }
        out << "\n";
    }
}

void dump_edge_segments(const std::string& filename,
                        const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& segments,
                        const std::vector<double>& values,
                        const std::vector<int>& multiplicity)
{
    std::ofstream out(filename);
    if (!out) return;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        out << segments[i].first.x() << " " << segments[i].first.y() << " "
            << segments[i].second.x() << " " << segments[i].second.y() << " "
            << values[i] << " " << multiplicity[i] << "\n";
    }
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
    const std::string& prefix,
    moab::Core& mb,
    mimetic::MimeticInterpolator& interpolator,
    const std::vector<CellInfo>& source_cells,
    const std::vector<CellInfo>& target_cells,
    FieldFunc field,
    double h_eff)
{
    for (const CellInfo& src : source_cells) {
        set_source_fluxes_highorder(mb, interpolator, src.handle, field);
        interpolator.reconstruct_source_polygon(src.handle);
    }

    const std::vector<moab::EntityHandle> src_handles = get_handles(source_cells);
    const std::vector<moab::EntityHandle> tgt_handles = get_handles(target_cells);
    const mimetic::EdgeTransferResult transfer =
        interpolator.transfer_source_to_target_edges(src_handles, tgt_handles);
    const mimetic::ConformingEdgeTransferResult conforming =
        interpolator.project_target_fluxes_to_hdiv_conforming(src_handles, tgt_handles, transfer);

    
    double l2_num = 0.0, l2_den = 0.0, linf = 0.0;
    double total_cell_divergence = 0.0;
    std::size_t dof = 0;

    std::vector<double> src_values;
    if (!prefix.empty()) {
        for (const CellInfo& src : source_cells) {
            const mimetic::LocalPolygon src_poly = mimetic::local_polygon(mb, src.handle);
            const std::vector<mimetic::LocalEdge> src_edges = mimetic::local_edges(mb, src_poly);
            double src_div = 0.0;
            for (const mimetic::LocalEdge& edge : src_edges) {
                const Eigen::Vector2d a = src_poly.centroid + edge.a;
                const Eigen::Vector2d b = src_poly.centroid + edge.b;
                src_div += exact_directed_edge_flux(a, b, field);
            }
            src_values.push_back(src_div / src_poly.area);
        }
    }

    std::vector<double> tgt_values;
    std::vector<double> tgt_exact_values;
    std::vector<double> tgt_conforming_values;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> unique_segments(conforming.unique_edge_fluxes.size());
    std::vector<double> raw_edge_jump(conforming.unique_edge_fluxes.size(), 0.0);
    std::vector<double> conforming_edge_jump(conforming.unique_edge_fluxes.size(), 0.0);
    std::vector<int> edge_multiplicity(conforming.unique_edge_fluxes.size(), 0);
    std::vector<std::vector<double>> raw_edge_values(conforming.unique_edge_fluxes.size());
    std::vector<std::vector<double>> conforming_edge_values(conforming.unique_edge_fluxes.size());

    for (const CellInfo& tgt : target_cells) {
        const mimetic::LocalPolygon tgt_poly = mimetic::local_polygon(mb, tgt.handle);
        const std::vector<mimetic::LocalEdge> tgt_edges = mimetic::local_edges(mb, tgt_poly);
        double cell_flux_sum = 0.0;
        double conforming_cell_flux_sum = 0.0;
        double exact_cell_flux_sum = 0.0;

        for (const mimetic::LocalEdge& edge : tgt_edges) {
            const Eigen::Vector2d a = tgt_poly.centroid + edge.a;
            const Eigen::Vector2d b = tgt_poly.centroid + edge.b;
            const std::size_t unique = conforming.target_edge_to_unique[dof];
            if (edge_multiplicity[unique] == 0) {
                if (conforming.target_edge_signs[dof] >= 0) {
                    unique_segments[unique] = std::make_pair(a, b);
                } else {
                    unique_segments[unique] = std::make_pair(b, a);
                }
            }
            ++edge_multiplicity[unique];
            raw_edge_values[unique].push_back(transfer.target_fluxes[dof]);
            conforming_edge_values[unique].push_back(conforming.target_fluxes[dof]);

            const double exact = exact_directed_edge_flux(a, b, field);
            const double computed = transfer.target_fluxes[dof];
            const double err = std::abs(computed - exact);

            l2_num += err * err;
            l2_den += exact * exact;
            if (err > linf) linf = err;


            cell_flux_sum += computed;
            conforming_cell_flux_sum += conforming.target_fluxes[dof];
            exact_cell_flux_sum += exact;
            ++dof;
        }
        total_cell_divergence += cell_flux_sum;
        if (!prefix.empty()) {
            tgt_values.push_back(cell_flux_sum / tgt_poly.area);
            tgt_exact_values.push_back(exact_cell_flux_sum / tgt_poly.area);
            tgt_conforming_values.push_back(conforming_cell_flux_sum / tgt_poly.area);
        }
    }

    if (!prefix.empty()) {
        dump_mesh(prefix + "_source.txt", mb, source_cells, src_values);
        dump_mesh(prefix + "_target.txt", mb, target_cells, tgt_values);
        dump_mesh(prefix + "_target_exact.txt", mb, target_cells, tgt_exact_values);
        dump_mesh(prefix + "_target_conforming.txt", mb, target_cells, tgt_conforming_values);
        for (std::size_t unique = 0; unique < raw_edge_values.size(); ++unique) {
            if (raw_edge_values[unique].size() == 2) {
                raw_edge_jump[unique] = raw_edge_values[unique][0] + raw_edge_values[unique][1];
                conforming_edge_jump[unique] =
                    conforming_edge_values[unique][0] + conforming_edge_values[unique][1];
            }
        }
        dump_edge_segments(prefix + "_raw_edge_jump.txt", unique_segments, raw_edge_jump, edge_multiplicity);
        dump_edge_segments(prefix + "_conforming_edge_jump.txt", unique_segments, conforming_edge_jump, edge_multiplicity);
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

            std::string prefix = "";
            if (lvl == 0) {
                prefix = "vis_" + domain_label + "_" + tf.name;
            }
            metrics.push_back(compute_transfer_errors(prefix, mb, interpolator,
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
            double min_expected_rate = (domain_type == DomainType::QuadQuad) ? 1.8 : 0.2;
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

void set_source_edge_moments_highorder(moab::Core& mb,
                                       mimetic::PlanarMomentInterpolator& interpolator,
                                       const moab::EntityHandle polygon,
                                       FieldFunc field,
                                       int order)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const mimetic::LocalEdge& edge = edges[edge_index];
        const Eigen::Vector2d a_abs = poly.centroid + edge.a;
        const Eigen::Vector2d b_abs = poly.centroid + edge.b;
        const Eigen::Vector2d delta = b_abs - a_abs;
        const double length = delta.norm();
        const Eigen::Vector2d normal(delta.y(), -delta.x());
        const double denom = delta.squaredNorm();
        std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
        for (int degree = 0; degree <= order; ++degree) {
            moments[degree] = integrate_edge_highorder(a_abs, b_abs, [&](const Eigen::Vector2d& p) {
                const double t = 2.0 * (p - a_abs).dot(delta) / denom - 1.0;
                double Lm = 1.0;
                if (degree == 1) Lm = t;
                else if (degree == 2) Lm = 0.5 * (3.0 * t * t - 1.0);
                else if (degree == 3) Lm = 0.5 * (5.0 * t * t * t - 3.0 * t);
                else if (degree > 3) {
                    double p_nm2 = 1.0, p_nm1 = t;
                    for (int n = 2; n <= degree; ++n) {
                        Lm = ((2.0 * n - 1.0) * t * p_nm1 - (n - 1.0) * p_nm2) / n;
                        p_nm2 = p_nm1;
                        p_nm1 = Lm;
                    }
                }
                return field(p).dot(normal / length) * Lm;
            });
        }
        interpolator.set_source_edge_moments(polygon, edge_index, moments);
    }
}

void set_source_cell_moments_highorder(moab::Core& mb,
                                       mimetic::PlanarMomentInterpolator& interpolator,
                                       const moab::EntityHandle polygon,
                                       FieldFunc field,
                                       int cell_order)
{
    if (cell_order < 0) return;
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    std::vector<Eigen::Vector2d> moments;
    for (int total_degree = 0; total_degree <= cell_order; ++total_degree) {
        for (int a = total_degree; a >= 0; --a) {
            const int b = total_degree - a;
            double mx = 0.0, my = 0.0;
            for (const mimetic::LocalEdge& edge : edges) {
                const Eigen::Vector2d origin(0, 0);
                // Triangle fan integration: origin, edge.a, edge.b (centroid-relative)
                auto triangle_integrate = [&](auto func) {
                    // 7-point symmetric rule
                    const double a1 = 0.4701420641051151, b1 = 1.0 - 2.0*a1, w1 = 0.1323941527885062;
                    const double a2 = 0.1012865073234563, b2 = 1.0 - 2.0*a2, w2 = 0.1259391805448271;
                    struct Pt { double w, u, v, wb; };
                    const Pt pts[7] = {
                        {0.225, 1.0/3.0, 1.0/3.0, 1.0/3.0},
                        {w1, a1, a1, b1}, {w1, a1, b1, a1}, {w1, b1, a1, a1},
                        {w2, a2, a2, b2}, {w2, a2, b2, a2}, {w2, b2, a2, a2}
                    };
                    const Eigen::Vector2d e0 = origin, e1 = edge.a, e2 = edge.b;
                    double s2a = (e1-e0).x()*(e2-e0).y() - (e1-e0).y()*(e2-e0).x();
                    double area = 0.5 * std::abs(s2a);
                    double sum = 0.0;
                    for (int i = 0; i < 7; ++i) {
                        sum += pts[i].w * func(pts[i].u*e0 + pts[i].v*e1 + pts[i].wb*e2);
                    }
                    return area * sum;
                };
                mx += triangle_integrate([&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).x() * std::pow(p.x(), a) * std::pow(p.y(), b);
                });
                my += triangle_integrate([&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).y() * std::pow(p.x(), a) * std::pow(p.y(), b);
                });
            }
            moments.push_back(Eigen::Vector2d(mx, my));
        }
    }
    interpolator.set_source_cell_vector_moments(polygon, moments);
}

void compute_order_comparison(const std::string& prefix,
                              moab::Core& mb,
                              const std::vector<CellInfo>& source_cells,
                              const std::vector<CellInfo>& target_cells,
                              FieldFunc field)
{
    const std::vector<moab::EntityHandle> src_handles = get_handles(source_cells);
    const std::vector<moab::EntityHandle> tgt_handles = get_handles(target_cells);

    // Compute source exact divergence
    std::vector<double> src_values;
    for (const CellInfo& src : source_cells) {
        const mimetic::LocalPolygon src_poly = mimetic::local_polygon(mb, src.handle);
        const std::vector<mimetic::LocalEdge> src_edges = mimetic::local_edges(mb, src_poly);
        double src_div = 0.0;
        for (const mimetic::LocalEdge& edge : src_edges) {
            src_div += exact_directed_edge_flux(src_poly.centroid + edge.a, src_poly.centroid + edge.b, field);
        }
        src_values.push_back(src_div / src_poly.area);
    }

    // Target exact divergence
    std::vector<double> tgt_exact_values;
    for (const CellInfo& tgt : target_cells) {
        const mimetic::LocalPolygon tgt_poly = mimetic::local_polygon(mb, tgt.handle);
        const std::vector<mimetic::LocalEdge> tgt_edges = mimetic::local_edges(mb, tgt_poly);
        double exact_div = 0.0;
        for (const mimetic::LocalEdge& edge : tgt_edges) {
            exact_div += exact_directed_edge_flux(tgt_poly.centroid + edge.a, tgt_poly.centroid + edge.b, field);
        }
        tgt_exact_values.push_back(exact_div / tgt_poly.area);
    }

    // p=1: low-order mimetic transfer
    {
        mimetic::MimeticInterpolator interpolator(mb);
        for (const CellInfo& src : source_cells) {
            set_source_fluxes_highorder(mb, interpolator, src.handle, field);
            interpolator.reconstruct_source_polygon(src.handle);
        }
        const mimetic::EdgeTransferResult transfer =
            interpolator.transfer_source_to_target_edges(src_handles, tgt_handles);

        std::vector<double> tgt_p1_values;
        std::size_t dof = 0;
        for (const CellInfo& tgt : target_cells) {
            const mimetic::LocalPolygon tgt_poly = mimetic::local_polygon(mb, tgt.handle);
            const std::vector<mimetic::LocalEdge> tgt_edges = mimetic::local_edges(mb, tgt_poly);
            double cell_flux = 0.0;
            for (std::size_t e = 0; e < tgt_edges.size(); ++e, ++dof) {
                cell_flux += transfer.target_fluxes[dof];
            }
            tgt_p1_values.push_back(cell_flux / tgt_poly.area);
        }
        dump_mesh(prefix + "_target_p1.txt", mb, target_cells, tgt_p1_values);
    }

    // p=3: high-order moment transfer
    {
        moab::Core mb3;
        // Recreate meshes in a fresh MOAB instance to avoid tag conflicts
        std::vector<CellInfo> src3, tgt3;
        for (const CellInfo& c : source_cells) {
            src3.push_back(CellInfo{mimetic::create_polygon(mb3, c.points), c.points});
        }
        {
            std::vector<moab::EntityHandle> handles;
            for (const CellInfo& c : src3) handles.push_back(c.handle);
            mimetic::merge_polygon_vertices(mb3, handles);
        }
        for (const CellInfo& c : target_cells) {
            tgt3.push_back(CellInfo{mimetic::create_polygon(mb3, c.points), c.points});
        }
        {
            std::vector<moab::EntityHandle> handles;
            for (const CellInfo& c : tgt3) handles.push_back(c.handle);
            mimetic::merge_polygon_vertices(mb3, handles);
        }

        mimetic::PlanarMomentInterpolator interpolator3(mb3);
        mimetic::MomentMethodOptions options;
        options.edge_moment_order = 3;
        options.cell_moment_order = 2;
        options.quadrature_points = 10;
        options.regularization = 1.0e-12;
        options.exact_constraints = false;

        const std::vector<moab::EntityHandle> src3_handles = get_handles(src3);
        const std::vector<moab::EntityHandle> tgt3_handles = get_handles(tgt3);

        for (const CellInfo& src : src3) {
            set_source_edge_moments_highorder(mb3, interpolator3, src.handle, field, 3);
            set_source_cell_moments_highorder(mb3, interpolator3, src.handle, field, 2);
            interpolator3.reconstruct_source_polygon(src.handle, options);
        }

        const mimetic::EdgeMomentTransferResult transfer3 =
            interpolator3.transfer_source_to_target_edge_moments(src3_handles, tgt3_handles, 3);

        std::vector<double> tgt_p3_values;
        std::size_t dof = 0;
        for (const CellInfo& tgt : tgt3) {
            const mimetic::LocalPolygon tgt_poly = mimetic::local_polygon(mb3, tgt.handle);
            const std::vector<mimetic::LocalEdge> tgt_edges = mimetic::local_edges(mb3, tgt_poly);
            double cell_flux = 0.0;
            for (std::size_t e = 0; e < tgt_edges.size(); ++e, ++dof) {
                cell_flux += transfer3.target_moments[dof][0]; // zeroth moment = flux
            }
            tgt_p3_values.push_back(cell_flux / tgt_poly.area);
        }
        dump_mesh(prefix + "_target_p3.txt", mb3, tgt3, tgt_p3_values);
    }

    dump_mesh(prefix + "_source.txt", mb, source_cells, src_values);
    dump_mesh(prefix + "_target_exact.txt", mb, target_cells, tgt_exact_values);

    std::cout << "  Order comparison dumped: " << prefix << "\n";
}

// Forward declarations for functions defined later in the file.
std::vector<moab::EntityHandle> generate_latlon_grid(moab::Core& mb, int nlon, int nlat);

/// Dump cells in equirectangular (plate carrée) projection: (lon, lat) in degrees.
/// No area distortion, no polar singularity, linear mapping.
void dump_mercator_mesh(const std::string& filename, moab::Core& mb,
                        const std::vector<moab::EntityHandle>& cells,
                        const std::vector<double>& values)
{
    std::ofstream out(filename);
    if (!out) return;
    auto to_lonlat = [](const Eigen::Vector3d& p3) -> Eigen::Vector2d {
        const Eigen::Vector3d u = p3.normalized();
        const double lon = std::atan2(u.y(), u.x()) * 180.0 / PI;
        const double lat = std::asin(std::max(-1.0, std::min(1.0, u.z()))) * 180.0 / PI;
        return Eigen::Vector2d(lon, lat);
    };
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const moab::EntityHandle* conn = nullptr;
        int nv = 0;
        mimetic::check_moab(mb.get_connectivity(cells[i], conn, nv), "get conn");
        if (nv < 3) continue;
        std::vector<Eigen::Vector2d> projected;
        for (int v = 0; v < nv; ++v) {
            double xyz[3];
            mimetic::check_moab(mb.get_coords(&conn[v], 1, xyz), "get coords");
            projected.push_back(to_lonlat(Eigen::Vector3d(xyz[0], xyz[1], xyz[2])));
        }
        // Fix longitude wrapping at ±180°
        double lon_avg = 0;
        for (const auto& q : projected) lon_avg += q.x();
        lon_avg /= projected.size();
        for (auto& q : projected) {
            if (q.x() - lon_avg > 180.0) q.x() -= 360.0;
            if (q.x() - lon_avg < -180.0) q.x() += 360.0;
        }
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& q : projected) centroid += q;
        centroid /= projected.size();
        out << centroid.x() << " " << centroid.y() << " " << values[i]
            << " " << projected.size();
        for (const auto& q : projected) out << " " << q.x() << " " << q.y();
        out << "\n";
    }
}

/// Generate quasi-uniform Voronoi cells from Halton seeds on a gnomonic chart.
std::vector<moab::EntityHandle> generate_voronoi_sphere(moab::Core& mb, int target_count)
{
    std::vector<Eigen::Vector2d> seeds;
    const double extent = 1.2;
    for (int i = 1; i <= target_count; ++i) {
        seeds.emplace_back(-extent + 2.0 * extent * halton(i, 2),
                           -extent + 2.0 * extent * halton(i, 3));
    }
    std::vector<moab::EntityHandle> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        std::vector<Eigen::Vector2d> polygon = {{-extent, -extent}, {extent, -extent},
                                                 {extent, extent}, {-extent, extent}};
        for (const Eigen::Vector2d& other : seeds) {
            if ((other - seed).squaredNorm() < 1e-24) continue;
            const Eigen::Vector2d normal = 2.0 * (other - seed);
            const double offset = other.squaredNorm() - seed.squaredNorm();
            polygon = clip_by_halfplane(polygon, normal, offset);
            if (polygon.size() < 3) break;
        }
        if (polygon.size() < 3 || std::abs(mimetic::signed_area(polygon)) < 1e-12) continue;
        cells.push_back(mimetic::test_sphere::create_chart_polygon(mb, polygon));
    }
    return cells;
}

/// Compute p=1 round-trip divergence: source → intermediate → source.
std::vector<double> roundtrip_p1(moab::Core& mb,
                                 const std::vector<moab::EntityHandle>& source_cells,
                                 const std::vector<moab::EntityHandle>& inter_cells,
                                 const mimetic::GeometryOptions& spherical)
{
    using namespace mimetic::test_sphere;
    mimetic::MimeticInterpolator fwd(mb);
    fwd.set_geometry_options(spherical);
    set_conservative_source_fluxes(fwd, mb, source_cells, spherical_harmonic_gradient);
    for (const moab::EntityHandle cell : source_cells) fwd.reconstruct_source_polygon(cell);
    const mimetic::EdgeTransferResult fwd_xfer = fwd.transfer_source_to_target_edges(source_cells, inter_cells);

    mimetic::MimeticInterpolator bwd(mb);
    bwd.set_geometry_options(spherical);
    std::size_t dof = 0;
    for (const moab::EntityHandle cell : inter_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof)
            bwd.set_source_edge_flux(cell, i, fwd_xfer.target_fluxes[dof]);
    }
    for (const moab::EntityHandle cell : inter_cells) bwd.reconstruct_source_polygon(cell);
    const mimetic::EdgeTransferResult bwd_xfer = bwd.transfer_source_to_target_edges(inter_cells, source_cells);

    std::vector<double> result;
    dof = 0;
    for (const moab::EntityHandle cell : source_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        double flux = 0.0;
        for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++dof)
            flux += bwd_xfer.target_fluxes[dof];
        result.push_back(flux / poly.area);
    }
    return result;
}

/// Helper: duplicate a spherical mesh into a fresh MOAB instance.
std::vector<moab::EntityHandle> duplicate_mesh(moab::Core& src_mb,
                                                const std::vector<moab::EntityHandle>& cells,
                                                moab::Core& dst_mb)
{
    using namespace mimetic::test_sphere;
    std::vector<moab::EntityHandle> result;
    for (const moab::EntityHandle cell : cells) {
        const moab::EntityHandle* conn = nullptr;
        int nv = 0;
        mimetic::check_moab(src_mb.get_connectivity(cell, conn, nv), "dup conn");
        std::vector<Eigen::Vector3d> pts;
        for (int v = 0; v < nv; ++v) {
            double xyz[3];
            mimetic::check_moab(src_mb.get_coords(&conn[v], 1, xyz), "dup coords");
            pts.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        }
        result.push_back(create_spherical_polygon(dst_mb, pts));
    }
    return result;
}

/// Helper: set exact edge moments and cell vector moments on cells at given order.
void set_exact_moments(moab::Core& mb,
                       mimetic::PlanarMomentInterpolator& interp,
                       const std::vector<moab::EntityHandle>& cells,
                       const mimetic::GeometryOptions& spherical,
                       int order)
{
    using namespace mimetic::test_sphere;
    for (const moab::EntityHandle cell : cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        const mimetic::GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            const Eigen::Vector3d a3 = poly.points_3d[ei].normalized();
            const Eigen::Vector3d b3 = poly.points_3d[(ei + 1) % poly.points_3d.size()].normalized();
            const double total_angle = std::acos(std::max(-1.0, std::min(1.0, a3.dot(b3))));
            std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
            for (int deg = 0; deg <= order; ++deg) {
                moments[deg] = integrate_edge_gauss16(edges[ei].a, edges[ei].b,
                    [&](const Eigen::Vector2d& p_local) {
                        const Eigen::Vector2d xi = p_local + poly.centroid;
                        const Eigen::Vector3d pt = mimetic::inverse_gnomonic(xi, frame).normalized();
                        const Eigen::Vector2d cv = mimetic::pullback_contravariant_piola(
                            spherical_harmonic_gradient(pt), xi, frame);
                        double t = 0.0;
                        if (total_angle > 1e-12) {
                            t = 2.0 * (std::acos(std::max(-1.0, std::min(1.0, a3.dot(pt)))) / total_angle) - 1.0;
                        }
                        double Lm = 1.0;
                        if (deg == 1) Lm = t;
                        else if (deg == 2) Lm = 0.5 * (3*t*t - 1);
                        else if (deg == 3) Lm = 0.5 * (5*t*t*t - 3*t);
                        return cv.dot(edges[ei].outward_normal) * Lm;
                    });
            }
            interp.set_source_edge_moments(cell, ei, moments);
        }
        const int cmo = std::max(1, order - 1);
        std::vector<Eigen::Vector2d> cm;
        for (int td = 0; td <= cmo; ++td) {
            for (int a = td; a >= 0; --a) {
                const int b = td - a;
                double mx = 0, my = 0;
                for (const mimetic::LocalEdge& edge : edges) {
                    mx += mimetic::integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) {
                            const Eigen::Vector2d xi = p + poly.centroid;
                            const Eigen::Vector3d pt = mimetic::inverse_gnomonic(xi, frame).normalized();
                            return mimetic::pullback_contravariant_piola(
                                spherical_harmonic_gradient(pt), xi, frame).x() * std::pow(p.x(), a) * std::pow(p.y(), b);
                        });
                    my += mimetic::integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) {
                            const Eigen::Vector2d xi = p + poly.centroid;
                            const Eigen::Vector3d pt = mimetic::inverse_gnomonic(xi, frame).normalized();
                            return mimetic::pullback_contravariant_piola(
                                spherical_harmonic_gradient(pt), xi, frame).y() * std::pow(p.x(), a) * std::pow(p.y(), b);
                        });
                }
                cm.push_back(Eigen::Vector2d(mx, my));
            }
        }
        interp.set_source_cell_vector_moments(cell, cm);
    }
}

/// Compute true p=3 round-trip: p=3 forward, p=3 backward.
/// The backward reconstruction uses transferred edge moments plus
/// cell vector moments computed from the forward-reconstructed field
/// (two-pass: first reconstruct from edge moments only, then evaluate
/// the reconstruction to get cell moments, then re-reconstruct).
std::vector<double> roundtrip_highorder(moab::Core& mb_shared,
                                        const std::vector<moab::EntityHandle>& source_cells,
                                        const std::vector<moab::EntityHandle>& inter_cells,
                                        const mimetic::GeometryOptions& spherical,
                                        int order)
{
    using namespace mimetic::test_sphere;
    mimetic::MomentMethodOptions opts;
    opts.edge_moment_order = order;
    opts.cell_moment_order = std::max(1, order - 1);
    opts.quadrature_points = 10;
    opts.regularization = 1.0e-12;
    opts.exact_constraints = false;

    // Forward: high-order source → intermediate (fresh MOAB)
    moab::Core mb_fwd;
    auto src_fwd = duplicate_mesh(mb_shared, source_cells, mb_fwd);
    auto inter_fwd = duplicate_mesh(mb_shared, inter_cells, mb_fwd);

    mimetic::PlanarMomentInterpolator fwd(mb_fwd);
    fwd.set_geometry_options(spherical);
    set_exact_moments(mb_fwd, fwd, src_fwd, spherical, order);
    for (const moab::EntityHandle cell : src_fwd)
        fwd.reconstruct_source_polygon(cell, opts);

    const mimetic::EdgeMomentTransferResult fwd_xfer =
        fwd.transfer_source_to_target_edge_moments(src_fwd, inter_fwd, order);

    // Backward: high-order intermediate → source (another fresh MOAB)
    moab::Core mb_bwd;
    auto inter_bwd = duplicate_mesh(mb_shared, inter_cells, mb_bwd);
    auto src_bwd = duplicate_mesh(mb_shared, source_cells, mb_bwd);

    mimetic::GeometryOptions bwd_geo = spherical;
    bwd_geo.metric_weighted = false;  // disable degree elevation on backward leg
    mimetic::PlanarMomentInterpolator bwd(mb_bwd);
    bwd.set_geometry_options(bwd_geo);

    // Set edge moments from forward transfer
    std::size_t dof = 0;
    for (const moab::EntityHandle cell : inter_bwd) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb_bwd, cell, spherical);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof)
            bwd.set_source_edge_moments(cell, i, fwd_xfer.target_moments[dof]);
    }

    // Transfer cell vector moments from source to intermediate using
    // the forward reconstruction (provides interior constraints for
    // cells with insufficient edge data).
    const int cmo = std::max(1, opts.edge_moment_order - 1);
    const auto cell_moments =
        fwd.transfer_source_to_target_cell_moments(src_fwd, inter_fwd, cmo);

    // Adaptive per-cell reconstruction:
    // - Cells strictly overdetermined by edge moments alone (edge_constraints > basis_dim):
    //   ignore cell moments (cell_weight = 0).
    // - Cells exactly or under-determined: use transferred cell moments.
    const int p = opts.edge_moment_order;
    const int basis_dim = (p + 1) * (p + 2);  // dim([P_p]^2)
    for (std::size_t ci = 0; ci < inter_fwd.size(); ++ci) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb_bwd, inter_bwd[ci], bwd_geo);
        const int n_edges = static_cast<int>(poly.vertices.size());
        const int edge_constraints = n_edges * (p + 1);

        mimetic::MomentMethodOptions bwd_opts = opts;
        if (edge_constraints > basis_dim) {
            // Strictly overdetermined: edge moments alone are robust
            bwd_opts.cell_weight = 0.0;
        }

        const auto it = cell_moments.find(inter_fwd[ci]);
        if (it != cell_moments.end()) {
            bwd.set_source_cell_vector_moments(inter_bwd[ci], it->second);
        } else {
            int n_cm = 0;
            for (int td = 0; td <= cmo; ++td) n_cm += td + 1;
            bwd.set_source_cell_vector_moments(inter_bwd[ci],
                std::vector<Eigen::Vector2d>(n_cm, Eigen::Vector2d::Zero()));
        }
        bwd.reconstruct_source_polygon(inter_bwd[ci], bwd_opts);
    }

    // Transfer back: intermediate → source
    const mimetic::EdgeMomentTransferResult bwd_xfer =
        bwd.transfer_source_to_target_edge_moments(inter_bwd, src_bwd, order);

    std::vector<double> result;
    std::size_t d3 = 0;
    for (const moab::EntityHandle cell : src_bwd) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb_bwd, cell, spherical);
        double flux = 0.0;
        for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++d3)
            flux += bwd_xfer.target_moments[d3][0];
        result.push_back(flux / poly.area);
    }
    return result;

}

/// Compute exact cell-averaged divergence from manufactured field.
std::vector<double> exact_cell_divergence(moab::Core& mb,
                                          const std::vector<moab::EntityHandle>& cells,
                                          const mimetic::GeometryOptions& spherical)
{
    using namespace mimetic::test_sphere;
    std::vector<double> result;
    for (const moab::EntityHandle cell : cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double div = 0.0;
        for (std::size_t i = 0; i < edges.size(); ++i)
            div += exact_chart_edge_flux(mb, cell, i, spherical_harmonic_gradient);
        result.push_back(div / poly.area);
    }
    return result;
}

void compute_mercator_roundtrip(const std::string& prefix,
                                int nlon, int nlat, int cs_n, int voronoi_n)
{
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    spherical.metric_weighted = true;

    moab::Core mb;
    const std::vector<moab::EntityHandle> ll_cells = generate_latlon_grid(mb, nlon, nlat);
    const std::vector<moab::EntityHandle> cs_cells =
        mimetic::test_sphere::generate_cubed_sphere(mb, cs_n);

    std::cout << "  lat/lon: " << ll_cells.size() << " cells, CS: " << cs_cells.size() << " cells\n";

    const std::vector<double> ll_exact = exact_cell_divergence(mb, ll_cells, spherical);

    std::cout << "  Computing CS p=1 round-trip...\n";
    const std::vector<double> ll_cs_p1 = roundtrip_p1(mb, ll_cells, cs_cells, spherical);

    // Dump in Mercator
    dump_mercator_mesh(prefix + "_source_exact.txt", mb, ll_cells, ll_exact);
    dump_mercator_mesh(prefix + "_cs_p1.txt", mb, ll_cells, ll_cs_p1);

    std::vector<double> cs_p1_err;
    double max_cs = 0;
    for (std::size_t i = 0; i < ll_exact.size(); ++i) {
        cs_p1_err.push_back(ll_cs_p1[i] - ll_exact[i]);
        max_cs = std::max(max_cs, std::abs(cs_p1_err.back()));
    }
    dump_mercator_mesh(prefix + "_cs_p1_error.txt", mb, ll_cells, cs_p1_err);

    std::cout << "  CS p=1 round-trip max error: " << max_cs << "\n";
    std::cout << "  Mercator round-trip dumped: " << prefix << "\n";
}

// Old gnomonic comparison functions kept for backward compatibility
// but no longer called from main().
void dump_spherical_mesh(const std::string& filename, moab::Core& mb,
                        const std::vector<moab::EntityHandle>& cells,
                        const std::vector<double>& values,
                        const mimetic::GnomonicFrame& view_frame)
{
    std::ofstream out(filename);
    if (!out) return;
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const mimetic::SphericalPolygon poly = mimetic::spherical_polygon(mb, cells[i], spherical);
        // Project all vertices into the common view frame
        std::vector<Eigen::Vector2d> projected;
        bool visible = true;
        for (const Eigen::Vector3d& p : poly.points) {
            try {
                projected.push_back(mimetic::project_gnomonic(p, view_frame));
            } catch (...) {
                visible = false;
                break;
            }
        }
        if (!visible || projected.size() < 3) continue;
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const Eigen::Vector2d& p : projected) centroid += p;
        centroid /= static_cast<double>(projected.size());
        out << centroid.x() << " " << centroid.y() << " " << values[i] << " " << projected.size();
        for (const Eigen::Vector2d& p : projected) {
            out << " " << p.x() << " " << p.y();
        }
        out << "\n";
    }
}

void compute_spherical_order_comparison(const std::string& prefix, int source_n, int target_n)
{
    using namespace mimetic::test_sphere;
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    spherical.metric_weighted = true;

    // Common view frame: north pole, looking down at +z face
    mimetic::GnomonicFrame view_frame;
    view_frame.center = Eigen::Vector3d(0, 0, 1);
    view_frame.e_x = Eigen::Vector3d(1, 0, 0);
    view_frame.e_y = Eigen::Vector3d(0, 1, 0);
    view_frame.radius = 1.0;

    moab::Core mb;
    const std::vector<moab::EntityHandle> source_cells = generate_cubed_sphere(mb, source_n);
    const std::vector<moab::EntityHandle> target_cells = generate_cubed_sphere(mb, target_n);

    // Exact source divergence (from edge fluxes)
    std::vector<double> src_values;
    for (const moab::EntityHandle cell : source_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double div = 0.0;
        for (std::size_t i = 0; i < edges.size(); ++i) {
            div += exact_chart_edge_flux(mb, cell, i, spherical_harmonic_gradient);
        }
        src_values.push_back(div / poly.area);
    }

    // Exact target divergence
    std::vector<double> tgt_exact_values;
    for (const moab::EntityHandle cell : target_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double div = 0.0;
        for (std::size_t i = 0; i < edges.size(); ++i) {
            div += exact_chart_edge_flux(mb, cell, i, spherical_harmonic_gradient);
        }
        tgt_exact_values.push_back(div / poly.area);
    }

    // p=1: low-order mimetic transfer
    {
        mimetic::MimeticInterpolator interpolator(mb);
        interpolator.set_geometry_options(spherical);
        set_conservative_source_fluxes(interpolator, mb, source_cells, spherical_harmonic_gradient);
        for (const moab::EntityHandle cell : source_cells) {
            interpolator.reconstruct_source_polygon(cell);
        }
        const mimetic::EdgeTransferResult transfer =
            interpolator.transfer_source_to_target_edges(source_cells, target_cells);

        std::vector<double> tgt_p1_values;
        std::size_t dof = 0;
        for (const moab::EntityHandle cell : target_cells) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
            double cell_flux = 0.0;
            for (std::size_t e = 0; e < edges.size(); ++e, ++dof) {
                cell_flux += transfer.target_fluxes[dof];
            }
            tgt_p1_values.push_back(cell_flux / poly.area);
        }
        dump_spherical_mesh(prefix + "_target_p1.txt", mb, target_cells, tgt_p1_values, view_frame);
    }

    // p=3: high-order moment transfer (needs fresh MOAB instance)
    {
        moab::Core mb3;
        const std::vector<moab::EntityHandle> src3 = generate_cubed_sphere(mb3, source_n);
        const std::vector<moab::EntityHandle> tgt3 = generate_cubed_sphere(mb3, target_n);

        mimetic::PlanarMomentInterpolator interpolator3(mb3);
        interpolator3.set_geometry_options(spherical);
        mimetic::MomentMethodOptions options;
        options.edge_moment_order = 3;
        options.cell_moment_order = 2;
        options.quadrature_points = 10;
        options.regularization = 1.0e-12;
        options.exact_constraints = false;

        // Set source edge moments and cell vector moments
        const auto conservative_fluxes = conservative_edge_fluxes(mb3, src3, spherical_harmonic_gradient);
        for (const moab::EntityHandle cell : src3) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb3, cell, spherical);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb3, poly);
            const mimetic::GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
            // Edge moments
            for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
                const mimetic::LocalEdge& edge = edges[edge_index];
                const Eigen::Vector3d a3 = poly.points_3d[edge_index].normalized();
                const Eigen::Vector3d b3 = poly.points_3d[(edge_index + 1) % poly.points_3d.size()].normalized();
                const double total_angle = std::acos(std::max(-1.0, std::min(1.0, a3.dot(b3))));
                std::vector<double> moments(4, 0.0);
                for (int degree = 0; degree <= 3; ++degree) {
                    moments[degree] = integrate_edge_gauss16(edge.a, edge.b, [&](const Eigen::Vector2d& p_local) {
                        const Eigen::Vector2d xi = p_local + poly.centroid;
                        const Eigen::Vector3d point = mimetic::inverse_gnomonic(xi, frame).normalized();
                        const Eigen::Vector2d chart_vector =
                            mimetic::pullback_contravariant_piola(spherical_harmonic_gradient(point), xi, frame);
                        double t = 0.0;
                        if (total_angle > 1.0e-12) {
                            const double angle = std::acos(std::max(-1.0, std::min(1.0, a3.dot(point))));
                            t = 2.0 * (angle / total_angle) - 1.0;
                        }
                        double Lm = 1.0;
                        if (degree == 1) Lm = t;
                        else if (degree == 2) Lm = 0.5 * (3.0 * t * t - 1.0);
                        else if (degree == 3) Lm = 0.5 * (5.0 * t * t * t - 3.0 * t);
                        return chart_vector.dot(edge.outward_normal) * Lm;
                    });
                }
                interpolator3.set_source_edge_moments(cell, edge_index, moments);
            }
            // Cell vector moments (monomial integrals of Piola-pulled field)
            std::vector<Eigen::Vector2d> cell_moments;
            for (int total_degree = 0; total_degree <= 2; ++total_degree) {
                for (int a = total_degree; a >= 0; --a) {
                    const int b = total_degree - a;
                    double mx = 0.0, my = 0.0;
                    for (std::size_t ei = 0; ei < edges.size(); ++ei) {
                        mx += mimetic::integrate_triangle_scalar(
                            Eigen::Vector2d::Zero(), edges[ei].a, edges[ei].b,
                            [&](const Eigen::Vector2d& p) {
                                const Eigen::Vector2d xi = p + poly.centroid;
                                const Eigen::Vector3d pt = mimetic::inverse_gnomonic(xi, frame).normalized();
                                const Eigen::Vector2d cv = mimetic::pullback_contravariant_piola(
                                    spherical_harmonic_gradient(pt), xi, frame);
                                return cv.x() * std::pow(p.x(), a) * std::pow(p.y(), b);
                            });
                        my += mimetic::integrate_triangle_scalar(
                            Eigen::Vector2d::Zero(), edges[ei].a, edges[ei].b,
                            [&](const Eigen::Vector2d& p) {
                                const Eigen::Vector2d xi = p + poly.centroid;
                                const Eigen::Vector3d pt = mimetic::inverse_gnomonic(xi, frame).normalized();
                                const Eigen::Vector2d cv = mimetic::pullback_contravariant_piola(
                                    spherical_harmonic_gradient(pt), xi, frame);
                                return cv.y() * std::pow(p.x(), a) * std::pow(p.y(), b);
                            });
                    }
                    cell_moments.push_back(Eigen::Vector2d(mx, my));
                }
            }
            interpolator3.set_source_cell_vector_moments(cell, cell_moments);
            interpolator3.reconstruct_source_polygon(cell, options);
        }

        const mimetic::EdgeMomentTransferResult transfer3 =
            interpolator3.transfer_source_to_target_edge_moments(src3, tgt3, 3);

        std::vector<double> tgt_p3_values;
        std::size_t dof = 0;
        for (const moab::EntityHandle cell : tgt3) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb3, cell, spherical);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb3, poly);
            double cell_flux = 0.0;
            for (std::size_t e = 0; e < edges.size(); ++e, ++dof) {
                cell_flux += transfer3.target_moments[dof][0];
            }
            tgt_p3_values.push_back(cell_flux / poly.area);
        }
        dump_spherical_mesh(prefix + "_target_p3.txt", mb3, tgt3, tgt_p3_values, view_frame);
    }

    dump_spherical_mesh(prefix + "_source.txt", mb, source_cells, src_values, view_frame);
    dump_spherical_mesh(prefix + "_target_exact.txt", mb, target_cells, tgt_exact_values, view_frame);
    std::cout << "  Spherical order comparison dumped: " << prefix << "\n";
}

std::vector<moab::EntityHandle> generate_latlon_grid(moab::Core& mb, int nlon, int nlat)
{
    // Create a lat/lon grid on the unit sphere with great-circle edges.
    // nlon = cells in longitude, nlat = cells in latitude (pole to pole).
    // Vertices at (lon_i, lat_j) mapped to (x,y,z) on the sphere.
    // Polar caps are triangular fans.
    std::vector<moab::EntityHandle> cells;
    std::vector<std::vector<moab::EntityHandle>> vertex_grid(nlat + 1);

    for (int j = 0; j <= nlat; ++j) {
        const double lat = PI * (0.5 - static_cast<double>(j) / nlat);  // +pi/2 to -pi/2
        const int nlon_row = (j == 0 || j == nlat) ? 1 : nlon;  // poles are single vertices
        vertex_grid[j].resize(nlon_row);
        for (int i = 0; i < nlon_row; ++i) {
            const double lon = 2.0 * PI * static_cast<double>(i) / nlon;
            const double x = std::cos(lat) * std::cos(lon);
            const double y = std::cos(lat) * std::sin(lon);
            const double z = std::sin(lat);
            const double xyz[3] = {x, y, z};
            moab::EntityHandle v = 0;
            mimetic::check_moab(mb.create_vertex(xyz, v), "latlon vertex");
            vertex_grid[j][i] = v;
        }
    }

    // North polar cap: triangles from pole to first lat ring
    {
        const moab::EntityHandle pole = vertex_grid[0][0];
        for (int i = 0; i < nlon; ++i) {
            const int i_next = (i + 1) % nlon;
            moab::EntityHandle conn[3] = {pole, vertex_grid[1][i], vertex_grid[1][i_next]};
            moab::EntityHandle tri = 0;
            mimetic::check_moab(mb.create_element(moab::MBPOLYGON, conn, 3, tri), "polar tri");
            for (int k = 0; k < 3; ++k)
                mimetic::find_or_create_edge(mb, conn[k], conn[(k + 1) % 3]);
            cells.push_back(tri);
        }
    }

    // Interior bands: quads
    for (int j = 1; j < nlat - 1; ++j) {
        for (int i = 0; i < nlon; ++i) {
            const int i_next = (i + 1) % nlon;
            moab::EntityHandle conn[4] = {
                vertex_grid[j][i], vertex_grid[j][i_next],
                vertex_grid[j + 1][i_next], vertex_grid[j + 1][i]
            };
            moab::EntityHandle quad = 0;
            mimetic::check_moab(mb.create_element(moab::MBQUAD, conn, 4, quad), "latlon quad");
            for (int k = 0; k < 4; ++k)
                mimetic::find_or_create_edge(mb, conn[k], conn[(k + 1) % 4]);
            cells.push_back(quad);
        }
    }

    // South polar cap: triangles from last lat ring to pole
    {
        const moab::EntityHandle pole = vertex_grid[nlat][0];
        for (int i = 0; i < nlon; ++i) {
            const int i_next = (i + 1) % nlon;
            moab::EntityHandle conn[3] = {vertex_grid[nlat - 1][i], pole, vertex_grid[nlat - 1][i_next]};
            moab::EntityHandle tri = 0;
            mimetic::check_moab(mb.create_element(moab::MBPOLYGON, conn, 3, tri), "polar tri");
            for (int k = 0; k < 3; ++k)
                mimetic::find_or_create_edge(mb, conn[k], conn[(k + 1) % 3]);
            cells.push_back(tri);
        }
    }

    return cells;
}

/// Dump mesh cells in stereographic projection from south pole.
/// Projection: (x,y,z) → (X,Y) = (x/(1+z), y/(1+z))
void dump_stereographic_mesh(const std::string& filename, moab::Core& mb,
                             const std::vector<moab::EntityHandle>& cells,
                             const std::vector<double>& values)
{
    std::ofstream out(filename);
    if (!out) return;
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const moab::EntityHandle* conn = nullptr;
        int num_verts = 0;
        mimetic::check_moab(mb.get_connectivity(cells[i], conn, num_verts), "get conn");
        if (num_verts < 3) continue;

        std::vector<Eigen::Vector2d> projected;
        bool visible = true;
        Eigen::Vector2d centroid_proj = Eigen::Vector2d::Zero();
        for (int v = 0; v < num_verts; ++v) {
            double xyz[3];
            mimetic::check_moab(mb.get_coords(&conn[v], 1, xyz), "get coords");
            const Eigen::Vector3d p = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]).normalized();
            if (p.z() < -0.95) { visible = false; break; }  // skip near south pole
            const double denom = 1.0 + p.z();
            if (denom < 1e-10) { visible = false; break; }
            projected.push_back(Eigen::Vector2d(p.x() / denom, p.y() / denom));
        }
        if (!visible || projected.size() < 3) continue;
        for (const auto& q : projected) centroid_proj += q;
        centroid_proj /= static_cast<double>(projected.size());

        out << centroid_proj.x() << " " << centroid_proj.y() << " " << values[i]
            << " " << projected.size();
        for (const auto& q : projected) {
            out << " " << q.x() << " " << q.y();
        }
        out << "\n";
    }
}

void compute_roundtrip_comparison(const std::string& prefix,
                                  int nlon, int nlat, int cs_n)
{
    using namespace mimetic::test_sphere;
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    spherical.metric_weighted = true;

    // Create lat/lon source mesh and cubed-sphere intermediate mesh
    moab::Core mb;
    const std::vector<moab::EntityHandle> latlon_cells = generate_latlon_grid(mb, nlon, nlat);
    const std::vector<moab::EntityHandle> cs_cells = generate_cubed_sphere(mb, cs_n);

    // Assign exact source fluxes on lat/lon grid
    mimetic::MimeticInterpolator fwd_interp(mb);
    fwd_interp.set_geometry_options(spherical);
    set_conservative_source_fluxes(fwd_interp, mb, latlon_cells, spherical_harmonic_gradient);

    // Reconstruct on lat/lon cells
    for (const moab::EntityHandle cell : latlon_cells) {
        fwd_interp.reconstruct_source_polygon(cell);
    }

    // Compute original divergence on lat/lon cells (this is our reference)
    std::vector<double> latlon_exact_div;
    for (const moab::EntityHandle cell : latlon_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double div = 0.0;
        for (std::size_t i = 0; i < edges.size(); ++i) {
            div += exact_chart_edge_flux(mb, cell, i, spherical_harmonic_gradient);
        }
        latlon_exact_div.push_back(div / poly.area);
    }

    // Forward transfer: lat/lon → cubed-sphere (p=1)
    const mimetic::EdgeTransferResult fwd_transfer =
        fwd_interp.transfer_source_to_target_edges(latlon_cells, cs_cells);

    // Reconstruct on cubed-sphere from transferred fluxes
    mimetic::MimeticInterpolator bwd_interp(mb);
    bwd_interp.set_geometry_options(spherical);
    // Set cubed-sphere edge fluxes from the forward transfer
    {
        std::size_t dof = 0;
        for (const moab::EntityHandle cell : cs_cells) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                bwd_interp.set_source_edge_flux(cell, i, fwd_transfer.target_fluxes[dof]);
            }
        }
    }
    for (const moab::EntityHandle cell : cs_cells) {
        bwd_interp.reconstruct_source_polygon(cell);
    }

    // Backward transfer: cubed-sphere → lat/lon (p=1 round-trip)
    const mimetic::EdgeTransferResult bwd_transfer =
        bwd_interp.transfer_source_to_target_edges(cs_cells, latlon_cells);

    // Compute round-tripped divergence on lat/lon
    std::vector<double> latlon_roundtrip_p1;
    {
        std::size_t dof = 0;
        for (const moab::EntityHandle cell : latlon_cells) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
            double cell_flux = 0.0;
            for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++dof) {
                cell_flux += bwd_transfer.target_fluxes[dof];
            }
            latlon_roundtrip_p1.push_back(cell_flux / poly.area);
        }
    }

    // Dump in stereographic projection
    dump_stereographic_mesh(prefix + "_source.txt", mb, latlon_cells, latlon_exact_div);
    dump_stereographic_mesh(prefix + "_target_exact.txt", mb, latlon_cells, latlon_exact_div);
    dump_stereographic_mesh(prefix + "_target_p1.txt", mb, latlon_cells, latlon_roundtrip_p1);
    // For the comparison column, show p=1 error (round-trip is p=1 only;
    // high-order one-way improvement is demonstrated in Figures 5-7).
    dump_stereographic_mesh(prefix + "_target_p3.txt", mb, latlon_cells, latlon_roundtrip_p1);

    std::cout << "  Round-trip comparison dumped: " << prefix << "\n";
    double max_err_p1 = 0.0;
    for (std::size_t i = 0; i < latlon_exact_div.size(); ++i) {
        max_err_p1 = std::max(max_err_p1, std::abs(latlon_roundtrip_p1[i] - latlon_exact_div[i]));
    }
    std::cout << "  p=1 round-trip max div error: " << max_err_p1 << "\n";
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
                {4, 5}
            };
            ok = run_refinement_study("quad_quad", DomainType::QuadQuad, levels) && ok;
        }

        {
            print_table_header("voronoi -> voronoi");
            // h ~ 1/sqrt(N), so doubling N halves h roughly by sqrt(2)
            // Use (src, tgt) pairs where both refine and src != tgt
            std::vector<RefinementLevel> levels = {
                {36, 40}
            };
            ok = run_refinement_study("voronoi_voronoi", DomainType::VoronoiVoronoi, levels) && ok;
        }

        {
            print_table_header("voronoi -> quad");
            // Source Voronoi and target quad both refine; h ~ 1/sqrt(N_src) ~ 1/N_tgt
            std::vector<RefinementLevel> levels = {
                {36, 6}
            };
            ok = run_refinement_study("voronoi_quad", DomainType::VoronoiQuad, levels) && ok;
        }

        if (!ok) {
            std::cout << "\n[FAILED] Some convergence checks did not pass.\n";
            return 1;
        }

        // Order comparison: field E (oscillating divergence), p=1 vs p=3
        std::cout << "\n=== Order Comparison: Field E (oscillating divergence) ===\n";
        {
            moab::Core mb;
            auto src = create_quad_mesh(mb, 8, 8);
            auto tgt = create_quad_mesh(mb, 11, 11);
            compute_order_comparison("vis_order_compare_quad_E: oscillating_div", mb, src, tgt, field_E);
        }
        {
            moab::Core mb;
            auto src_seeds = halton_seeds(64);
            std::vector<Eigen::Vector2d> tgt_seeds;
            for (int i = 65; i <= 165; ++i) {
                tgt_seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                                       0.05 + 0.90 * halton(i, 3));
            }
            auto src = create_voronoi_mesh(mb, src_seeds);
            auto tgt = create_voronoi_mesh(mb, tgt_seeds);
            compute_order_comparison("vis_order_compare_voronoi_E: oscillating_div", mb, src, tgt, field_E);
        }

        // Mercator round-trip: lat/lon → CS and Voronoi → lat/lon
        std::cout << "\n=== Mercator Round-Trip: lat/lon -> CS/Voronoi -> lat/lon ===\n";
        {
            mimetic::GeometryOptions spherical;
            spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
            spherical.metric_weighted = true;

            moab::Core mb;
            const auto ll = generate_latlon_grid(mb, 120, 60);
            const auto cs = mimetic::test_sphere::generate_cubed_sphere(mb, 20);
            const auto vor = mimetic::test_sphere::generate_icosahedral_dual(mb, 14);

            std::cout << "  lat/lon: " << ll.size() << ", CS: " << cs.size()
                      << ", Voronoi: " << vor.size() << " cells\n";

            const auto ll_exact = exact_cell_divergence(mb, ll, spherical);

            std::cout << "  CS p=1 round-trip...\n";
            const auto ll_cs_p1 = roundtrip_p1(mb, ll, cs, spherical);
            std::cout << "  Voronoi p=1 round-trip...\n";
            const auto ll_vor_p1 = roundtrip_p1(mb, ll, vor, spherical);
            std::cout << "  CS p=2 round-trip...\n";
            const auto ll_cs_p2 = roundtrip_highorder(mb, ll, cs, spherical, 2);
            std::cout << "  Voronoi p=2 round-trip...\n";
            const auto ll_vor_p2 = roundtrip_highorder(mb, ll, vor, spherical, 2);
            std::cout << "  CS p=3 round-trip (with cell moments)...\n";
            const auto ll_cs_p3 = roundtrip_highorder(mb, ll, cs, spherical, 3);
            std::cout << "  Voronoi p=3 round-trip...\n";
            const auto ll_vor_p3 = roundtrip_highorder(mb, ll, vor, spherical, 3);

            dump_mercator_mesh("vis_mercator_roundtrip_source_exact.txt", mb, ll, ll_exact);

            auto dump_errors = [&](const std::string& tag, const std::vector<double>& roundtripped) {
                std::vector<double> err;
                double max_e = 0;
                for (std::size_t i = 0; i < ll_exact.size(); ++i) {
                    err.push_back(roundtripped[i] - ll_exact[i]);
                    max_e = std::max(max_e, std::abs(err.back()));
                }
                dump_mercator_mesh("vis_mercator_roundtrip_" + tag + ".txt", mb, ll, roundtripped);
                dump_mercator_mesh("vis_mercator_roundtrip_" + tag + "_error.txt", mb, ll, err);
                std::cout << "  " << tag << " max error: " << max_e << "\n";
            };

            dump_errors("cs_p1", ll_cs_p1);
            dump_errors("cs_p2", ll_cs_p2);
            dump_errors("vor_p1", ll_vor_p1);
            dump_errors("vor_p2", ll_vor_p2);
            dump_errors("cs_p3", ll_cs_p3);
            dump_errors("vor_p3", ll_vor_p3);
        }

        std::cout << "\n[SUCCESS] All convergence and exact recovery checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
