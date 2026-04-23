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

        // Spherical order comparison: cubed-sphere, p=1 vs p=3
        std::cout << "\n=== Spherical Order Comparison: Y_2^0 gradient ===\n";
        compute_spherical_order_comparison("vis_order_compare_spherical", 8, 12);

        std::cout << "\n[SUCCESS] All convergence and exact recovery checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
