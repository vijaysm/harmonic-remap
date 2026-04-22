#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

struct GaussPoint {
    double x;
    double w;
};

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
    for (const GaussPoint& q : gauss10_table) {
        sum += q.w * func(mid + q.x * half_delta);
    }
    return 0.5 * length * sum;
}

template <typename Func>
double integrate_triangle_duffy_highorder(const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b,
                                          const Eigen::Vector2d& c,
                                          const Func& func)
{
    const Eigen::Vector2d ab = b - a;
    const Eigen::Vector2d bc = c - b;
    const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
    double sum = 0.0;

    for (const GaussPoint& qu : gauss10_table) {
        const double u = 0.5 * (qu.x + 1.0);
        const double wu = 0.5 * qu.w;
        for (const GaussPoint& qv : gauss10_table) {
            const double v = 0.5 * (qv.x + 1.0);
            const double wv = 0.5 * qv.w;
            const Eigen::Vector2d p = a + u * ab + (u * v) * bc;
            sum += wu * wv * u * func(p);
        }
    }

    return det_j * sum;
}

double legendre(const int degree, const double x)
{
    if (degree == 0) {
        return 1.0;
    }
    if (degree == 1) {
        return x;
    }
    double p_nm2 = 1.0;
    double p_nm1 = x;
    double p_n = x;
    for (int n = 2; n <= degree; ++n) {
        p_n = ((2.0 * n - 1.0) * x * p_nm1 - (n - 1.0) * p_nm2) / static_cast<double>(n);
        p_nm2 = p_nm1;
        p_nm1 = p_n;
    }
    return p_n;
}

using FieldFunc = Eigen::Vector2d(*)(const Eigen::Vector2d&);

Eigen::Vector2d smooth_variable_divergence_field(const Eigen::Vector2d& p)
{
    const double x = p.x();
    const double y = p.y();
    return Eigen::Vector2d(
        std::exp(0.20 * x) * (std::sin(PI * x) + 0.20 * y * y) + 0.05 * x * x * x,
        std::exp(0.20 * y) * (std::cos(PI * y) + 0.20 * x * y) + 0.05 * y * y * y);
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

std::vector<Eigen::Vector2d> halton_seeds(const int count, const int offset)
{
    std::vector<Eigen::Vector2d> seeds;
    seeds.reserve(static_cast<std::size_t>(count));
    for (int i = 1; i <= count; ++i) {
        const int k = i + offset;
        seeds.emplace_back(0.05 + 0.90 * halton(k, 2),
                           0.05 + 0.90 * halton(k, 3));
    }
    return seeds;
}

std::vector<Eigen::Vector2d> clip_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                               const Eigen::Vector2d& normal,
                                               const double offset)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) {
        return output;
    }

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
            if (!previous_inside) {
                output.push_back(intersect(previous, current));
            }
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
        Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(1.0, 0.0),
        Eigen::Vector2d(1.0, 1.0),
        Eigen::Vector2d(0.0, 1.0),
    };
    for (const Eigen::Vector2d& other : seeds) {
        if ((other - seed).squaredNorm() < 1.0e-24) {
            continue;
        }
        const Eigen::Vector2d normal = 2.0 * (other - seed);
        const double offset = other.squaredNorm() - seed.squaredNorm();
        cell = clip_by_halfplane(cell, normal, offset);
        if (cell.size() < 3) {
            return {};
        }
    }
    return cell;
}

double polygon_area(const std::vector<Eigen::Vector2d>& polygon)
{
    return std::abs(mimetic::signed_area(polygon));
}

struct CellInfo {
    moab::EntityHandle handle;
    std::vector<Eigen::Vector2d> points;
};

std::vector<CellInfo> create_quad_mesh(moab::Core& mb, const int nx, const int ny)
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

std::vector<CellInfo> create_voronoi_mesh(moab::Core& mb,
                                          const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<CellInfo> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        const std::vector<Eigen::Vector2d> polygon = voronoi_cell_polygon(seed, seeds);
        if (polygon.size() < 3 || polygon_area(polygon) < 1.0e-12) {
            continue;
        }
        cells.push_back(CellInfo{mimetic::create_polygon(mb, polygon), polygon});
    }
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        polygons.push_back(cell.handle);
    }
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<moab::EntityHandle> handles(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> result;
    result.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        result.push_back(cell.handle);
    }
    return result;
}

std::vector<double> exact_edge_moments_absolute(const Eigen::Vector2d& a,
                                                const Eigen::Vector2d& b,
                                                const int order,
                                                const FieldFunc& field)
{
    std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    const double denom = delta.squaredNorm();
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    for (int degree = 0; degree <= order; ++degree) {
        moments[degree] = integrate_edge_highorder(a, b, [&](const Eigen::Vector2d& p) {
            const double t = (denom > 0.0) ? (2.0 * (p - a).dot(delta) / denom - 1.0) : 0.0;
            return field(p).dot(normal / length) * legendre(degree, t);
        });
    }
    return moments;
}

std::vector<Eigen::Vector2d> exact_cell_vector_moments(const mimetic::LocalPolygon& poly,
                                                       const int order,
                                                       const FieldFunc& field)
{
    std::vector<Eigen::Vector2d> moments;
    const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    for (int total_degree = 0; total_degree <= order; ++total_degree) {
        for (int a_pow = total_degree; a_pow >= 0; --a_pow) {
            const int b_pow = total_degree - a_pow;
            Eigen::Vector2d integral = Eigen::Vector2d::Zero();
            for (std::size_t i = 0; i < poly.points.size(); ++i) {
                const Eigen::Vector2d a = origin;
                const Eigen::Vector2d b = poly.points[i];
                const Eigen::Vector2d c = poly.points[(i + 1) % poly.points.size()];
                integral.x() += integrate_triangle_duffy_highorder(a, b, c, [&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).x() * std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
                integral.y() += integrate_triangle_duffy_highorder(a, b, c, [&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).y() * std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
            }
            moments.push_back(integral);
        }
    }
    return moments;
}

void set_source_moments(moab::Core& mb,
                        mimetic::PlanarMomentInterpolator& interpolator,
                        const moab::EntityHandle polygon,
                        const int order,
                        const int cell_order,
                        const FieldFunc& field)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const Eigen::Vector2d a = poly.centroid + edges[edge_index].a;
        const Eigen::Vector2d b = poly.centroid + edges[edge_index].b;
        interpolator.set_source_edge_moments(polygon, edge_index,
                                             exact_edge_moments_absolute(a, b, order, field));
    }
    interpolator.set_source_cell_vector_moments(polygon, exact_cell_vector_moments(poly, cell_order, field));
}

double exact_domain_divergence_integral(const FieldFunc& field)
{
    double flux = 0.0;
    flux += integrate_edge_highorder(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
                                     [&](const Eigen::Vector2d& p) { return -field(p).y(); });
    flux += integrate_edge_highorder(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 1.0),
                                     [&](const Eigen::Vector2d& p) { return field(p).x(); });
    flux += integrate_edge_highorder(Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(0.0, 1.0),
                                     [&](const Eigen::Vector2d& p) { return field(p).y(); });
    flux += integrate_edge_highorder(Eigen::Vector2d(0.0, 1.0), Eigen::Vector2d(0.0, 0.0),
                                     [&](const Eigen::Vector2d& p) { return -field(p).x(); });
    return flux;
}

struct CaseMetrics {
    int order = 0;
    std::string domain;
    double h = 0.0;
    double l2_flux_rel = 0.0;
    double l2_all_rel = 0.0;
    double linf_all = 0.0;
    double conservation = 0.0;
};

double convergence_rate(const double e_coarse, const double e_fine, const double h_coarse, const double h_fine)
{
    if (e_coarse < 1.0e-30 || e_fine < 1.0e-30) {
        return 0.0;
    }
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

CaseMetrics run_case(const std::string& domain,
                     moab::Core& mb,
                     const std::vector<CellInfo>& source_cells,
                     const std::vector<CellInfo>& target_cells,
                     const int order,
                     const double h)
{
    if (order == 1) {
        mimetic::MimeticInterpolator interpolator(mb);
        for (const CellInfo& source : source_cells) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, source.handle);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
            for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
                const Eigen::Vector2d a = poly.centroid + edges[edge_index].a;
                const Eigen::Vector2d b = poly.centroid + edges[edge_index].b;
                const double flux = exact_edge_moments_absolute(a, b, 0, smooth_variable_divergence_field)[0];
                interpolator.set_source_edge_flux(source.handle, edge_index, flux);
            }
            interpolator.reconstruct_source_polygon(source.handle);
        }

        const mimetic::EdgeTransferResult transfer =
            interpolator.transfer_source_to_target_edges(handles(source_cells), handles(target_cells));

        double flux_num = 0.0;
        double flux_den = 0.0;
        double linf_flux = 0.0;
        double total_target_flux = 0.0;

        std::size_t dof = 0;
        for (const CellInfo& target : target_cells) {
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, target.handle);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
            for (const mimetic::LocalEdge& edge : edges) {
                const Eigen::Vector2d a = poly.centroid + edge.a;
                const Eigen::Vector2d b = poly.centroid + edge.b;
                const double exact = exact_edge_moments_absolute(a, b, 0, smooth_variable_divergence_field)[0];
                const double computed = transfer.target_fluxes[dof];
                const double error = computed - exact;
                flux_num += error * error;
                flux_den += exact * exact;
                linf_flux = std::max(linf_flux, std::abs(error));
                total_target_flux += computed;
                ++dof;
            }
        }

        const double conservation =
            std::abs(total_target_flux - exact_domain_divergence_integral(smooth_variable_divergence_field));
        const double l2_rel = (flux_den > 1.0e-30) ? std::sqrt(flux_num / flux_den) : std::sqrt(flux_num);

        return CaseMetrics{
            order,
            domain,
            h,
            l2_rel,
            l2_rel,
            linf_flux,
            conservation,
        };
    }

    mimetic::PlanarMomentInterpolator interpolator(mb);
    mimetic::MomentMethodOptions options;
    options.edge_moment_order = order;
    options.cell_moment_order = std::max(1, order - 1);
    options.quadrature_points = 10;
    options.regularization = 1.0e-12;
    options.exact_constraints = false;
    options.edge_weight = 1.0;
    options.cell_weight = 1.0;

    for (const CellInfo& source : source_cells) {
        set_source_moments(mb, interpolator, source.handle, order, options.cell_moment_order,
                           smooth_variable_divergence_field);
        interpolator.reconstruct_source_polygon(source.handle, options);
    }

    const mimetic::EdgeMomentTransferResult transfer =
        interpolator.transfer_source_to_target_edge_moments(handles(source_cells), handles(target_cells), order);

    double flux_num = 0.0;
    double flux_den = 0.0;
    double all_num = 0.0;
    double all_den = 0.0;
    double linf_all = 0.0;
    double total_target_flux = 0.0;

    std::size_t dof = 0;
    for (const CellInfo& target : target_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, target.handle);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        for (const mimetic::LocalEdge& edge : edges) {
            const Eigen::Vector2d a = poly.centroid + edge.a;
            const Eigen::Vector2d b = poly.centroid + edge.b;
            const std::vector<double> exact =
                exact_edge_moments_absolute(a, b, order, smooth_variable_divergence_field);
            const std::vector<double>& computed = transfer.target_moments[dof];

            const double flux_error = computed[0] - exact[0];
            flux_num += flux_error * flux_error;
            flux_den += exact[0] * exact[0];
            total_target_flux += computed[0];

            for (int degree = 0; degree <= order; ++degree) {
                const double error = computed[degree] - exact[degree];
                all_num += error * error;
                all_den += exact[degree] * exact[degree];
                linf_all = std::max(linf_all, std::abs(error));
            }
            ++dof;
        }
    }

    const double conservation =
        std::abs(total_target_flux - exact_domain_divergence_integral(smooth_variable_divergence_field));

    return CaseMetrics{
        order,
        domain,
        h,
        (flux_den > 1.0e-30) ? std::sqrt(flux_num / flux_den) : std::sqrt(flux_num),
        (all_den > 1.0e-30) ? std::sqrt(all_num / all_den) : std::sqrt(all_num),
        linf_all,
        conservation,
    };
}

void print_header(const std::string& domain)
{
    std::cout << "\n=== High-Order H(div) Convergence: " << domain << " ===\n";
    std::cout << std::left << std::setw(8) << "p"
              << std::setw(12) << "h"
              << std::setw(16) << "L2_flux"
              << std::setw(16) << "L2_all"
              << std::setw(16) << "Linf_all"
              << std::setw(16) << "Conserv"
              << std::setw(12) << "rate_flux"
              << std::setw(12) << "rate_all"
              << "\n";
}

void print_row(const CaseMetrics& m, const double rate_flux, const double rate_all)
{
    std::cout << std::left << std::setw(8) << m.order
              << std::scientific << std::setprecision(3)
              << std::setw(12) << m.h
              << std::setw(16) << m.l2_flux_rel
              << std::setw(16) << m.l2_all_rel
              << std::setw(16) << m.linf_all
              << std::setw(16) << m.conservation;
    if (rate_flux > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << rate_flux;
    } else {
        std::cout << std::setw(12) << "---";
    }
    if (rate_all > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << rate_all;
    } else {
        std::cout << std::setw(12) << "---";
    }
    std::cout << "\n";
}

struct RefinementLevel {
    int source_resolution = 0;
    int target_resolution = 0;
};

bool run_domain(const std::string& domain,
                const std::vector<RefinementLevel>& levels,
                const bool voronoi,
                std::ofstream* csv)
{
    print_header(domain);
    bool ok = true;

    for (int order = 1; order <= 3; ++order) {
        std::vector<CaseMetrics> metrics;
        for (std::size_t level = 0; level < levels.size(); ++level) {
            moab::Core mb;
            std::vector<CellInfo> source_cells;
            std::vector<CellInfo> target_cells;
            double h = 0.0;

            if (voronoi) {
                source_cells = create_voronoi_mesh(mb, halton_seeds(levels[level].source_resolution, 0));
                target_cells = create_voronoi_mesh(mb, halton_seeds(levels[level].target_resolution, 500));
                h = 1.0 / std::sqrt(static_cast<double>(std::max(levels[level].source_resolution,
                                                                 levels[level].target_resolution)));
            } else {
                source_cells = create_quad_mesh(mb, levels[level].source_resolution, levels[level].source_resolution);
                target_cells = create_quad_mesh(mb, levels[level].target_resolution, levels[level].target_resolution);
                h = 1.0 / static_cast<double>(std::max(levels[level].source_resolution,
                                                       levels[level].target_resolution));
            }

            metrics.push_back(run_case(domain, mb, source_cells, target_cells, order, h));
        }

        double rate_sum_flux = 0.0;
        double rate_sum_all = 0.0;
        int rate_count = 0;
        for (std::size_t i = 0; i < metrics.size(); ++i) {
            double rate_flux = 0.0;
            double rate_all = 0.0;
            if (i > 0) {
                rate_flux = convergence_rate(metrics[i - 1].l2_flux_rel, metrics[i].l2_flux_rel,
                                             metrics[i - 1].h, metrics[i].h);
                rate_all = convergence_rate(metrics[i - 1].l2_all_rel, metrics[i].l2_all_rel,
                                            metrics[i - 1].h, metrics[i].h);
                rate_sum_flux += rate_flux;
                rate_sum_all += rate_all;
                ++rate_count;
            }
            print_row(metrics[i], rate_flux, rate_all);
            if (csv) {
                (*csv) << domain << "," << order << "," << i << ","
                       << metrics[i].h << "," << metrics[i].l2_flux_rel << ","
                       << metrics[i].l2_all_rel << "," << metrics[i].linf_all << ","
                       << metrics[i].conservation << "\n";
            }
            ok = mimetic::test::near(metrics[i].conservation, 0.0, 5.0e-13,
                                     domain + " p=" + std::to_string(order) + " conservation") && ok;
        }

        const double avg_rate_flux = (rate_count > 0) ? (rate_sum_flux / rate_count) : 0.0;
        const double avg_rate_all = (rate_count > 0) ? (rate_sum_all / rate_count) : 0.0;
        const double min_expected_flux = voronoi ? (0.65 * (order + 1.0)) : (0.85 * (order + 1.0));
        const double min_expected_all = voronoi ? (0.60 * (order + 1.0)) : (0.80 * (order + 1.0));

        std::cout << "  avg rate p=" << order
                  << " flux=" << std::fixed << std::setprecision(2) << avg_rate_flux
                  << " all=" << avg_rate_all << "\n";

        if (avg_rate_flux < min_expected_flux || avg_rate_all < min_expected_all) {
            std::cout << "  [FAIL] " << domain << " p=" << order
                      << " average convergence rates below expectation\n";
            ok = false;
        }
    }

    return ok;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::ofstream csv;
        if (argc > 1) {
            csv.open(argv[1]);
            if (!csv) {
                throw std::runtime_error("Failed to open CSV output path");
            }
            csv << "domain,order,level,h,l2_flux_rel,l2_all_rel,linf_all,conservation\n";
        }

        const std::vector<RefinementLevel> quad_levels = {
            {4, 5}, {8, 9}, {16, 17},
        };
        const std::vector<RefinementLevel> voronoi_levels = {
            {49, 64}, {100, 121}, {196, 225},
        };

        bool ok = true;
        ok = run_domain("quad_to_quad", quad_levels, false, csv.is_open() ? &csv : nullptr) && ok;
        ok = run_domain("voronoi_to_voronoi", voronoi_levels, true, csv.is_open() ? &csv : nullptr) && ok;

        if (!ok) {
            throw std::runtime_error("High-order H(div) convergence validation failed");
        }

        std::cout << "\n[SUCCESS] High-order polygonal H(div)-style moment transfer converges for p=1,2,3.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
