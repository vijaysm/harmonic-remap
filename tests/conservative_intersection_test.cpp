#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

struct Rect {
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    moab::EntityHandle cell;
};

std::vector<Rect> create_rect_mesh(moab::Core& mb, const std::vector<double>& xs, const std::vector<double>& ys)
{
    std::vector<Rect> cells;
    for (std::size_t j = 0; j + 1 < ys.size(); ++j) {
        for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
            const std::array<Eigen::Vector2d, 4> points = {{
                Eigen::Vector2d(xs[i], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j + 1]),
                Eigen::Vector2d(xs[i], ys[j + 1]),
            }};
            cells.push_back(Rect{xs[i], xs[i + 1], ys[j], ys[j + 1], mimetic::create_quad(mb, points)});
        }
    }
    return cells;
}

bool rect_intersection(const Rect& a, const Rect& b, Rect& intersection)
{
    intersection.xmin = std::max(a.xmin, b.xmin);
    intersection.xmax = std::min(a.xmax, b.xmax);
    intersection.ymin = std::max(a.ymin, b.ymin);
    intersection.ymax = std::min(a.ymax, b.ymax);
    intersection.cell = 0;
    return intersection.xmax > intersection.xmin + mimetic::kTolerance &&
           intersection.ymax > intersection.ymin + mimetic::kTolerance;
}

double rect_area(const Rect& rect)
{
    return (rect.xmax - rect.xmin) * (rect.ymax - rect.ymin);
}

std::vector<moab::EntityHandle> cell_handles(const std::vector<Rect>& cells)
{
    std::vector<moab::EntityHandle> handles;
    handles.reserve(cells.size());
    for (const Rect& cell : cells) {
        handles.push_back(cell.cell);
    }
    return handles;
}

std::vector<Eigen::Vector2d> rect_points_in_source_frame(const Rect& rect, const mimetic::LocalPolygon& source)
{
    return {
        Eigen::Vector2d(rect.xmin, rect.ymin) - source.centroid,
        Eigen::Vector2d(rect.xmax, rect.ymin) - source.centroid,
        Eigen::Vector2d(rect.xmax, rect.ymax) - source.centroid,
        Eigen::Vector2d(rect.xmin, rect.ymax) - source.centroid,
    };
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Conservative Source/Target Intersection Test ---\n\n";

        moab::Core mb;
        mimetic::MimeticInterpolator interpolator(mb);

        const std::vector<Rect> source_cells = create_rect_mesh(mb, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
        const std::vector<Rect> target_cells = create_rect_mesh(mb, {0.0, 0.25, 0.65, 1.0}, {0.0, 0.40, 0.70, 1.0});
        const std::vector<moab::EntityHandle> source_handles = cell_handles(source_cells);
        const std::vector<moab::EntityHandle> target_handles = cell_handles(target_cells);

        for (const Rect& source : source_cells) {
            mimetic::test::set_source_fluxes_from_absolute_field(
                mb, interpolator.source_flux_tag(), source.cell, mimetic::test::linear_absolute_field);
            interpolator.reconstruct_source_polygon(source.cell);
        }

        bool ok = true;
        double total_overlap_area = 0.0;
        double total_boundary_flux = 0.0;
        double total_expected_flux = 0.0;
        int overlap_count = 0;

        for (const Rect& source_rect : source_cells) {
            const mimetic::LocalPolygon source_poly = mimetic::local_polygon(mb, source_rect.cell);
            
            const void* ptr = nullptr;
            int size = 0;
            mimetic::check_moab(mb.tag_get_by_ptr(interpolator.coeffs_tag(), &source_rect.cell, 1, &ptr, &size),
                                "Failed to read source coefficients");
            
            const double* d_ptr = static_cast<const double*>(ptr);
            mimetic::ReconstructionCoeffs coeffs{};
            if (size > 0) {
                coeffs.d = d_ptr[0];
                coeffs.harmonic.assign(d_ptr + 1, d_ptr + size);
            }

            for (const Rect& target_rect : target_cells) {
                Rect overlap{};
                if (!rect_intersection(source_rect, target_rect, overlap)) {
                    continue;
                }

                const std::vector<Eigen::Vector2d> overlap_points = rect_points_in_source_frame(overlap, source_poly);
                const double boundary_flux = interpolator.polygon_boundary_flux(coeffs, overlap_points);
                const double expected_flux = coeffs.d * rect_area(overlap);

                ++overlap_count;
                total_overlap_area += rect_area(overlap);
                total_boundary_flux += boundary_flux;
                total_expected_flux += expected_flux;

                ok = mimetic::test::near(boundary_flux, expected_flux, mimetic::kConservationTolerance,
                                         "overlap " + std::to_string(overlap_count) + " boundary flux") &&
                     ok;
            }
        }

        std::cout << "\nAggregate overlap checks:\n";
        ok = mimetic::test::near(total_overlap_area, 1.0, mimetic::kTolerance, "source-target overlap area") && ok;
        ok = mimetic::test::near(total_boundary_flux, total_expected_flux, mimetic::kConservationTolerance,
                                 "summed overlap divergence theorem") &&
             ok;
        ok = mimetic::test::near(total_expected_flux, 2.0, mimetic::kConservationTolerance,
                                 "exact integral of div(x,y)") && ok;

        std::cout << "\nEdge-wise source-to-target transfer checks:\n";
        const mimetic::EdgeTransferResult edge_transfer =
            interpolator.transfer_source_to_target_edges(source_handles, target_handles);
        ok = mimetic::test::near(static_cast<double>(edge_transfer.target_edges.size()), 36.0, mimetic::kTolerance,
                                 "directed target edge DOFs") &&
             ok;
        std::vector<double> coverage(edge_transfer.target_edges.size(), 0.0);
        for (const mimetic::EdgeTransferContribution& contrib : edge_transfer.contributions) {
            coverage[contrib.target_dof_index] += (contrib.segment_b - contrib.segment_a).norm();
        }

        std::size_t target_dof = 0;
        for (const Rect& target_rect : target_cells) {
            const mimetic::LocalPolygon target_poly = mimetic::local_polygon(mb, target_rect.cell);
            const std::vector<mimetic::LocalEdge> target_edges = mimetic::local_edges(mb, target_poly);
            double target_cell_flux = 0.0;
            for (const mimetic::LocalEdge& edge : target_edges) {
                const Eigen::Vector2d a = target_poly.centroid + edge.a;
                const Eigen::Vector2d b = target_poly.centroid + edge.b;
                const double exact_flux =
                    mimetic::test::directed_edge_flux_from_absolute_field(a, b, mimetic::test::linear_absolute_field);
                target_cell_flux += edge_transfer.target_fluxes[target_dof];
                ok = mimetic::test::near(edge_transfer.target_fluxes[target_dof], exact_flux,
                                         mimetic::kConservationTolerance,
                                         "target directed edge " + std::to_string(target_dof)) &&
                     ok;

                double tagged_flux = 0.0;
                mimetic::check_moab(mb.tag_get_data(interpolator.target_flux_tag(), &edge.handle, 1, &tagged_flux),
                                    "Failed to read target flux tag");
                ok = mimetic::test::near(tagged_flux, exact_flux, mimetic::kConservationTolerance,
                                         "target flux tag " + std::to_string(target_dof)) &&
                     ok;
                ok = mimetic::test::near(coverage[target_dof], edge.length, mimetic::kConservationTolerance,
                                         "target edge coverage " + std::to_string(target_dof)) &&
                     ok;
                ++target_dof;
            }
            ok = mimetic::test::near(target_cell_flux, 2.0 * rect_area(target_rect),
                                     mimetic::kConservationTolerance,
                                     "target cell edge-flux divergence") &&
                 ok;
        }

        const mimetic::SparseEdgeProjection projection =
            interpolator.assemble_edge_projection_operator(source_handles, target_handles);
        Eigen::VectorXd source_flux_vector(projection.source_edges.size());
        for (std::size_t i = 0; i < projection.source_edges.size(); ++i) {
            const mimetic::LocalPolygon source_poly = mimetic::local_polygon(mb, projection.source_edges[i].polygon);
            const std::vector<mimetic::LocalEdge> source_edges = mimetic::local_edges(mb, source_poly);
            const mimetic::LocalEdge& edge = source_edges[projection.source_edges[i].local_edge_index];
            source_flux_vector(static_cast<Eigen::Index>(i)) =
                mimetic::test::directed_edge_flux_from_absolute_field(source_poly.centroid + edge.a, source_poly.centroid + edge.b,
                                                                      mimetic::test::linear_absolute_field);
        }
        const Eigen::VectorXd projected_fluxes = projection.matrix * source_flux_vector;
        ok = mimetic::test::near(static_cast<double>(projection.matrix.rows()), 36.0, mimetic::kTolerance,
                                 "projection rows") &&
             ok;
        ok = mimetic::test::near(static_cast<double>(projection.matrix.cols()), 16.0, mimetic::kTolerance,
                                 "projection columns") &&
             ok;
        for (Eigen::Index i = 0; i < projected_fluxes.size(); ++i) {
            ok = mimetic::test::near(projected_fluxes(i), edge_transfer.target_fluxes[static_cast<std::size_t>(i)],
                                     mimetic::kConservationTolerance,
                                     "matrix-applied target edge " + std::to_string(i)) &&
                 ok;
        }

        const std::string matrix_path = "/tmp/mimetic_rect_edge_projection.mtx";
        const std::string source_map_path = "/tmp/mimetic_rect_source_edges.csv";
        const std::string target_map_path = "/tmp/mimetic_rect_target_edges.csv";
        mimetic::write_matrix_market(projection, matrix_path, source_map_path, target_map_path);
        std::ifstream matrix_file(matrix_path.c_str());
        std::string matrix_header;
        std::getline(matrix_file, matrix_header);
        ok = mimetic::test::near(matrix_header == "%%MatrixMarket matrix coordinate real general" ? 1.0 : 0.0, 1.0,
                                 mimetic::kTolerance, "MatrixMarket header") &&
             ok;

        if (!ok) {
            std::cout << "\n[FAILED] Conservative intersection test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Every source-target intersection integrates the reconstructed field exactly.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
