#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

struct CaseMetrics {
    int source_n;
    int target_n;
    std::size_t source_cells;
    std::size_t target_cells;
    std::size_t target_edges;
    double source_divergence;
    double target_divergence;
    double global_conservation_residual;
    double max_target_cell_reintegration_residual;
    double max_direct_sparse_delta;
    double max_conforming_target_cell_residual;
    double conforming_correction_l2_relative;
    double l2_relative_flux_error;
    double max_flux_error;
    double direct_cell_average_l1_error;
    double direct_cell_average_linf_error;
    double reconstructed_cell_average_l1_error;
    double reconstructed_cell_average_linf_error;
};

int parse_positive_int(const char* text, const char* name)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

std::vector<moab::EntityHandle> handles_from_transfer_edges(const mimetic::EdgeTransferResult& transfer)
{
    std::vector<moab::EntityHandle> handles;
    handles.reserve(transfer.target_edges.size());
    for (const mimetic::DirectedEdgeDof& dof : transfer.target_edges) {
        handles.push_back(dof.edge);
    }
    return handles;
}

moab::Tag scalar_tag(moab::Core& mb, const std::string& name)
{
    moab::Tag tag = 0;
    const double default_value = 0.0;
    mimetic::check_moab(mb.tag_get_handle(name.c_str(), 1, moab::MB_TYPE_DOUBLE, tag,
                                          moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_value),
                        "Failed to create scalar diagnostic tag " + name);
    return tag;
}

moab::Tag vector_tag(moab::Core& mb, const std::string& name)
{
    moab::Tag tag = 0;
    const double default_value[3] = {0.0, 0.0, 0.0};
    mimetic::check_moab(mb.tag_get_handle(name.c_str(), 3, moab::MB_TYPE_DOUBLE, tag,
                                          moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_value),
                        "Failed to create vector diagnostic tag " + name);
    return tag;
}

CaseMetrics run_case(const int source_n,
                     const int target_n,
                     const std::string& output_prefix,
                     const bool write_vtk,
                     const bool metric_weighted = true)
{
    moab::Core mb;
    mimetic::MimeticInterpolator interpolator(mb);
    mimetic::GeometryOptions options;
    options.mode = mimetic::GeometryMode::SphericalGnomonic;
    options.conservation_tolerance = mimetic::kConservationTolerance;
    options.metric_weighted = metric_weighted;
    interpolator.set_geometry_options(options);

    const std::vector<moab::EntityHandle> source_mesh =
        mimetic::test_sphere::generate_cubed_sphere(mb, source_n);
    const std::vector<moab::EntityHandle> target_mesh =
        mimetic::test_sphere::generate_cubed_sphere(mb, target_n);

    const moab::Tag source_div_tag = scalar_tag(mb, "SOURCE_DIV_EXACT");
    const moab::Tag target_div_tag = scalar_tag(mb, "TARGET_DIV_RECON");
    const moab::Tag target_flux_error_tag = scalar_tag(mb, "TARGET_FLUX_ERROR");
    const moab::Tag target_field_error_norm_tag = scalar_tag(mb, "TARGET_FIELD_ERROR_NORM");
    const moab::Tag target_field_direct_error_norm_tag = scalar_tag(mb, "TARGET_FIELD_DIRECT_ERROR_NORM");
    const moab::Tag target_field_recon_tag = vector_tag(mb, "TARGET_FIELD_RECON");
    const moab::Tag target_field_direct_tag = vector_tag(mb, "TARGET_FIELD_DIRECT");
    const moab::Tag target_field_exact_tag = vector_tag(mb, "TARGET_FIELD_EXACT");
    const moab::Tag target_field_error_tag = vector_tag(mb, "TARGET_FIELD_ERROR");
    const moab::Tag target_field_direct_error_tag = vector_tag(mb, "TARGET_FIELD_DIRECT_ERROR");

    double total_source_divergence = 0.0;
    mimetic::test_sphere::set_conservative_source_fluxes(
        interpolator, mb, source_mesh, mimetic::test_sphere::spherical_harmonic_gradient);
    for (const moab::EntityHandle cell : source_mesh) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, options);
        double cell_divergence = 0.0;
        for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
            const moab::EntityHandle edge =
                mimetic::find_or_create_edge(mb, poly.vertices[i], poly.vertices[(i + 1) % poly.vertices.size()]);
            cell_divergence += interpolator.source_edge_flux(cell, i, edge);
        }
        total_source_divergence += cell_divergence;
        interpolator.reconstruct_source_polygon(cell);
        mimetic::check_moab(mb.tag_set_data(source_div_tag, &cell, 1, &cell_divergence),
                            "Failed to write source divergence diagnostic");
    }

    const mimetic::EdgeTransferResult transfer =
        interpolator.transfer_source_to_target_edges(source_mesh, target_mesh);
    const mimetic::ConformingEdgeTransferResult conforming =
        interpolator.project_target_fluxes_to_hdiv_conforming(source_mesh, target_mesh, transfer);
    const mimetic::CellAverageTransferResult direct_cell =
        interpolator.transfer_source_to_target_cell_averages(
            source_mesh, target_mesh, mimetic::CellAverageReductionMode::Harmonic);
    const mimetic::SparseEdgeProjection projection =
        interpolator.assemble_edge_projection_operator(source_mesh, target_mesh);
    const double direct_sparse_delta =
        mimetic::test_sphere::max_direct_sparse_delta(interpolator, projection, transfer);
    const std::map<std::pair<moab::EntityHandle, std::size_t>, double> target_exact_fluxes =
        mimetic::test_sphere::conservative_edge_fluxes(
            mb, target_mesh, mimetic::test_sphere::spherical_harmonic_gradient);

    double total_target_divergence = 0.0;
    double l2_num = 0.0;
    double l2_den = 0.0;
    double max_flux_error = 0.0;
    double max_reintegration_residual = 0.0;
    double max_conforming_target_cell_residual = 0.0;
    double direct_cell_average_l1_error = 0.0;
    double direct_cell_average_linf_error = 0.0;
    double reconstructed_cell_average_l1_error = 0.0;
    double reconstructed_cell_average_linf_error = 0.0;
    double correction_num = 0.0;
    double correction_den = 0.0;
    std::size_t dof = 0;

    for (std::size_t cell_index = 0; cell_index < target_mesh.size(); ++cell_index) {
        const moab::EntityHandle cell = target_mesh[cell_index];
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, options);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double cell_divergence = 0.0;
        double conforming_cell_divergence = 0.0;
        double cell_max_flux_error = 0.0;

        for (std::size_t i = 0; i < edges.size(); ++i) {
            const double exact = target_exact_fluxes.at(std::make_pair(cell, i));
            const double transferred = transfer.target_fluxes[dof++];
            const double conforming_transferred = conforming.target_fluxes[dof - 1];
            const double error = std::abs(transferred - exact);
            l2_num += error * error;
            l2_den += exact * exact;
            max_flux_error = std::max(max_flux_error, error);
            cell_max_flux_error = std::max(cell_max_flux_error, error);
            cell_divergence += transferred;
            conforming_cell_divergence += conforming_transferred;
            correction_num += (conforming_transferred - transferred) * (conforming_transferred - transferred);
            correction_den += conforming_transferred * conforming_transferred;
            interpolator.set_source_edge_flux(cell, i, transferred);
        }

        total_target_divergence += cell_divergence;
        max_conforming_target_cell_residual =
            std::max(max_conforming_target_cell_residual,
                     std::abs(conforming_cell_divergence - conforming.target_divergence_integrals[cell_index]));
        interpolator.reconstruct_source_polygon(cell);
        const mimetic::ReconstructionCoeffs target_coeffs = mimetic::test_sphere::read_coeffs(mb, interpolator, cell);
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const double reintegrated = interpolator.edge_flux(target_coeffs, edges[i].a, edges[i].b);
            const double transferred = interpolator.source_edge_flux(cell, i, edges[i].handle);
            max_reintegration_residual =
                std::max(max_reintegration_residual, std::abs(reintegrated - transferred));
        }

        const Eigen::Vector3d field_direct = direct_cell.target_averages[cell_index];
        const Eigen::Vector3d field_recon = interpolator.cell_average(cell);
        const Eigen::Vector3d field_exact =
            mimetic::test_sphere::exact_surface_cell_average(
                mb, cell, mimetic::test_sphere::spherical_harmonic_gradient);
        const Eigen::Vector3d field_error = field_recon - field_exact;
        const Eigen::Vector3d direct_field_error = field_direct - field_exact;
        const double field_error_norm = field_error.norm();
        const double direct_field_error_norm = direct_field_error.norm();
        const double area_weight = poly.spherical_area > 0.0 ? poly.spherical_area : poly.area;
        reconstructed_cell_average_l1_error += area_weight * field_error_norm;
        reconstructed_cell_average_linf_error =
            std::max(reconstructed_cell_average_linf_error, field_error_norm);
        direct_cell_average_l1_error += area_weight * direct_field_error_norm;
        direct_cell_average_linf_error = std::max(direct_cell_average_linf_error, direct_field_error_norm);

        const double direct_data[3] = {field_direct.x(), field_direct.y(), field_direct.z()};
        const double recon_data[3] = {field_recon.x(), field_recon.y(), field_recon.z()};
        const double exact_data[3] = {field_exact.x(), field_exact.y(), field_exact.z()};
        const double error_data[3] = {field_error.x(), field_error.y(), field_error.z()};
        const double direct_error_data[3] = {direct_field_error.x(), direct_field_error.y(), direct_field_error.z()};
        mimetic::check_moab(mb.tag_set_data(target_div_tag, &cell, 1, &cell_divergence),
                            "Failed to write target divergence diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_direct_tag, &cell, 1, direct_data),
                            "Failed to write direct field diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_recon_tag, &cell, 1, recon_data),
                            "Failed to write reconstructed field diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_exact_tag, &cell, 1, exact_data),
                            "Failed to write exact field diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_error_tag, &cell, 1, error_data),
                            "Failed to write reconstructed field error diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_direct_error_tag, &cell, 1, direct_error_data),
                            "Failed to write direct field error diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_error_norm_tag, &cell, 1, &field_error_norm),
                            "Failed to write field error norm diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_field_direct_error_norm_tag, &cell, 1, &direct_field_error_norm),
                            "Failed to write direct field error norm diagnostic");
        mimetic::check_moab(mb.tag_set_data(target_flux_error_tag, &cell, 1, &cell_max_flux_error),
                            "Failed to write target flux error diagnostic");
    }

    if (write_vtk) {
        const std::string source_path = output_prefix + "_source.vtk";
        const std::string target_path = output_prefix + "_target.vtk";
        moab::EntityHandle source_set = 0;
        moab::EntityHandle target_set = 0;
        mimetic::check_moab(mb.create_meshset(moab::MESHSET_SET, source_set), "Failed to create source meshset");
        mimetic::check_moab(mb.add_entities(source_set, source_mesh.data(), source_mesh.size()),
                            "Failed to populate source meshset");
        mimetic::check_moab(mb.write_file(source_path.c_str(), nullptr, nullptr, &source_set, 1),
                            "Failed to write source VTK output");
        mimetic::check_moab(mb.create_meshset(moab::MESHSET_SET, target_set), "Failed to create target meshset");
        mimetic::check_moab(mb.add_entities(target_set, target_mesh.data(), target_mesh.size()),
                            "Failed to populate target meshset");
        mimetic::check_moab(mb.write_file(target_path.c_str(), nullptr, nullptr, &target_set, 1),
                            "Failed to write target VTK output");
        std::cout << "  Wrote " << source_path << " and " << target_path << "\n";
    }

    CaseMetrics metrics{};
    metrics.source_n = source_n;
    metrics.target_n = target_n;
    metrics.source_cells = source_mesh.size();
    metrics.target_cells = target_mesh.size();
    metrics.target_edges = transfer.target_edges.size();
    metrics.source_divergence = total_source_divergence;
    metrics.target_divergence = total_target_divergence;
    metrics.global_conservation_residual = std::abs(total_source_divergence - total_target_divergence);
    metrics.max_target_cell_reintegration_residual = max_reintegration_residual;
    metrics.max_direct_sparse_delta = direct_sparse_delta;
    metrics.max_conforming_target_cell_residual = max_conforming_target_cell_residual;
    metrics.conforming_correction_l2_relative =
        (correction_den > std::numeric_limits<double>::epsilon()) ? std::sqrt(correction_num / correction_den)
                                                                  : std::sqrt(correction_num);
    metrics.l2_relative_flux_error =
        (l2_den > std::numeric_limits<double>::epsilon()) ? std::sqrt(l2_num / l2_den) : std::sqrt(l2_num);
    metrics.max_flux_error = max_flux_error;
    metrics.direct_cell_average_l1_error = direct_cell_average_l1_error;
    metrics.direct_cell_average_linf_error = direct_cell_average_linf_error;
    metrics.reconstructed_cell_average_l1_error = reconstructed_cell_average_l1_error;
    metrics.reconstructed_cell_average_linf_error = reconstructed_cell_average_linf_error;
    return metrics;
}

void print_case_metrics(const CaseMetrics& m)
{
    std::cout << "  source_n=" << m.source_n << " target_n=" << m.target_n
              << " source_cells=" << m.source_cells
              << " target_cells=" << m.target_cells
              << " target_edges=" << m.target_edges << "\n";
    std::cout << std::scientific << std::setprecision(6)
              << "  source_div=" << m.source_divergence
              << " target_div=" << m.target_divergence
              << " global_residual=" << m.global_conservation_residual << "\n"
              << "  direct_sparse_delta=" << m.max_direct_sparse_delta
              << " target_reintegration=" << m.max_target_cell_reintegration_residual
              << " conforming_target_residual=" << m.max_conforming_target_cell_residual << "\n"
              << "  conforming_correction_l2_rel=" << m.conforming_correction_l2_relative << "\n"
              << "  edge_flux_l2_rel=" << m.l2_relative_flux_error
              << " edge_flux_linf=" << m.max_flux_error << "\n"
              << "  direct_cell_avg_l1=" << m.direct_cell_average_l1_error
              << " direct_cell_avg_linf=" << m.direct_cell_average_linf_error << "\n"
              << "  recon_cell_avg_l1=" << m.reconstructed_cell_average_l1_error
              << " recon_cell_avg_linf=" << m.reconstructed_cell_average_linf_error << "\n";
}

bool invariants_pass(const CaseMetrics& m)
{
    return m.global_conservation_residual <= mimetic::kConservationTolerance &&
           m.max_target_cell_reintegration_residual <= mimetic::kConservationTolerance &&
           m.max_direct_sparse_delta <= mimetic::kConservationTolerance &&
           m.max_conforming_target_cell_residual <= mimetic::kConservationTolerance;
}

void print_usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " [source_n target_n output_prefix]\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::cout << "--- Spherical Cubed-Sphere Mimetic Edge Transfer Test ---\n\n";

        if (argc != 1 && argc != 4) {
            print_usage(argv[0]);
            return 2;
        }

        if (argc == 4) {
            const int source_n = parse_positive_int(argv[1], "source_n");
            const int target_n = parse_positive_int(argv[2], "target_n");
            const CaseMetrics metrics = run_case(source_n, target_n, argv[3], true);
            print_case_metrics(metrics);
            bool ok = invariants_pass(metrics);
            if (source_n == target_n) {
                ok = (metrics.max_flux_error <= mimetic::kConservationTolerance) && ok;
            }
            if (!ok) {
                std::cout << "\n[FAILED] Spherical cubed-sphere case failed acceptance checks.\n";
                return 1;
            }
            std::cout << "\n[SUCCESS] Spherical cubed-sphere case passed.\n";
            return 0;
        }

        const CaseMetrics identity = run_case(4, 4, "cubed_sphere_identity", false);
        const CaseMetrics refine = run_case(4, 6, "cubed_sphere", true);
        const CaseMetrics coarsen = run_case(6, 4, "cubed_sphere_coarsen", false);
        const CaseMetrics fine = run_case(6, 8, "cubed_sphere_fine", false);
        const CaseMetrics refine_unweighted = run_case(4, 6, "cubed_sphere_unweighted", false, false);

        std::cout << "Identity case:\n";
        print_case_metrics(identity);
        std::cout << "Refinement case:\n";
        print_case_metrics(refine);
        std::cout << "Coarsening case:\n";
        print_case_metrics(coarsen);
        std::cout << "Fine refinement case:\n";
        print_case_metrics(fine);
        std::cout << "Unweighted parity case:\n";
        print_case_metrics(refine_unweighted);

        bool ok = invariants_pass(identity) && invariants_pass(refine) &&
                  invariants_pass(coarsen) && invariants_pass(fine) &&
                  invariants_pass(refine_unweighted);
        ok = (identity.max_flux_error <= mimetic::kConservationTolerance) && ok;
        ok = (fine.l2_relative_flux_error < refine.l2_relative_flux_error) && ok;
        ok = (fine.max_flux_error < refine.max_flux_error) && ok;
        ok = (fine.direct_cell_average_l1_error < refine.direct_cell_average_l1_error) && ok;
        ok = (fine.reconstructed_cell_average_l1_error < refine.reconstructed_cell_average_l1_error) && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Spherical cubed-sphere transfer did not meet acceptance checks.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Spherical structured transfer is conservative, sparse-consistent, and convergent.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
