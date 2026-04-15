#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <map>
#include <array>
#include <string>
#include <stdexcept>
#include "moab/Core.hpp"
#include "mimetic/mimetic.hpp"

// Gauss-Legendre quadrature points and weights for [-1, 1] (n=7)
const double gauss_pts[7] = {
    0.0,
    -0.4058451513773972, 0.4058451513773972,
    -0.7415311855993945, 0.7415311855993945,
    -0.9491079123427585, 0.9491079123427585
};
const double gauss_wts[7] = {
    0.4179591836734694,
    0.3818300505051189, 0.3818300505051189,
    0.2797053914892766, 0.2797053914892766,
    0.1294849661688697, 0.1294849661688697
};

// Project a point on the cube to the unit sphere
void project_to_sphere(double x, double y, double z, double out[3]) {
    double mag = std::sqrt(x*x + y*y + z*z);
    out[0] = x / mag;
    out[1] = y / mag;
    out[2] = z / mag;
}

std::vector<moab::EntityHandle> generate_cubed_sphere(moab::Core& mb, int N) {
    std::vector<moab::EntityHandle> elements;
    std::map<std::array<int, 3>, moab::EntityHandle> vertex_map;
    
    auto get_or_create_vertex = [&](int ix, int iy, int iz) -> moab::EntityHandle {
        std::array<int, 3> key = {ix, iy, iz};
        if (vertex_map.count(key) == 0) {
            double x = (double)ix / N;
            double y = (double)iy / N;
            double z = (double)iz / N;
            double p[3];
            project_to_sphere(x, y, z, p);
            moab::EntityHandle v;
            mb.create_vertex(p, v);
            vertex_map[key] = v;
        }
        return vertex_map[key];
    };

    auto build_face = [&](int dim1, int dim2, int fixed_dim, int fixed_val, bool flip) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int u1 = -N + 2 * i;
                int u2 = -N + 2 * (i + 1);
                int v1 = -N + 2 * j;
                int v2 = -N + 2 * (j + 1);
                
                int c[4][3];
                for (int k = 0; k < 4; ++k) {
                    int u = (k == 0 || k == 3) ? u1 : u2;
                    int v = (k == 0 || k == 1) ? v1 : v2;
                    c[k][dim1] = u;
                    c[k][dim2] = v;
                    c[k][fixed_dim] = fixed_val;
                }
                
                moab::EntityHandle conn[4];
                for (int k = 0; k < 4; ++k) {
                    conn[k] = get_or_create_vertex(c[k][0], c[k][1], c[k][2]);
                }
                
                if (flip) {
                    std::swap(conn[1], conn[3]);
                }
                
                moab::EntityHandle quad;
                mb.create_element(moab::MBQUAD, conn, 4, quad);
                elements.push_back(quad);
            }
        }
    };
    
    build_face(0, 1, 2,  N, false); // +Z
    build_face(0, 1, 2, -N, true);  // -Z
    build_face(1, 2, 0,  N, false); // +X
    build_face(1, 2, 0, -N, true);  // -X
    build_face(2, 0, 1,  N, false); // +Y
    build_face(2, 0, 1, -N, true);  // -Y
    
    return elements;
}

// Analytical vector field: Gradient of Y_2^0 harmonic
// f(x,y,z) = 0.5 * (3z^2 - 1)
// u = grad_S f = grad_{3D} f - (grad_{3D} f . r) r
Eigen::Vector3d spherical_harmonic_gradient(const Eigen::Vector3d& p) {
    Eigen::Vector3d grad_3d(0.0, 0.0, 3.0 * p.z());
    double dot = grad_3d.dot(p);
    return grad_3d - dot * p;
}

// Exact normal flux across a great circle arc from A to B
double exact_edge_flux(const Eigen::Vector3d& A, const Eigen::Vector3d& B) {
    double dot = std::max(-1.0, std::min(1.0, A.dot(B)));
    double theta = std::acos(dot);
    if (theta < 1e-12) return 0.0;
    
    Eigen::Vector3d N = A.cross(B);
    double N_norm = N.norm();
    if (N_norm < 1e-12) return 0.0;
    N /= N_norm;
    
    Eigen::Vector3d T = (B - dot * A).normalized();
    
    double integral = 0.0;
    for (int i = 0; i < 7; ++i) {
        double s = 0.5 * theta * (gauss_pts[i] + 1.0);
        Eigen::Vector3d p = std::cos(s) * A + std::sin(s) * T;
        Eigen::Vector3d u = spherical_harmonic_gradient(p);
        integral += gauss_wts[i] * u.dot(N);
    }
    integral *= 0.5 * theta;
    return integral;
}

int parse_positive_int(const char* text, const char* name)
{
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

void print_usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " [source_n] [target_n] [output_prefix]\n";
}

Eigen::Vector3d reconstructed_vector_from_local_point(const mimetic::MimeticInterpolator& interpolator,
                                                     const mimetic::ReconstructionCoeffs& coeffs,
                                                     const mimetic::LocalPolygon& poly,
                                                     const Eigen::Vector2d& local_point)
{
    const Eigen::Vector2d local_velocity = interpolator.velocity(coeffs, local_point);
    return local_velocity.x() * poly.e_x + local_velocity.y() * poly.e_y;
}

Eigen::Vector3d reconstructed_vector_at_centroid(const mimetic::MimeticInterpolator& interpolator,
                                                 const mimetic::ReconstructionCoeffs& coeffs,
                                                 const mimetic::LocalPolygon& poly)
{
    return reconstructed_vector_from_local_point(interpolator, coeffs, poly, Eigen::Vector2d::Zero());
}

mimetic::ReconstructionCoeffs read_coeffs(moab::Core& mb,
                                          const mimetic::MimeticInterpolator& interpolator,
                                          moab::EntityHandle cell)
{
    const void* ptr = nullptr;
    int size = 0;
    mimetic::check_moab(mb.tag_get_by_ptr(interpolator.coeffs_tag(), &cell, 1, &ptr, &size),
                        "Failed to read reconstruction coefficients");
    const double* data = static_cast<const double*>(ptr);
    mimetic::ReconstructionCoeffs coeffs{};
    coeffs.d = data[0];
    coeffs.harmonic.assign(data + 1, data + size);
    return coeffs;
}

Eigen::Vector2d great_circle_point_in_target_frame(const Eigen::Vector3d& p,
                                                   const mimetic::LocalPolygon& target_poly)
{
    const double dot = p.dot(target_poly.n);
    if (dot < 1e-12) {
        throw std::runtime_error("Great-circle quadrature point cannot be projected to target frame");
    }
    const Eigen::Vector3d p_gnom = p / dot;
    return Eigen::Vector2d(p_gnom.dot(target_poly.e_x) - target_poly.centroid.x(),
                           p_gnom.dot(target_poly.e_y) - target_poly.centroid.y());
}

void edge_error_metrics(const mimetic::MimeticInterpolator& interpolator,
                        const mimetic::ReconstructionCoeffs& coeffs,
                        const mimetic::LocalPolygon& target_poly,
                        const Eigen::Vector3d& A,
                        const Eigen::Vector3d& B,
                        double& l1_error,
                        double& l2_error_squared,
                        double& linf_error)
{
    const double dot = std::max(-1.0, std::min(1.0, A.dot(B)));
    const double theta = std::acos(dot);
    if (theta < 1e-12) {
        return;
    }

    Eigen::Vector3d N = A.cross(B);
    const double N_norm = N.norm();
    if (N_norm < 1e-12) {
        return;
    }
    N /= N_norm;
    const Eigen::Vector3d T = (B - dot * A).normalized();

    double edge_l1 = 0.0;
    double edge_l2 = 0.0;
    double edge_linf = 0.0;
    for (int i = 0; i < 7; ++i) {
        const double s = 0.5 * theta * (gauss_pts[i] + 1.0);
        const Eigen::Vector3d p = std::cos(s) * A + std::sin(s) * T;
        const Eigen::Vector3d exact = spherical_harmonic_gradient(p);
        const Eigen::Vector2d local_point = great_circle_point_in_target_frame(p, target_poly);
        const Eigen::Vector3d recon = reconstructed_vector_from_local_point(interpolator, coeffs, target_poly, local_point);
        const double point_error = std::abs((recon - exact).dot(N));
        edge_l1 += gauss_wts[i] * point_error;
        edge_l2 += gauss_wts[i] * point_error * point_error;
        edge_linf = std::max(edge_linf, point_error);
    }

    const double jacobian = 0.5 * theta;
    l1_error += jacobian * edge_l1;
    l2_error_squared += jacobian * edge_l2;
    linf_error = std::max(linf_error, edge_linf);
}

int main(int argc, char** argv) {
    try {
        std::cout << "--- Spherical Quad Conservation/Convergence Test ---\n\n";

        int source_n = 4;
        int target_n = 6;
        std::string output_prefix = "cubed_sphere";
        if (argc >= 2) {
            source_n = parse_positive_int(argv[1], "source_n");
        }
        if (argc >= 3) {
            target_n = parse_positive_int(argv[2], "target_n");
        }
        if (argc >= 4) {
            output_prefix = argv[3];
        }
        if (argc > 4) {
            print_usage(argv[0]);
            return 2;
        }

        moab::Core mb;
        mimetic::MimeticInterpolator interpolator(mb);
        interpolator.set_spherical(true);

        std::vector<moab::EntityHandle> source_mesh = generate_cubed_sphere(mb, source_n);
        std::vector<moab::EntityHandle> target_mesh = generate_cubed_sphere(mb, target_n);
        
        std::cout << "Generated source cubed-sphere with " << source_mesh.size() << " elements (n=" << source_n << ").\n";
        std::cout << "Generated target cubed-sphere with " << target_mesh.size() << " elements (n=" << target_n << ").\n";

        double default_scalar = 0.0;
        double default_vector[3] = {0.0, 0.0, 0.0};
        moab::Tag source_div_exact_tag;
        moab::Tag source_flux_l1_exact_tag;
        moab::Tag source_field_exact_tag;
        moab::Tag target_div_recon_tag;
        moab::Tag target_flux_l1_recon_tag;
        moab::Tag target_div_error_tag;
        moab::Tag target_field_recon_tag;
        moab::Tag target_field_exact_tag;
        moab::Tag target_field_error_tag;
        moab::Tag target_field_error_norm_tag;

        mimetic::check_moab(mb.tag_get_handle("SOURCE_DIV_EXACT", 1, moab::MB_TYPE_DOUBLE, source_div_exact_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create SOURCE_DIV_EXACT tag");
        mimetic::check_moab(mb.tag_get_handle("SOURCE_FLUX_L1_EXACT", 1, moab::MB_TYPE_DOUBLE, source_flux_l1_exact_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create SOURCE_FLUX_L1_EXACT tag");
        mimetic::check_moab(mb.tag_get_handle("SOURCE_FIELD_EXACT", 3, moab::MB_TYPE_DOUBLE, source_field_exact_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_vector),
                            "Failed to create SOURCE_FIELD_EXACT tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_DIV_RECON", 1, moab::MB_TYPE_DOUBLE, target_div_recon_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create TARGET_DIV_RECON tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_FLUX_L1_RECON", 1, moab::MB_TYPE_DOUBLE, target_flux_l1_recon_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create TARGET_FLUX_L1_RECON tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_DIV_ERROR", 1, moab::MB_TYPE_DOUBLE, target_div_error_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create TARGET_DIV_ERROR tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_FIELD_RECON", 3, moab::MB_TYPE_DOUBLE, target_field_recon_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_vector),
                            "Failed to create TARGET_FIELD_RECON tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_FIELD_EXACT", 3, moab::MB_TYPE_DOUBLE, target_field_exact_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_vector),
                            "Failed to create TARGET_FIELD_EXACT tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_FIELD_ERROR", 3, moab::MB_TYPE_DOUBLE, target_field_error_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_vector),
                            "Failed to create TARGET_FIELD_ERROR tag");
        mimetic::check_moab(mb.tag_get_handle("TARGET_FIELD_ERROR_NORM", 1, moab::MB_TYPE_DOUBLE, target_field_error_norm_tag,
                                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_scalar),
                            "Failed to create TARGET_FIELD_ERROR_NORM tag");

        double total_source_divergence = 0.0;
        for (moab::EntityHandle cell : source_mesh) {
            mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, true);
            double cell_div = 0.0;
            double cell_flux_l1 = 0.0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
                std::size_t j = (i + 1) % poly.vertices.size();
                Eigen::Vector3d A = poly.points_3d[i];
                Eigen::Vector3d B = poly.points_3d[j];
                double flux = exact_edge_flux(A, B);
                cell_div += flux;
                cell_flux_l1 += std::abs(flux);

                moab::EntityHandle moab_edge = mimetic::find_or_create_edge(mb, poly.vertices[i], poly.vertices[j]);
                mimetic::check_moab(mb.tag_set_data(interpolator.source_flux_tag(), &moab_edge, 1, &flux), "Failed to write source flux");
            }
            total_source_divergence += cell_div;
            const mimetic::ReconstructionCoeffs source_coeffs = interpolator.reconstruct_source_polygon(cell);
            const Eigen::Vector3d source_field_exact = spherical_harmonic_gradient(poly.centroid_3d.normalized());
            double source_field_exact_data[3] = {source_field_exact.x(), source_field_exact.y(), source_field_exact.z()};
            mimetic::check_moab(mb.tag_set_data(source_div_exact_tag, &cell, 1, &cell_div), "Failed to write SOURCE_DIV_EXACT");
            mimetic::check_moab(mb.tag_set_data(source_flux_l1_exact_tag, &cell, 1, &cell_flux_l1), "Failed to write SOURCE_FLUX_L1_EXACT");
            mimetic::check_moab(mb.tag_set_data(source_field_exact_tag, &cell, 1, source_field_exact_data), "Failed to write SOURCE_FIELD_EXACT");
            (void)source_coeffs;
        }
        
        std::cout << "Total source discrete divergence: " << total_source_divergence << " (Expected ~0)\n";

        // Transfer to target mesh
        mimetic::EdgeTransferResult edge_transfer = interpolator.transfer_source_to_target_edges(source_mesh, target_mesh);
        
        double max_flux_error = 0.0;
        double total_target_divergence = 0.0;
        double edge_l1_error = 0.0;
        double edge_l2_error_squared = 0.0;
        double edge_linf_error = 0.0;
        double cell_l1_error = 0.0;
        double cell_linf_error = 0.0;
        std::size_t dof = 0;
        
        for (moab::EntityHandle cell : target_mesh) {
            mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, true);
            double cell_div_exact = 0.0;
            double cell_div_recon = 0.0;
            double cell_flux_l1_recon = 0.0;

            for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
                std::size_t j = (i + 1) % poly.vertices.size();
                Eigen::Vector3d A = poly.points_3d[i];
                Eigen::Vector3d B = poly.points_3d[j];
                double exact = exact_edge_flux(A, B);
                double transferred = edge_transfer.target_fluxes[dof++];

                double err = std::abs(exact - transferred);
                if (err > max_flux_error) max_flux_error = err;

                cell_div_exact += exact;
                cell_div_recon += transferred;
                cell_flux_l1_recon += std::abs(transferred);

                moab::EntityHandle moab_edge = mimetic::find_or_create_edge(mb, poly.vertices[i], poly.vertices[j]);
                mimetic::check_moab(mb.tag_set_data(interpolator.source_flux_tag(), &moab_edge, 1, &transferred),
                                    "Failed to write transferred target flux as reconstruction source flux");
            }

            total_target_divergence += cell_div_recon;
            const mimetic::ReconstructionCoeffs target_coeffs = interpolator.reconstruct_source_polygon(cell);
            const Eigen::Vector3d field_recon = reconstructed_vector_at_centroid(interpolator, target_coeffs, poly);
            const Eigen::Vector3d field_exact = spherical_harmonic_gradient(poly.centroid_3d.normalized());
            const Eigen::Vector3d field_error = field_recon - field_exact;
            const double field_error_norm = field_error.norm();
            const double div_error = cell_div_recon - cell_div_exact;
            cell_l1_error += poly.area * field_error_norm;
            cell_linf_error = std::max(cell_linf_error, field_error_norm);
            double field_recon_data[3] = {field_recon.x(), field_recon.y(), field_recon.z()};
            double field_exact_data[3] = {field_exact.x(), field_exact.y(), field_exact.z()};
            double field_error_data[3] = {field_error.x(), field_error.y(), field_error.z()};
            mimetic::check_moab(mb.tag_set_data(target_div_recon_tag, &cell, 1, &cell_div_recon), "Failed to write TARGET_DIV_RECON");
            mimetic::check_moab(mb.tag_set_data(target_flux_l1_recon_tag, &cell, 1, &cell_flux_l1_recon), "Failed to write TARGET_FLUX_L1_RECON");
            mimetic::check_moab(mb.tag_set_data(target_div_error_tag, &cell, 1, &div_error), "Failed to write TARGET_DIV_ERROR");
            mimetic::check_moab(mb.tag_set_data(target_field_recon_tag, &cell, 1, field_recon_data), "Failed to write TARGET_FIELD_RECON");
            mimetic::check_moab(mb.tag_set_data(target_field_exact_tag, &cell, 1, field_exact_data), "Failed to write TARGET_FIELD_EXACT");
            mimetic::check_moab(mb.tag_set_data(target_field_error_tag, &cell, 1, field_error_data), "Failed to write TARGET_FIELD_ERROR");
            mimetic::check_moab(mb.tag_set_data(target_field_error_norm_tag, &cell, 1, &field_error_norm), "Failed to write TARGET_FIELD_ERROR_NORM");
            for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
                std::size_t j = (i + 1) % poly.vertices.size();
                edge_error_metrics(interpolator, target_coeffs, poly, poly.points_3d[i], poly.points_3d[j],
                                   edge_l1_error, edge_l2_error_squared, edge_linf_error);
            }
        }
        
        std::cout << "Total target discrete divergence: " << total_target_divergence << " (Expected ~0)\n";
        std::cout << "Max target edge flux error vs analytical: " << max_flux_error << "\n";
        std::cout << "Target edge flux-density error norms: L1=" << edge_l1_error
                  << ", L2=" << std::sqrt(edge_l2_error_squared)
                  << ", Linf=" << edge_linf_error << "\n";
        std::cout << "Target cell-centered vector error norms: L1=" << cell_l1_error
                  << ", Linf=" << cell_linf_error << "\n";
        
        // Strict conservation check: target divergence should match source divergence perfectly
        if (std::isnan(total_target_divergence) || std::abs(total_source_divergence - total_target_divergence) > 1e-12) {
            std::cerr << "[ERROR] Global conservation violated! Diff: " << std::abs(total_source_divergence - total_target_divergence) << "\n";
            return 1;
        }

        const std::string source_path = output_prefix + "_source.vtk";
        const std::string target_path = output_prefix + "_target.vtk";
        moab::EntityHandle src_set, tgt_set;
        mb.create_meshset(moab::MESHSET_SET, src_set);
        mb.add_entities(src_set, &source_mesh[0], source_mesh.size());
        mimetic::check_moab(mb.write_file(source_path.c_str(), nullptr, nullptr, &src_set, 1),
                            "Failed to write source VTK output");
        
        mb.create_meshset(moab::MESHSET_SET, tgt_set);
        mb.add_entities(tgt_set, &target_mesh[0], target_mesh.size());
        mimetic::check_moab(mb.write_file(target_path.c_str(), nullptr, nullptr, &tgt_set, 1),
                            "Failed to write target VTK output");
        std::cout << "Wrote source visualization to " << source_path << "\n";
        std::cout << "Wrote target visualization to " << target_path << "\n";
        
        std::cout << "\n[SUCCESS] Spherical manifold interpolation test completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
