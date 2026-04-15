#include "mimetic/mimetic.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mimetic {

void check_moab(const moab::ErrorCode code, const std::string& message)
{
    if (code != moab::MB_SUCCESS) {
        throw std::runtime_error(message + " (MOAB error " + std::to_string(static_cast<int>(code)) + ")");
    }
}

#include <complex>

void eval_harmonic_basis(int k, const Eigen::Vector2d& p,
                         double& P, double& Q,
                         Eigen::Vector2d& gradP, Eigen::Vector2d& gradQ)
{
    std::complex<double> z(p.x(), p.y());
    std::complex<double> zk = std::pow(z, k);
    P = zk.real();
    Q = zk.imag();
    std::complex<double> zk1 = (k == 1) ? std::complex<double>(1.0, 0.0) : std::pow(z, k - 1);
    gradP = Eigen::Vector2d(k * zk1.real(), -k * zk1.imag());
    gradQ = Eigen::Vector2d(k * zk1.imag(),  k * zk1.real());
}

namespace {

std::vector<Eigen::Vector2d> absolute_points(const LocalPolygon& polygon)
{
    std::vector<Eigen::Vector2d> points;
    points.reserve(polygon.points.size());
    for (const Eigen::Vector2d& p : polygon.points) {
        points.push_back(p + polygon.centroid);
    }
    return points;
}

Eigen::MatrixXd source_reconstruction_matrix(const LocalPolygon& poly, const std::vector<LocalEdge>& edges)
{
    const int N = static_cast<int>(edges.size());
    const int K_max = N / 2;
    const int N_h = 2 * K_max;
    const int S = N_h + N - 1;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N_h, N);
    const Eigen::Vector2d origin(0.0, 0.0);

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    return gi.dot(gj);
                });
            }
            V(i, j) = val;
        }
        
        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            });
            div_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return p.dot(is_Q_i ? gQ : gP);
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            V(i, i) += 1.0e6 * poly.area;
        }
        
        for (int e = 0; e < N; ++e) {
            const double edge_average = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            }) / edges[e].length;
            M(i, e) = edge_average - cell_basis_average - 0.5 * div_integral / poly.area;
        }
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N - 1, N_h);
    for (int e = 0; e < N - 1; ++e) {
        for (int i = 0; i < N_h; ++i) {
            int ki = (i / 2) + 1;
            bool is_Q_i = (i % 2 == 1);
            C(e, i) = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q;
                Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                Eigen::Vector2d g = is_Q_i ? gQ : gP;
                return g.dot(edges[e].outward_normal);
            });
        }
    }

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(N - 1, N);
    for (int e = 0; e < N - 1; ++e) {
        double Ee = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
            return p.dot(edges[e].outward_normal);
        });
        for (int j = 0; j < N; ++j) {
            F(e, j) = (e == j ? 1.0 : 0.0) - Ee / (2.0 * poly.area);
        }
    }

    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(S, S);
    KKT.block(0, 0, N_h, N_h) = V;
    KKT.block(N_h, 0, N - 1, N_h) = C;
    KKT.block(0, N_h, N_h, N - 1) = C.transpose();

    Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(S, N);
    RHS.block(0, 0, N_h, N) = M;
    RHS.block(N_h, 0, N - 1, N) = F;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
    Eigen::MatrixXd X = lu.solve(RHS).block(0, 0, N_h, N);

    Eigen::MatrixXd reconstruction = Eigen::MatrixXd::Zero(1 + N_h, N);
    reconstruction.row(0).setConstant(1.0 / poly.area);
    reconstruction.block(1, 0, N_h, N) = X;

    return reconstruction;
}

void write_edge_map_csv(const std::string& path, const std::vector<DirectedEdgeDof>& edges)
{
    std::ofstream out(path.c_str());
    if (!out) {
        throw std::runtime_error("Failed to open edge map for writing: " + path);
    }

    out << "index,polygon_handle,edge_handle,local_edge_index\n";
    for (std::size_t i = 0; i < edges.size(); ++i) {
        out << i << "," << edges[i].polygon << "," << edges[i].edge << "," << edges[i].local_edge_index << "\n";
    }
}

}  // namespace

// Shoelace geometry utilities; these correspond to the polygon preprocessing
// stage in report Algorithm 1, steps 1--2.
double signed_area(const std::vector<Eigen::Vector2d>& points)
{
    double area2 = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d& a = points[i];
        const Eigen::Vector2d& b = points[(i + 1) % points.size()];
        area2 += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * area2;
}

Eigen::Vector2d polygon_centroid(const std::vector<Eigen::Vector2d>& points)
{
    double area2 = 0.0;
    Eigen::Vector2d weighted(0.0, 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d& a = points[i];
        const Eigen::Vector2d& b = points[(i + 1) % points.size()];
        const double cross = a.x() * b.y() - b.x() * a.y();
        area2 += cross;
        weighted += (a + b) * cross;
    }
    if (std::abs(area2) < kTolerance) {
        throw std::runtime_error("Degenerate polygon area while computing centroid");
    }
    return weighted / (3.0 * area2);
}

moab::EntityHandle find_or_create_edge(moab::Core& mb, const moab::EntityHandle v0, const moab::EntityHandle v1)
{
    moab::EntityHandle verts[2] = {v0, v1};
    std::vector<moab::EntityHandle> edges;
    check_moab(mb.get_adjacencies(verts, 2, 1, false, edges, moab::Interface::INTERSECT),
               "Failed to query edge adjacency");
    if (!edges.empty()) {
        return edges.front();
    }

    moab::EntityHandle edge = 0;
    check_moab(mb.create_element(moab::MBEDGE, verts, 2, edge), "Failed to create edge");
    return edge;
}

// Extract MOAB connectivity into a local cell frame. The code enforces positive
// orientation because all later outward-normal signs assume counter-clockwise
// boundary order.
LocalPolygon local_polygon(moab::Core& mb, const moab::EntityHandle polygon)
{
    const moab::EntityHandle* conn = nullptr;
    int num_vertices = 0;
    check_moab(mb.get_connectivity(polygon, conn, num_vertices), "Failed to get polygon connectivity");
    if (num_vertices < 3) {
        throw std::runtime_error("Polygon must have at least three vertices");
    }

    std::vector<moab::EntityHandle> vertices(conn, conn + num_vertices);
    std::vector<Eigen::Vector2d> absolute_points;
    absolute_points.reserve(vertices.size());

    for (const moab::EntityHandle vertex : vertices) {
        double xyz[3] = {0.0, 0.0, 0.0};
        check_moab(mb.get_coords(&vertex, 1, xyz), "Failed to get vertex coordinates");
        absolute_points.emplace_back(xyz[0], xyz[1]);
    }

    if (signed_area(absolute_points) < 0.0) {
        std::reverse(vertices.begin(), vertices.end());
        std::reverse(absolute_points.begin(), absolute_points.end());
    }

    const Eigen::Vector2d centroid = polygon_centroid(absolute_points);
    std::vector<Eigen::Vector2d> relative_points;
    relative_points.reserve(absolute_points.size());
    for (const Eigen::Vector2d& p : absolute_points) {
        relative_points.push_back(p - centroid);
    }

    return LocalPolygon{vertices, relative_points, centroid, std::abs(signed_area(absolute_points))};
}

std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon)
{
    std::vector<LocalEdge> edges;
    edges.reserve(polygon.points.size());
    for (std::size_t i = 0; i < polygon.points.size(); ++i) {
        const std::size_t j = (i + 1) % polygon.points.size();
        const Eigen::Vector2d a = polygon.points[i];
        const Eigen::Vector2d b = polygon.points[j];
        const Eigen::Vector2d delta = b - a;
        const double length = delta.norm();
        if (length < kTolerance) {
            throw std::runtime_error("Degenerate edge length");
        }
        const Eigen::Vector2d outward(delta.y(), -delta.x());
        edges.push_back(LocalEdge{
            find_or_create_edge(mb, polygon.vertices[i], polygon.vertices[j]),
            a,
            b,
            outward / length,
            length,
        });
    }
    return edges;
}

moab::EntityHandle create_quad(moab::Core& mb, const std::array<Eigen::Vector2d, 4>& points)
{
    return create_polygon(mb, std::vector<Eigen::Vector2d>(points.begin(), points.end()));
}

moab::EntityHandle create_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& points)
{
    if (points.size() < 3) {
        throw std::runtime_error("Cannot create a polygon with fewer than three points");
    }

    std::vector<Eigen::Vector2d> ordered_points = points;
    if (signed_area(ordered_points) < 0.0) {
        std::reverse(ordered_points.begin(), ordered_points.end());
    }

    std::vector<moab::EntityHandle> vertices(ordered_points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double xyz[3] = {ordered_points[i].x(), ordered_points[i].y(), 0.0};
        check_moab(mb.create_vertex(xyz, vertices[i]), "Failed to create vertex");
    }

    moab::EntityHandle polygon = 0;
    const moab::EntityType type = (vertices.size() == 4) ? moab::MBQUAD : moab::MBPOLYGON;
    check_moab(mb.create_element(type, vertices.data(), static_cast<int>(vertices.size()), polygon),
               "Failed to create polygon");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        find_or_create_edge(mb, vertices[i], vertices[(i + 1) % vertices.size()]);
    }
    return polygon;
}

bool clip_segment_to_convex_polygon(const Eigen::Vector2d& segment_a,
                                    const Eigen::Vector2d& segment_b,
                                    const std::vector<Eigen::Vector2d>& polygon,
                                    Eigen::Vector2d& clipped_a,
                                    Eigen::Vector2d& clipped_b,
                                    const double tolerance)
{
    if (polygon.size() < 3) {
        return false;
    }

    std::vector<Eigen::Vector2d> clip = polygon;
    if (signed_area(clip) < 0.0) {
        std::reverse(clip.begin(), clip.end());
    }

    const Eigen::Vector2d delta = segment_b - segment_a;
    double t_enter = 0.0;
    double t_exit = 1.0;

    for (std::size_t i = 0; i < clip.size(); ++i) {
        const Eigen::Vector2d a = clip[i];
        const Eigen::Vector2d b = clip[(i + 1) % clip.size()];
        const Eigen::Vector2d edge = b - a;
        const Eigen::Vector2d inward_normal(-edge.y(), edge.x());
        const double offset = inward_normal.dot(a);
        const double signed_distance = inward_normal.dot(segment_a) - offset;
        const double rate = inward_normal.dot(delta);

        if (std::abs(rate) < tolerance) {
            if (signed_distance < -tolerance) {
                return false;
            }
            continue;
        }

        const double t = -signed_distance / rate;
        if (rate > 0.0) {
            t_enter = std::max(t_enter, t);
        } else {
            t_exit = std::min(t_exit, t);
        }

        if (t_enter > t_exit + tolerance) {
            return false;
        }
    }

    t_enter = std::max(0.0, t_enter);
    t_exit = std::min(1.0, t_exit);
    if (t_exit - t_enter <= tolerance) {
        return false;
    }

    clipped_a = segment_a + t_enter * delta;
    clipped_b = segment_a + t_exit * delta;
    return true;
}

void write_matrix_market(const SparseEdgeProjection& projection,
                         const std::string& matrix_path,
                         const std::string& source_edges_path,
                         const std::string& target_edges_path)
{
    std::ofstream out(matrix_path.c_str());
    if (!out) {
        throw std::runtime_error("Failed to open MatrixMarket file for writing: " + matrix_path);
    }

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << "% rows are directed target edges; columns are directed source edges\n";
    out << projection.matrix.rows() << " " << projection.matrix.cols() << " " << projection.matrix.nonZeros() << "\n";
    for (int row = 0; row < projection.matrix.outerSize(); ++row) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(projection.matrix, row); it; ++it) {
            out << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }

    write_edge_map_csv(source_edges_path, projection.source_edges);
    write_edge_map_csv(target_edges_path, projection.target_edges);
}

MimeticInterpolator::MimeticInterpolator(moab::Core& moab_instance) : mb_(moab_instance)
{
    const double default_flux = 0.0;
    check_moab(mb_.tag_get_handle("SOURCE_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_source_flux_,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
               "Failed to create SOURCE_FLUX tag");
    check_moab(mb_.tag_get_handle("TARGET_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_target_flux_,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
               "Failed to create TARGET_FLUX tag");

        check_moab(mb_.tag_get_handle("COEFFS", 0, moab::MB_TYPE_DOUBLE, tag_coeffs_,
                                  moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT),
               "Failed to create COEFFS tag");
}

moab::Tag MimeticInterpolator::source_flux_tag() const { return tag_source_flux_; }
moab::Tag MimeticInterpolator::target_flux_tag() const { return tag_target_flux_; }
moab::Tag MimeticInterpolator::coeffs_tag() const { return tag_coeffs_; }

ReconstructionCoeffs MimeticInterpolator::reconstruct_source_polygon(const moab::EntityHandle polygon)
{
    const LocalPolygon poly = local_polygon(mb_, polygon);
    const std::vector<LocalEdge> edges = local_edges(mb_, poly);

    const int N = static_cast<int>(edges.size());
    const int K_max = N / 2;
    const int N_h = 2 * K_max;
    const int S = N_h + N - 1;

    Eigen::VectorXd source_flux(N);
    for (int i = 0; i < N; ++i) {
        double flux = 0.0;
        check_moab(mb_.tag_get_data(tag_source_flux_, &edges[i].handle, 1, &flux), "Failed to read source flux tag");
        source_flux(i) = flux;
    }

    const double divergence = source_flux.sum() / poly.area;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::VectorXd M = Eigen::VectorXd::Zero(N_h);
    const Eigen::Vector2d origin(0.0, 0.0);

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    return gi.dot(gj);
                });
            }
            V(i, j) = val;
        }

        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            });
            div_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return p.dot(is_Q_i ? gQ : gP);
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            V(i, i) += 1.0e6 * poly.area;
        }
        
        for (int e = 0; e < N; ++e) {
            const double edge_average = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            }) / edges[e].length;
            M(i) += (edge_average - cell_basis_average - 0.5 * div_integral / poly.area) * source_flux(e);
        }
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N - 1, N_h);
    Eigen::VectorXd F_vec = Eigen::VectorXd::Zero(N - 1);
    for (int e = 0; e < N - 1; ++e) {
        for (int i = 0; i < N_h; ++i) {
            int ki = (i / 2) + 1;
            bool is_Q_i = (i % 2 == 1);
            C(e, i) = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q;
                Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                Eigen::Vector2d g = is_Q_i ? gQ : gP;
                return g.dot(edges[e].outward_normal);
            });
        }
        double Ee = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
            return p.dot(edges[e].outward_normal);
        });
        F_vec(e) = source_flux(e) - 0.5 * divergence * Ee;
    }

    // Solve KKT system: [V C^T; C 0] [X; lambda] = [M; F_vec]
    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(S, S);
    KKT.block(0, 0, N_h, N_h) = V;
    KKT.block(N_h, 0, N - 1, N_h) = C;
    KKT.block(0, N_h, N_h, N - 1) = C.transpose();

    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(S);
    RHS.segment(0, N_h) = M;
    RHS.segment(N_h, N - 1) = F_vec;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
    Eigen::VectorXd X = lu.solve(RHS).segment(0, N_h);
    
    ReconstructionCoeffs coeffs;
    coeffs.d = divergence;
    coeffs.harmonic.assign(X.data(), X.data() + N_h);

    std::vector<double> tag_data(1 + N_h);
    tag_data[0] = coeffs.d;
    for (int i = 0; i < N_h; ++i) tag_data[1 + i] = coeffs.harmonic[i];

    const void* ptr = tag_data.data();
    int size = tag_data.size();
    check_moab(mb_.tag_set_by_ptr(tag_coeffs_, &polygon, 1, &ptr, &size), "Failed to store reconstruction coefficients");

    return coeffs;
}

Eigen::Vector2d MimeticInterpolator::velocity(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& p) const
{
    Eigen::Vector2d v = 0.5 * coeffs.d * p;
    const int N_h = static_cast<int>(coeffs.harmonic.size());
    for (int i = 0; i < N_h; ++i) {
        int k = (i / 2) + 1;
        bool is_Q = (i % 2 == 1);
        double P, Q;
        Eigen::Vector2d gP, gQ;
        eval_harmonic_basis(k, p, P, Q, gP, gQ);
        v += coeffs.harmonic[i] * (is_Q ? gQ : gP);
    }
    return v;
}

double MimeticInterpolator::line_integral(const moab::EntityHandle source_polygon,
                                          const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b) const
{
    const void* ptr = nullptr;
    int size = 0;
    check_moab(mb_.tag_get_by_ptr(tag_coeffs_, &source_polygon, 1, &ptr, &size), "Failed to read reconstruction coefficients");
    const double* dptr = static_cast<const double*>(ptr);
    const double d = dptr[0];

    double val = 0.25 * d * (b.squaredNorm() - a.squaredNorm());
    for (int i = 0; i < size - 1; ++i) {
        int k = (i / 2) + 1;
        bool is_Q = (i % 2 == 1);
        double Pa, Qa, Pb, Qb;
        Eigen::Vector2d gPa, gQa, gPb, gQb;
        eval_harmonic_basis(k, a, Pa, Qa, gPa, gQa);
        eval_harmonic_basis(k, b, Pb, Qb, gPb, gQb);
        val += dptr[1 + i] * (is_Q ? (Qb - Qa) : (Pb - Pa));
    }
    return val;
}

double MimeticInterpolator::edge_flux(const ReconstructionCoeffs& coeffs,
                                      const Eigen::Vector2d& a,
                                      const Eigen::Vector2d& b) const
{
    // Boundary edges are treated as directed counter-clockwise edges of the
    // integration polygon. The outward normal is therefore the right normal.
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    const Eigen::Vector2d outward(delta.y(), -delta.x());
    const Eigen::Vector2d normal = outward / length;
    return integrate_edge_scalar(a, b, [&](const Eigen::Vector2d& p) { return velocity(coeffs, p).dot(normal); });
}

double MimeticInterpolator::polygon_boundary_flux(const ReconstructionCoeffs& coeffs,
                                                  const std::vector<Eigen::Vector2d>& points) const
{
    // Report Eq. (9): this is the discrete divergence-theorem check on one
    // source-target overlap polygon.
    double flux = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        flux += edge_flux(coeffs, points[i], points[(i + 1) % points.size()]);
    }
    return flux;
}

std::vector<double> MimeticInterpolator::transfer_to_target_polygon_edges(const moab::EntityHandle source_polygon,
                                                                          const moab::EntityHandle target_polygon)
{
    // Single-source-cell target-edge transfer used by the patch test. General
    // nonmatching meshes use clipped overlap polygons in the tests instead.
    const void* ptr = nullptr;
    int size = 0;
    check_moab(mb_.tag_get_by_ptr(tag_coeffs_, &source_polygon, 1, &ptr, &size),
               "Failed to read source reconstruction coefficients");
    const double* dptr = static_cast<const double*>(ptr);
    ReconstructionCoeffs coeffs;
    coeffs.d = dptr[0];
    coeffs.harmonic.assign(dptr + 1, dptr + size);

    const LocalPolygon source = local_polygon(mb_, source_polygon);
    const LocalPolygon target_absolute = local_polygon(mb_, target_polygon);

    std::vector<Eigen::Vector2d> target_points_in_source_frame;
    target_points_in_source_frame.reserve(target_absolute.points.size());
    for (const Eigen::Vector2d& target_relative : target_absolute.points) {
        target_points_in_source_frame.push_back(target_relative + target_absolute.centroid - source.centroid);
    }

    std::vector<double> target_fluxes;
    target_fluxes.reserve(target_points_in_source_frame.size());
    for (std::size_t i = 0; i < target_points_in_source_frame.size(); ++i) {
        const std::size_t j = (i + 1) % target_points_in_source_frame.size();
        const double flux = edge_flux(coeffs, target_points_in_source_frame[i], target_points_in_source_frame[j]);
        target_fluxes.push_back(flux);

        const moab::EntityHandle edge = find_or_create_edge(mb_, target_absolute.vertices[i], target_absolute.vertices[j]);
        check_moab(mb_.tag_set_data(tag_target_flux_, &edge, 1, &flux), "Failed to write target flux tag");
    }
    return target_fluxes;
}

EdgeTransferResult MimeticInterpolator::transfer_source_to_target_edges(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute_points;
        ReconstructionCoeffs coeffs;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const void* ptr = nullptr;
        int size = 0;
        check_moab(mb_.tag_get_by_ptr(tag_coeffs_, &source_polygon, 1, &ptr, &size),
                   "Failed to read source reconstruction coefficients");
        const double* dptr = static_cast<const double*>(ptr);
        ReconstructionCoeffs coeffs;
        coeffs.d = dptr[0];
        coeffs.harmonic.assign(dptr + 1, dptr + size);
        const LocalPolygon local = local_polygon(mb_, source_polygon);
        sources.push_back(SourceCache{source_polygon, local, absolute_points(local), coeffs});
    }

    EdgeTransferResult result;

    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon target = local_polygon(mb_, target_polygon);
        const std::vector<LocalEdge> target_edges = local_edges(mb_, target);

        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target_edges[edge_index];
            const Eigen::Vector2d target_a = target.centroid + target_edge.a;
            const Eigen::Vector2d target_b = target.centroid + target_edge.b;

            const std::size_t target_dof = result.target_edges.size();
            result.target_edges.push_back(DirectedEdgeDof{target_polygon, target_edge.handle, edge_index});
            result.target_fluxes.push_back(0.0);

            for (const SourceCache& source : sources) {
                Eigen::Vector2d clipped_a;
                Eigen::Vector2d clipped_b;
                if (!clip_segment_to_convex_polygon(target_a, target_b, source.absolute_points, clipped_a, clipped_b)) {
                    continue;
                }

                const Eigen::Vector2d local_a = clipped_a - source.local.centroid;
                const Eigen::Vector2d local_b = clipped_b - source.local.centroid;
                const double flux = edge_flux(source.coeffs, local_a, local_b);
                result.target_fluxes[target_dof] += flux;
                result.contributions.push_back(EdgeTransferContribution{
                    target_dof,
                    source.polygon,
                    clipped_a,
                    clipped_b,
                    flux,
                });
            }

            check_moab(mb_.tag_set_data(tag_target_flux_, &target_edge.handle, 1, &result.target_fluxes[target_dof]),
                       "Failed to write edge-wise target flux tag");
        }
    }

    return result;
}

SparseEdgeProjection MimeticInterpolator::assemble_edge_projection_operator(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<LocalEdge> edges;
        std::vector<Eigen::Vector2d> absolute_points;
        Eigen::MatrixXd reconstruction;
        std::vector<std::size_t> columns;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    SparseEdgeProjection projection;

    for (const moab::EntityHandle source_polygon : source_polygons) {
        const LocalPolygon local = local_polygon(mb_, source_polygon);
        const std::vector<LocalEdge> edges = local_edges(mb_, local);
        SourceCache cache{
            source_polygon,
            local,
            edges,
            absolute_points(local),
            source_reconstruction_matrix(local, edges),
            std::vector<std::size_t>(),
        };
        cache.columns.reserve(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            cache.columns.push_back(projection.source_edges.size());
            projection.source_edges.push_back(DirectedEdgeDof{source_polygon, edges[i].handle, i});
        }
        sources.push_back(cache);
    }

    std::vector<Eigen::Triplet<double>> triplets;
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon target = local_polygon(mb_, target_polygon);
        const std::vector<LocalEdge> target_edges = local_edges(mb_, target);

        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target_edges[edge_index];
            const Eigen::Vector2d target_a = target.centroid + target_edge.a;
            const Eigen::Vector2d target_b = target.centroid + target_edge.b;
            const std::size_t target_dof = projection.target_edges.size();
            projection.target_edges.push_back(DirectedEdgeDof{target_polygon, target_edge.handle, edge_index});

            for (const SourceCache& source : sources) {
                Eigen::Vector2d clipped_a;
                Eigen::Vector2d clipped_b;
                if (!clip_segment_to_convex_polygon(target_a, target_b, source.absolute_points, clipped_a, clipped_b)) {
                    continue;
                }

                const Eigen::Vector2d local_a = clipped_a - source.local.centroid;
                const Eigen::Vector2d local_b = clipped_b - source.local.centroid;
                const int N_h = source.reconstruction.rows() - 1;
                Eigen::MatrixXd evaluation = Eigen::MatrixXd::Zero(1, 1 + N_h);
                
                ReconstructionCoeffs c_div;
                c_div.d = 1.0;
                c_div.harmonic.resize(N_h, 0.0);
                evaluation(0, 0) = edge_flux(c_div, local_a, local_b);

                for (int i = 0; i < N_h; ++i) {
                    ReconstructionCoeffs c_harm;
                    c_harm.d = 0.0;
                    c_harm.harmonic.resize(N_h, 0.0);
                    c_harm.harmonic[i] = 1.0;
                    evaluation(0, 1 + i) = edge_flux(c_harm, local_a, local_b);
                }

                const Eigen::RowVectorXd weights = evaluation * source.reconstruction;
                for (Eigen::Index j = 0; j < weights.cols(); ++j) {
                    const double weight = weights(j);
                    if (std::abs(weight) > 1.0e-14) {
                        triplets.push_back(Eigen::Triplet<double>(
                            static_cast<int>(target_dof), static_cast<int>(source.columns[static_cast<std::size_t>(j)]), weight));
                    }
                }
            }
        }
    }

    projection.matrix.resize(static_cast<int>(projection.target_edges.size()), static_cast<int>(projection.source_edges.size()));
    projection.matrix.setFromTriplets(triplets.begin(), triplets.end());
    projection.matrix.makeCompressed();
    return projection;
}

}  // namespace mimetic
