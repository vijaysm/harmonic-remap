#ifndef MIMETIC_SPHERICAL_TRANSFER_TEST_UTILS_HPP
#define MIMETIC_SPHERICAL_TRANSFER_TEST_UTILS_HPP

#include "mimetic/mimetic.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimetic {
namespace test_sphere {

struct GaussPoint {
    double x;
    double w;
};

static const GaussPoint gauss16_table[16] = {
    {-0.9894009349916499, 0.0271524594117541},
    {-0.9445750230732326, 0.0622535239386479},
    {-0.8656312023878318, 0.0951585116824928},
    {-0.7554044083550030, 0.1246289712555339},
    {-0.6178762444026438, 0.1495959888165767},
    {-0.4580167776572274, 0.1691565193950025},
    {-0.2816035507792589, 0.1826034150449236},
    {-0.0950125098376374, 0.1894506104550685},
    { 0.0950125098376374, 0.1894506104550685},
    { 0.2816035507792589, 0.1826034150449236},
    { 0.4580167776572274, 0.1691565193950025},
    { 0.6178762444026438, 0.1495959888165767},
    { 0.7554044083550030, 0.1246289712555339},
    { 0.8656312023878318, 0.0951585116824928},
    { 0.9445750230732326, 0.0622535239386479},
    { 0.9894009349916499, 0.0271524594117541},
};

template <typename Func>
double integrate_edge_gauss16(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& func)
{
    const double length = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    double sum = 0.0;
    for (const GaussPoint& q : gauss16_table) {
        sum += q.w * func(mid + q.x * half_delta);
    }
    return 0.5 * length * sum;
}

inline void project_to_sphere(const double x, const double y, const double z, double out[3])
{
    const double mag = std::sqrt(x * x + y * y + z * z);
    out[0] = x / mag;
    out[1] = y / mag;
    out[2] = z / mag;
}

inline moab::EntityHandle create_spherical_polygon(moab::Core& mb, const std::vector<Eigen::Vector3d>& points)
{
    if (points.size() < 3) {
        throw std::runtime_error("Cannot create spherical polygon with fewer than three points");
    }

    std::vector<moab::EntityHandle> vertices(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d p = points[i].normalized();
        const double xyz[3] = {p.x(), p.y(), p.z()};
        check_moab(mb.create_vertex(xyz, vertices[i]), "Failed to create spherical vertex");
    }

    moab::EntityHandle polygon = 0;
    const moab::EntityType type = (vertices.size() == 4) ? moab::MBQUAD : moab::MBPOLYGON;
    check_moab(mb.create_element(type, vertices.data(), static_cast<int>(vertices.size()), polygon),
               "Failed to create spherical polygon");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        find_or_create_edge(mb, vertices[i], vertices[(i + 1) % vertices.size()]);
    }
    return polygon;
}

inline moab::EntityHandle create_chart_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& chart_points)
{
    const GnomonicFrame frame{
        Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        1.0,
    };

    std::vector<Eigen::Vector3d> points;
    points.reserve(chart_points.size());
    for (const Eigen::Vector2d& point : chart_points) {
        points.push_back(inverse_gnomonic(point, frame));
    }
    return create_spherical_polygon(mb, points);
}

inline std::vector<moab::EntityHandle> generate_cubed_sphere(moab::Core& mb, const int n)
{
    std::vector<moab::EntityHandle> elements;
    std::map<std::array<int, 3>, moab::EntityHandle> vertex_map;

    auto get_or_create_vertex = [&](const int ix, const int iy, const int iz) -> moab::EntityHandle {
        const std::array<int, 3> key = {{ix, iy, iz}};
        const auto existing = vertex_map.find(key);
        if (existing != vertex_map.end()) {
            return existing->second;
        }

        const double x = static_cast<double>(ix) / n;
        const double y = static_cast<double>(iy) / n;
        const double z = static_cast<double>(iz) / n;
        double p[3] = {0.0, 0.0, 0.0};
        project_to_sphere(x, y, z, p);
        moab::EntityHandle vertex = 0;
        check_moab(mb.create_vertex(p, vertex), "Failed to create cubed-sphere vertex");
        vertex_map[key] = vertex;
        return vertex;
    };

    auto build_face = [&](const int dim1, const int dim2, const int fixed_dim,
                          const int fixed_val, const bool flip) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const int u1 = -n + 2 * i;
                const int u2 = -n + 2 * (i + 1);
                const int v1 = -n + 2 * j;
                const int v2 = -n + 2 * (j + 1);

                int c[4][3] = {};
                for (int k = 0; k < 4; ++k) {
                    const int u = (k == 0 || k == 3) ? u1 : u2;
                    const int v = (k == 0 || k == 1) ? v1 : v2;
                    c[k][dim1] = u;
                    c[k][dim2] = v;
                    c[k][fixed_dim] = fixed_val;
                }

                moab::EntityHandle conn[4] = {};
                for (int k = 0; k < 4; ++k) {
                    conn[k] = get_or_create_vertex(c[k][0], c[k][1], c[k][2]);
                }
                if (flip) {
                    std::swap(conn[1], conn[3]);
                }

                moab::EntityHandle quad = 0;
                check_moab(mb.create_element(moab::MBQUAD, conn, 4, quad),
                           "Failed to create cubed-sphere cell");
                for (int k = 0; k < 4; ++k) {
                    find_or_create_edge(mb, conn[k], conn[(k + 1) % 4]);
                }
                elements.push_back(quad);
            }
        }
    };

    build_face(0, 1, 2,  n, false);
    build_face(0, 1, 2, -n, true);
    build_face(1, 2, 0,  n, false);
    build_face(1, 2, 0, -n, true);
    build_face(2, 0, 1,  n, false);
    build_face(2, 0, 1, -n, true);
    return elements;
}

inline Eigen::Vector3d spherical_harmonic_gradient(const Eigen::Vector3d& point)
{
    const Eigen::Vector3d p = point.normalized();
    const Eigen::Vector3d grad_3d(0.0, 0.0, 3.0 * p.z());
    return grad_3d - grad_3d.dot(p) * p;
}

template <typename Field>
double exact_chart_edge_flux(moab::Core& mb,
                             const moab::EntityHandle cell,
                             const std::size_t local_edge_index,
                             const Field& field)
{
    GeometryOptions spherical;
    spherical.mode = GeometryMode::SphericalGnomonic;
    const LocalPolygon poly = local_polygon(mb, cell, spherical);
    const std::vector<LocalEdge> edges = local_edges(mb, poly);
    if (local_edge_index >= edges.size()) {
        throw std::runtime_error("Spherical exact edge index out of range");
    }

    const LocalEdge& edge = edges[local_edge_index];
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
    return integrate_edge_gauss16(edge.a, edge.b, [&](const Eigen::Vector2d& p_local) {
        const Eigen::Vector2d xi = p_local + poly.centroid;
        const Eigen::Vector3d point = inverse_gnomonic(xi, frame);
        const Eigen::Vector2d chart_vector = pullback_contravariant_piola(field(point.normalized()), xi, frame);
        return chart_vector.dot(edge.outward_normal);
    });
}

template <typename Field>
void set_source_fluxes(MimeticInterpolator& interpolator,
                       moab::Core& mb,
                       const moab::EntityHandle cell,
                       const Field& field)
{
    GeometryOptions spherical;
    spherical.mode = GeometryMode::SphericalGnomonic;
    const LocalPolygon poly = local_polygon(mb, cell, spherical);
    for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
        interpolator.set_source_edge_flux(cell, i, exact_chart_edge_flux(mb, cell, i, field));
    }
}

inline std::array<long long, 3> rounded_point_key(const Eigen::Vector3d& point)
{
    const double scale = 1.0e12;
    const Eigen::Vector3d p = point.normalized();
    return std::array<long long, 3>{{
        static_cast<long long>(std::llround(scale * p.x())),
        static_cast<long long>(std::llround(scale * p.y())),
        static_cast<long long>(std::llround(scale * p.z())),
    }};
}

inline std::array<long long, 6> undirected_edge_key(const Eigen::Vector3d& a,
                                                    const Eigen::Vector3d& b)
{
    std::array<long long, 3> ak = rounded_point_key(a);
    std::array<long long, 3> bk = rounded_point_key(b);
    if (bk < ak) {
        std::swap(ak, bk);
    }
    return std::array<long long, 6>{{ak[0], ak[1], ak[2], bk[0], bk[1], bk[2]}};
}

struct ConservativeEdgeRecord {
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    double flux;
};

template <typename Field>
std::map<std::pair<moab::EntityHandle, std::size_t>, double>
conservative_edge_fluxes(moab::Core& mb,
                         const std::vector<moab::EntityHandle>& cells,
                         const Field& field)
{
    GeometryOptions spherical;
    spherical.mode = GeometryMode::SphericalGnomonic;
    std::map<std::array<long long, 6>, ConservativeEdgeRecord> edge_records;
    std::map<std::pair<moab::EntityHandle, std::size_t>, double> fluxes;

    for (const moab::EntityHandle cell : cells) {
        const LocalPolygon poly = local_polygon(mb, cell, spherical);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
            const std::size_t j = (i + 1) % poly.vertices.size();
            const Eigen::Vector3d a = poly.points_3d[i].normalized();
            const Eigen::Vector3d b = poly.points_3d[j].normalized();
            const std::array<long long, 6> key = undirected_edge_key(a, b);
            const auto existing = edge_records.find(key);

            double flux = 0.0;
            if (existing == edge_records.end()) {
                flux = exact_chart_edge_flux(mb, cell, i, field);
                edge_records[key] = ConservativeEdgeRecord{a, b, flux};
            } else {
                const ConservativeEdgeRecord& record = existing->second;
                const bool same_orientation =
                    (a - record.a).norm() < 1.0e-10 && (b - record.b).norm() < 1.0e-10;
                flux = same_orientation ? record.flux : -record.flux;
            }
            fluxes[std::make_pair(cell, i)] = flux;
        }
    }
    return fluxes;
}

template <typename Field>
void set_conservative_source_fluxes(MimeticInterpolator& interpolator,
                                    moab::Core& mb,
                                    const std::vector<moab::EntityHandle>& cells,
                                    const Field& field)
{
    const std::map<std::pair<moab::EntityHandle, std::size_t>, double> fluxes =
        conservative_edge_fluxes(mb, cells, field);
    for (const auto& item : fluxes) {
        interpolator.set_source_edge_flux(item.first.first, item.first.second, item.second);
    }
}

inline ReconstructionCoeffs read_coeffs(moab::Core& mb,
                                        const MimeticInterpolator& interpolator,
                                        const moab::EntityHandle cell)
{
    const void* ptr = nullptr;
    int size = 0;
    check_moab(mb.tag_get_by_ptr(interpolator.coeffs_tag(), &cell, 1, &ptr, &size),
               "Failed to read reconstruction coefficients");
    const double* data = static_cast<const double*>(ptr);
    ReconstructionCoeffs coeffs;
    coeffs.d = data[0];
    coeffs.harmonic.assign(data + 1, data + size);
    return coeffs;
}

inline Eigen::Vector3d reconstructed_surface_vector(const MimeticInterpolator& interpolator,
                                                    const ReconstructionCoeffs& coeffs,
                                                    const LocalPolygon& poly,
                                                    const Eigen::Vector2d& local_point)
{
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
    return lift_contravariant_piola(interpolator.velocity(coeffs, local_point),
                                    local_point + poly.centroid,
                                    frame);
}

template <typename Field>
Eigen::Vector3d exact_surface_cell_average(moab::Core& mb,
                                           const moab::EntityHandle cell,
                                           const Field& field)
{
    GeometryOptions spherical;
    spherical.mode = GeometryMode::SphericalGnomonic;
    const SphericalPolygon poly = spherical_polygon(mb, cell, spherical);
    const Eigen::Vector2d center = poly.projected_centroid;

    Eigen::Vector3d integral = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < poly.local_points.size(); ++i) {
        const Eigen::Vector2d a = poly.local_points[i];
        const Eigen::Vector2d b = poly.local_points[(i + 1) % poly.local_points.size()];
        integral.x() += integrate_triangle_scalar(Eigen::Vector2d::Zero(), a, b, [&](const Eigen::Vector2d& local_point) {
            const Eigen::Vector2d xi = local_point + center;
            return field(inverse_gnomonic(xi, poly.frame)).x() * gnomonic_area_scale(xi, poly.frame);
        });
        integral.y() += integrate_triangle_scalar(Eigen::Vector2d::Zero(), a, b, [&](const Eigen::Vector2d& local_point) {
            const Eigen::Vector2d xi = local_point + center;
            return field(inverse_gnomonic(xi, poly.frame)).y() * gnomonic_area_scale(xi, poly.frame);
        });
        integral.z() += integrate_triangle_scalar(Eigen::Vector2d::Zero(), a, b, [&](const Eigen::Vector2d& local_point) {
            const Eigen::Vector2d xi = local_point + center;
            return field(inverse_gnomonic(xi, poly.frame)).z() * gnomonic_area_scale(xi, poly.frame);
        });
    }
    return integral / poly.spherical_area;
}

inline double max_direct_sparse_delta(const MimeticInterpolator& interpolator,
                                      const SparseEdgeProjection& projection,
                                      const EdgeTransferResult& transfer)
{
    Eigen::VectorXd source_flux(projection.source_edges.size());
    for (std::size_t i = 0; i < projection.source_edges.size(); ++i) {
        const DirectedEdgeDof& dof = projection.source_edges[i];
        source_flux(static_cast<Eigen::Index>(i)) =
            interpolator.source_edge_flux(dof.polygon, dof.local_edge_index, dof.edge);
    }

    const Eigen::VectorXd projected = projection.matrix * source_flux;
    double max_delta = 0.0;
    for (Eigen::Index i = 0; i < projected.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(projected(i) - transfer.target_fluxes[static_cast<std::size_t>(i)]));
    }
    return max_delta;
}

/// Generate an icosahedral triangular mesh on the unit sphere with the given
/// refinement level.  Level 0 gives the base 20-triangle icosahedron (12 vertices).
/// Level n subdivides each triangle into n^2 sub-triangles, giving 20*n^2 faces
/// and 10*n^2+2 vertices.
inline std::vector<moab::EntityHandle> generate_icosahedral_triangles(
    moab::Core& mb, int refine_level)
{
    constexpr double kPi = 3.14159265358979323846;
    const double node_lat = std::atan(0.5);

    // 12 base icosahedral vertices in (lon, lat)
    struct LL { double lon, lat; };
    const LL base_ll[12] = {
        {0.0,               -0.5*kPi},                    // 0: south pole
        {0.0,               -node_lat},                    // 1-5: southern ring
        {2.0*kPi*0.2,       -node_lat},
        {2.0*kPi*0.4,       -node_lat},
        {2.0*kPi*0.6,       -node_lat},
        {2.0*kPi*0.8,       -node_lat},
        {2.0*kPi*0.1,       +node_lat},                   // 6-10: northern ring
        {2.0*kPi*0.3,       +node_lat},
        {2.0*kPi*0.5,       +node_lat},
        {2.0*kPi*0.7,       +node_lat},
        {2.0*kPi*0.9,       +node_lat},
        {0.0,               +0.5*kPi},                    // 11: north pole
    };

    // Convert to Cartesian on unit sphere
    std::vector<Eigen::Vector3d> nodes;
    nodes.reserve(12);
    for (int i = 0; i < 12; ++i) {
        nodes.push_back(Eigen::Vector3d(
            std::sin(base_ll[i].lon) * std::cos(base_ll[i].lat),
            std::cos(base_ll[i].lon) * std::cos(base_ll[i].lat),
            std::sin(base_ll[i].lat)));
    }

    // Insert a sub-node along the great-circle arc from nodes[i0] to nodes[i1]
    // at fraction alpha in [0,1], projected back to the sphere.
    auto insert_sub_node = [&](int i0, int i1, double alpha) -> int {
        const Eigen::Vector3d& a = nodes[i0];
        const Eigen::Vector3d& b = nodes[i1];
        const double theta = std::acos(std::max(-1.0, std::min(1.0, a.dot(b))));
        Eigen::Vector3d p;
        if (theta < 1e-14) {
            p = a;
        } else {
            p = (std::sin((1.0 - alpha) * theta) * a + std::sin(alpha * theta) * b) / std::sin(theta);
        }
        p.normalize();
        const int idx = static_cast<int>(nodes.size());
        nodes.push_back(p);
        return idx;
    };

    // Generate edge vertices: returns list of node indices along the edge
    using Edge = std::vector<int>;
    auto gen_edge = [&](int i0, int i1) -> Edge {
        Edge e;
        e.push_back(i0);
        for (int k = 1; k < refine_level; ++k) {
            e.push_back(insert_sub_node(i0, i1, static_cast<double>(k) / refine_level));
        }
        e.push_back(i1);
        return e;
    };
    auto flip_edge = [](const Edge& e) -> Edge {
        Edge f(e.rbegin(), e.rend());
        return f;
    };

    // 30 edges of the icosahedron, each subdivided
    std::vector<Edge> edges(30);
    for (int i = 0; i < 5; ++i) edges[i] = gen_edge(0, i + 1);
    for (int i = 0; i < 5; ++i) edges[i + 5] = gen_edge(i + 1, ((i + 1) % 5) + 1);
    edges[10] = gen_edge(1, 6);  edges[11] = gen_edge(6, 2);
    edges[12] = gen_edge(2, 7);  edges[13] = gen_edge(7, 3);
    edges[14] = gen_edge(3, 8);  edges[15] = gen_edge(8, 4);
    edges[16] = gen_edge(4, 9);  edges[17] = gen_edge(9, 5);
    edges[18] = gen_edge(5, 10); edges[19] = gen_edge(10, 1);
    for (int i = 0; i < 5; ++i) edges[i + 20] = gen_edge(i + 6, ((i + 1) % 5) + 6);
    for (int i = 0; i < 5; ++i) edges[i + 25] = gen_edge(i + 6, 11);

    // Generate sub-triangles from one icosahedral face defined by 3 edges
    struct Tri { int v[3]; };
    std::vector<Tri> triangles;
    auto gen_faces = [&](const Edge& e0, const Edge& e1, const Edge& e2) {
        Edge bot;
        bot.push_back(e0[0]);
        for (int j = 0; j < refine_level; ++j) {
            Edge top;
            if (j == refine_level - 1) {
                top = e2;
            } else {
                top.push_back(e0[j + 1]);
                for (int k = 1; k <= j; ++k) {
                    top.push_back(insert_sub_node(e0[j + 1], e1[j + 1],
                                                  static_cast<double>(k) / (j + 1)));
                }
                top.push_back(e1[j + 1]);
            }
            for (int i = 0; i < 2 * j + 1; ++i) {
                if (i % 2 == 0) {
                    triangles.push_back(Tri{{bot[i / 2], top[i / 2], top[i / 2 + 1]}});
                } else {
                    triangles.push_back(Tri{{top[(i + 1) / 2], bot[(i + 1) / 2], bot[(i - 1) / 2]}});
                }
            }
            bot = top;
        }
    };

    // 20 icosahedral faces
    for (int i = 0; i < 5; ++i)
        gen_faces(edges[i], edges[(i + 1) % 5], edges[i + 5]);
    for (int i = 0; i < 5; ++i)
        gen_faces(edges[2 * i + 10], edges[i + 5], edges[2 * i + 11]);
    for (int i = 0; i < 5; ++i)
        gen_faces(edges[i + 20], edges[2 * i + 11], flip_edge(edges[2 * ((i + 1) % 5) + 10]));
    for (int i = 0; i < 5; ++i)
        gen_faces(edges[i + 25], edges[i + 20], flip_edge(edges[((i + 1) % 5) + 25]));

    // Create MOAB vertices
    std::vector<moab::EntityHandle> moab_verts(nodes.size(), 0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double xyz[3] = {nodes[i].x(), nodes[i].y(), nodes[i].z()};
        check_moab(mb.create_vertex(xyz, moab_verts[i]), "ico vertex");
    }

    // Create MOAB triangles
    std::vector<moab::EntityHandle> cells;
    cells.reserve(triangles.size());
    for (const Tri& t : triangles) {
        moab::EntityHandle conn[3] = {moab_verts[t.v[0]], moab_verts[t.v[1]], moab_verts[t.v[2]]};
        moab::EntityHandle tri = 0;
        check_moab(mb.create_element(moab::MBPOLYGON, conn, 3, tri), "ico tri");
        for (int k = 0; k < 3; ++k)
            find_or_create_edge(mb, conn[k], conn[(k + 1) % 3]);
        cells.push_back(tri);
    }
    return cells;
}

/// Generate the dual (Voronoi) mesh of an icosahedral triangulation.
/// Each original vertex becomes a Voronoi cell whose vertices are the
/// circumcenters (normalized centroids) of the surrounding triangles.
inline std::vector<moab::EntityHandle> generate_icosahedral_dual(
    moab::Core& mb, int refine_level)
{
    // First generate the triangular mesh to get vertices and faces
    moab::Core tri_mb;
    const std::vector<moab::EntityHandle> tri_cells =
        generate_icosahedral_triangles(tri_mb, refine_level);

    // Collect all vertices and triangle connectivity
    struct TriConn { int v[3]; };
    std::vector<Eigen::Vector3d> tri_nodes;
    std::vector<TriConn> tri_faces;

    // Build node list
    std::map<moab::EntityHandle, int> vert_to_idx;
    for (const moab::EntityHandle tri : tri_cells) {
        const moab::EntityHandle* conn = nullptr;
        int nv = 0;
        check_moab(tri_mb.get_connectivity(tri, conn, nv), "get tri conn");
        for (int i = 0; i < nv; ++i) {
            if (vert_to_idx.find(conn[i]) == vert_to_idx.end()) {
                double xyz[3];
                check_moab(tri_mb.get_coords(&conn[i], 1, xyz), "get tri coords");
                vert_to_idx[conn[i]] = static_cast<int>(tri_nodes.size());
                tri_nodes.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]).normalized());
            }
        }
    }
    for (const moab::EntityHandle tri : tri_cells) {
        const moab::EntityHandle* conn = nullptr;
        int nv = 0;
        check_moab(tri_mb.get_connectivity(tri, conn, nv), "get tri conn");
        tri_faces.push_back(TriConn{{vert_to_idx[conn[0]], vert_to_idx[conn[1]], vert_to_idx[conn[2]]}});
    }

    // Compute circumcenter (normalized centroid) for each triangle
    std::vector<Eigen::Vector3d> circumcenters(tri_faces.size());
    for (std::size_t i = 0; i < tri_faces.size(); ++i) {
        const Eigen::Vector3d c = (tri_nodes[tri_faces[i].v[0]] +
                                   tri_nodes[tri_faces[i].v[1]] +
                                   tri_nodes[tri_faces[i].v[2]]).normalized();
        circumcenters[i] = c;
    }

    // For each original vertex, collect surrounding triangle indices (in order)
    // vertex_to_tris[v] = list of triangle indices sharing vertex v
    std::vector<std::vector<int>> vertex_to_tris(tri_nodes.size());
    for (std::size_t fi = 0; fi < tri_faces.size(); ++fi) {
        for (int k = 0; k < 3; ++k) {
            vertex_to_tris[tri_faces[fi].v[k]].push_back(static_cast<int>(fi));
        }
    }

    // Order the triangles around each vertex by adjacency
    auto order_ring = [&](int vi) -> std::vector<int> {
        std::vector<int> ring;
        const std::vector<int>& tris = vertex_to_tris[vi];
        if (tris.empty()) return ring;

        std::vector<bool> used(tris.size(), false);
        ring.push_back(tris[0]);
        used[0] = true;

        for (std::size_t step = 1; step < tris.size(); ++step) {
            const int last_tri = ring.back();
            bool found = false;
            for (std::size_t j = 0; j < tris.size(); ++j) {
                if (used[j]) continue;
                // Check if tris[j] shares an edge with last_tri that includes vi
                int shared = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        if (tri_faces[last_tri].v[a] == tri_faces[tris[j]].v[b])
                            ++shared;
                    }
                }
                if (shared >= 2) {  // share vertex vi plus one other vertex = adjacent
                    ring.push_back(tris[j]);
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if (!found) break;  // incomplete ring (shouldn't happen on closed manifold)
        }
        return ring;
    };

    // Create Voronoi cells in the target MOAB instance
    std::vector<moab::EntityHandle> dual_cells;
    dual_cells.reserve(tri_nodes.size());

    for (std::size_t vi = 0; vi < tri_nodes.size(); ++vi) {
        const std::vector<int> ring = order_ring(static_cast<int>(vi));
        if (ring.size() < 3) continue;

        // Voronoi cell vertices are the circumcenters of the ring triangles
        std::vector<Eigen::Vector3d> cell_pts;
        cell_pts.reserve(ring.size());
        for (const int fi : ring) {
            cell_pts.push_back(circumcenters[fi]);
        }

        // Ensure counter-clockwise orientation (outward normal = vertex direction)
        const Eigen::Vector3d center = tri_nodes[vi];
        Eigen::Vector3d cross_sum = Eigen::Vector3d::Zero();
        for (std::size_t k = 0; k < cell_pts.size(); ++k) {
            cross_sum += cell_pts[k].cross(cell_pts[(k + 1) % cell_pts.size()]);
        }
        if (cross_sum.dot(center) < 0) {
            std::reverse(cell_pts.begin(), cell_pts.end());
        }

        dual_cells.push_back(create_spherical_polygon(mb, cell_pts));
    }

    return dual_cells;
}

}  // namespace test_sphere
}  // namespace mimetic

#endif
