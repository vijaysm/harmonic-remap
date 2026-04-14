#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <array>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    try {
        using Eigen::Vector2d;

        std::cout << "--- Mimetic Polygon Patch Test (Level 2) ---\n\n";

        moab::Core mb;
        mimetic::MimeticInterpolator interpolator(mb);

        const moab::EntityHandle source_quad = mimetic::create_quad(mb, {{
                                                                            Vector2d(0.0, 0.0),
                                                                            Vector2d(1.0, 0.0),
                                                                            Vector2d(1.0, 1.0),
                                                                            Vector2d(0.0, 1.0),
                                                                        }});
        mimetic::test::set_source_fluxes_from_field(
            mb, interpolator.source_flux_tag(), source_quad, mimetic::test::constant_field);

        const mimetic::ReconstructionCoeffs coeffs = interpolator.reconstruct_source_polygon(source_quad);

        std::cout << "Reconstructed coefficients:\n";
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "  c  = " << coeffs.c << "\n";
        std::cout << "  d  = " << coeffs.d << "\n";
        std::cout << "  a1 = " << coeffs.a1 << "\n";
        std::cout << "  b1 = " << coeffs.b1 << "\n";
        std::cout << "  a2 = " << coeffs.a2 << "\n";
        std::cout << "  b2 = " << coeffs.b2 << "\n\n";

        bool ok = true;
        ok = mimetic::test::near(coeffs.d, 0.0, mimetic::kTolerance, "zero divergence") && ok;
        ok = mimetic::test::near(coeffs.a1, 1.0, mimetic::kTolerance, "constant x velocity coeff") && ok;
        ok = mimetic::test::near(coeffs.b1, 1.0, mimetic::kTolerance, "constant y velocity coeff") && ok;
        ok = mimetic::test::near(coeffs.a2, 0.0, mimetic::kTolerance, "zero P2 coefficient") && ok;
        ok = mimetic::test::near(coeffs.b2, 0.0, mimetic::kTolerance, "zero Q2 coefficient") && ok;

        const Vector2d segment_a(-0.2, -0.3);
        const Vector2d segment_b(0.4, 0.2);
        const double exact_line = mimetic::test::constant_field(segment_a).dot(segment_b - segment_a);
        const double reconstructed_line = interpolator.line_integral(source_quad, segment_a, segment_b);
        std::cout << "\nInterior target line integral:\n";
        ok = mimetic::test::near(reconstructed_line, exact_line, mimetic::kTolerance, "line integral") && ok;

        const moab::EntityHandle target_quad = mimetic::create_quad(mb, {{
                                                                            Vector2d(0.25, 0.20),
                                                                            Vector2d(0.80, 0.20),
                                                                            Vector2d(0.80, 0.75),
                                                                            Vector2d(0.25, 0.75),
                                                                        }});
        const std::vector<double> target_fluxes = interpolator.transfer_to_target_polygon_edges(source_quad, target_quad);

        std::cout << "\nNon-matching target edge fluxes:\n";
        const std::array<double, 4> exact_target_fluxes = {{-0.55, 0.55, 0.55, -0.55}};
        double target_flux_sum = 0.0;
        for (std::size_t i = 0; i < target_fluxes.size(); ++i) {
            target_flux_sum += target_fluxes[i];
            ok = mimetic::test::near(target_fluxes[i], exact_target_fluxes[i], mimetic::kTolerance,
                                     "target edge " + std::to_string(i) + " flux") &&
                 ok;
        }
        ok = mimetic::test::near(target_flux_sum, 0.0, mimetic::kTolerance, "closed target conservation") && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Conservative patch test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Recovered the constant conservative interpolant and target edge fluxes.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
