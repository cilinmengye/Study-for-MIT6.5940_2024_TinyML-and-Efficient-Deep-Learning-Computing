#include <cmath>

#include "operators.h"
#include "utils.h"

void ReLU::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    PROFILE_START(profile_name);

    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);

    for (int i = 0; i < x.m_dim_x; i++) {
        for (int j = 0; j < x.m_dim_y; j++) {
            for (int k = 0; k < x.m_dim_z; k++) {
                output(i, j, k) = std::fmax(static_cast<float>(0), x(i, j, k));
            }
        }
    }

    PROFILE_END(profile_name);
}