#include "common.h"
#include "operators.h"

class MockModel {
    public:
        MockModel(std::string param_path, const int hidden_size, const int extend_ratio);

        Matrix3D<float> forward(const Matrix3D<float> &x);

    private:
        // Weight 相关参数
        int hidden_size, extend_ratio;
        
        // Weight data point
        float* up_proj_weight;
        float* down_proj_weight;
        
        // layer
        Linear_FP up_proj;
        ReLU relu;
        Linear_FP down_proj;

        std::string profile_name = "MockModel";
};