#include "common.h"

class ReLU {
    public:
        ReLU() {}
        void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    
    private:
        std::string profile_name = "ReLU";    
};