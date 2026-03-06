#include "MockModel.h"

MockModel::MockModel(std::string param_path, const int hidden_size, const int extend_ratio) {
    this->hidden_size = hidden_size;
    this->extend_ratio = extend_ratio;
    allocate_aligned_memory(this->up_proj_weight, hidden_size * hidden_size * extend_ratio * sizeof(float));
    allocate_aligned_memory(this->down_proj_weight, hidden_size * hidden_size * extend_ratio * sizeof(float));

    this->up_proj = Linear_FP(Matrix3D<float>(this->up_proj_weight, 1, hidden_size * extend_ratio, hidden_size), param_path + "/0/weight.bin");
    this->relu = ReLU();
    this->down_proj = Linear_FP(Matrix3D<float>(this->down_proj_weight, 1, hidden_size, hidden_size * extend_ratio), param_path + "/2/weight.bin");
}

// 这里实现我认为还是有缺陷的，我始终没有释放 Activation 的空间，一般而言在 PyTorch 的 LLM 推理（Inference） 场景下，
// 处理中间 Activation（激活值）的核心逻辑为：即用即丢。
// 但是在整个库下都没有调用 deallocate_memory 删除过 Activation
Matrix3D<float> MockModel::forward(const Matrix3D<float> &x){
    PROFILE_START(profile_name);

    float *up_output_ptr;
    float *relu_output_ptr;
    float *down_output_ptr;
    const int batch_size = x.m_dim_y;
    
    // up_proj
    allocate_aligned_memory(up_output_ptr, batch_size * this->hidden_size * this->extend_ratio * sizeof(float));
    Matrix3D<float> up_output = Matrix3D<float>(up_output_ptr, 1, batch_size, this->hidden_size * this->extend_ratio);
    this->up_proj.forward(x, up_output);

    // ReLU
    allocate_aligned_memory(relu_output_ptr, batch_size * this->hidden_size * this->extend_ratio * sizeof(float));
    Matrix3D<float> relu_output = Matrix3D<float>(relu_output_ptr, 1, batch_size, this->hidden_size * this->extend_ratio);
    this->relu.forward(up_output, relu_output);

    // down_proj
    allocate_aligned_memory(down_output_ptr, batch_size * this->hidden_size * sizeof(float));
    Matrix3D<float> down_output = Matrix3D<float>(up_output_ptr, 1, batch_size, this->hidden_size);
    this->down_proj.forward(relu_output, down_output);
    
    PROFILE_END(profile_name);

    return down_output;
}