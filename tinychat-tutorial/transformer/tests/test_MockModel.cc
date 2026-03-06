#include <chrono>
#include <cstring>

#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"
#include "MockModel.h"

void test_MockModel() {
    // 获取配置
    const int batch_size = 12, hidden_size = 128, extend_ratio = 4;

    // 读取 Input Activation
    MemoryAllocator mem_buf;
    Matrix3D<float> mockmodel_input(mem_buf.get_fpbuffer(batch_size*hidden_size), 1, batch_size, hidden_size); // 32bit
    mockmodel_input.load("tests/assets/mockmodel_input.bin");

    // 配置 model
    std::string param_path = "/home/yxlin/Github/Study-for-MIT6.5940_2024_TinyML-and-Efficient-Deep-Learning-Computing/tinychat-tutorial/utils/generatebin/mockmodel/net";
    MockModel mockmodel = MockModel(param_path, 128, 4);

    // 读取 Output Activation
    Matrix3D<float> mockmodel_output(mem_buf.get_fpbuffer(batch_size*hidden_size), 1, batch_size, hidden_size);
    mockmodel_output.load("tests/assets/mockmodel_output.bin");
    
    // forward
    Matrix3D<float> output = mockmodel.forward(mockmodel_input);

    // compare
    bool success = check_two_equal(mockmodel_output.m_data, output.m_data, output.length(), 1e-5);

    Profiler::getInstance().report();

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { 
    test_MockModel();
}
