import torch  
import time  
import ttnn  
import pytest  
from loguru import logger  
  
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)  
def test_trace_vs_no_trace_performance(device):  
    """对比使用 trace 和不使用 trace 的性能"""  
    device.enable_program_cache()  
      
    # 测试参数  
    shape = [1, 3, 512, 512]  
    num_iterations = 10  
      
    # 预分配张量  
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)  
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)  
      
    # 操作链定义  
    def run_op_chain(input_0, input_1):  
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))  
      
    # 准备测试数据  
    torch_input_0 = torch.rand(shape, dtype=torch.bfloat16)  
    torch_input_1 = torch.rand(shape, dtype=torch.bfloat16)  
    ttnn_input_0 = ttnn.from_torch(torch_input_0, layout=ttnn.TILE_LAYOUT)  
    ttnn_input_1 = ttnn.from_torch(torch_input_1, layout=ttnn.TILE_LAYOUT)  
      
    # 编译模型（预热）  
    ttnn.copy_host_to_device_tensor(ttnn_input_0, input_0_dev)  
    ttnn.copy_host_to_device_tensor(ttnn_input_1, input_1_dev)  
    run_op_chain(input_0_dev, input_1_dev)  
      
    print("=" * 60)  
    print("性能对比测试：使用 trace vs 不使用 trace")  
    print("=" * 60)  
      
    # 测试1：不使用 trace  
    logger.info("开始测试：不使用 trace")  
    ttnn.synchronize_device(device)  
      
    start_time = time.time()  
    for i in range(num_iterations):  
        ttnn.copy_host_to_device_tensor(ttnn_input_0, input_0_dev)  
        ttnn.copy_host_to_device_tensor(ttnn_input_1, input_1_dev)  
        output = run_op_chain(input_0_dev, input_1_dev)  
        _ = ttnn.to_torch(output)  # 读回结果  
    ttnn.synchronize_device(device)  
    no_trace_time = time.time() - start_time  
      
    print(f"不使用 trace - 总时间: {no_trace_time:.4f}s")  
    print(f"不使用 trace - 平均每次迭代: {no_trace_time/num_iterations:.4f}s")  
      
    # 测试2：使用 trace  
    logger.info("开始测试：使用 trace")  
      
    # 捕获 trace  
    ttnn.copy_host_to_device_tensor(ttnn_input_0, input_0_dev)  
    ttnn.copy_host_to_device_tensor(ttnn_input_1, input_1_dev)  
    tid = ttnn.begin_trace_capture(device, cq_id=0)  
    output_tensor = run_op_chain(input_0_dev, input_1_dev)  
    ttnn.end_trace_capture(device, tid, cq_id=0)  
      
    ttnn.synchronize_device(device)  
      
    # 执行 trace 性能测试  
    start_time = time.time()  
    for i in range(num_iterations):  
        ttnn.copy_host_to_device_tensor(ttnn_input_0, input_0_dev)  
        ttnn.copy_host_to_device_tensor(ttnn_input_1, input_1_dev)  
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)  
        _ = ttnn.to_torch(output_tensor)  # 读回结果  
    ttnn.synchronize_device(device)  
    trace_time = time.time() - start_time  
      
    print(f"使用 trace - 总时间: {trace_time:.4f}s")  
    print(f"使用 trace - 平均每次迭代: {trace_time/num_iterations:.4f}s")  
      
    # 性能对比分析  
    speedup = no_trace_time / trace_time  
    improvement = ((no_trace_time - trace_time) / no_trace_time) * 100  
      
    print("=" * 60)  
    print("性能对比结果:")  
    print(f"加速比: {speedup:.2f}x")  
    print(f"性能提升: {improvement:.1f}%")  
    print("=" * 60)  
      
    # 清理资源  
    ttnn.release_trace(device, tid)  
      
    return {  
        'no_trace_time': no_trace_time,  
        'trace_time': trace_time,  
        'speedup': speedup,  
        'improvement': improvement  
    }  
  
# 简化版本的独立测试函数  
def simple_trace_performance_test():  
    """简化的性能测试，可以独立运行"""  
    import ttnn  
      
    # 初始化设备  
    device = ttnn.open_device(0, {"trace_region_size": 200000})  
    device.enable_program_cache()  
      
    try:  
        # 运行性能测试  
        results = test_trace_vs_no_trace_performance(device)  
          
        print("\n测试完成！")  
        print(f"使用 trace 相比不使用 trace 快了 {results['speedup']:.2f} 倍")  
          
    finally:  
        ttnn.close_device(device)  
  
if __name__ == "__main__":  
    simple_trace_performance_test()