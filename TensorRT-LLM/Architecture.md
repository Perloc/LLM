# TensorRT-LLM Architecture

## 1 Model Definition

*tensorrt_llm.Builder*：在*tensorrt_llm.Builder.create_network*函数中调用*tensorrt.Builder*创建一个*tensorrt.INetworkDefinition*实例，*tensorrt.INetworkDefinition*对象将会被*tensorrt_llm.functional*中定义的free function填充

```python
# examples of free function

def activation(input: Tensor, act_type: trt.ActivationType) -> Tensor:
    layer = default_trtnet().add_activation(input.trt_tensor, act_type)   # default_trtnet() -> INetworkDefinition
    return _create_tensor(layer.get_output(0), layer)

# derived examples from activation
relu    = partial(activation, act_type=trt.ActivationType.RELU)
sigmoid = partial(activation, act_type=trt.ActivationType.SIGMOID)

# Specialized activation functions can be used to assemble more advanced functions
def silu(input: Tensor) -> Tensor:
    return input * sigmoid(input)
```

*tensorrt.ILayer*：build的计算图会通过*tensorrt.ILayer*进行遍历和转换
## Compilation

*tensorrt_llm.Builder*：在*build_serialized_network*函数中调用*tensorrt.Builder*将*tensorrt.INetworkDefinition*编译到engine中，编译成功将获得一个*tensorrt.IHostMemory*实例，该实例可以被存储为要给二进制文件
## Weight Bindings

网络的权重在编译时必须拿到，因此，必须在调用*tensorrt_llm.Builder.build_engine*前绑定权重参数
```python
# The Linear operator exposes two parameters (see tensorrt_llm/layers/linear.py):
class Linear(Module):
    def __init__(self, ...):
        self.weight = Parameter(shape=(self.out_features, self.in_features), dtype=dtype)
        self.bias   = Parameter(shape=(self.out_features, ), dtype=dtype)

# The parameters are bound to the weights before compiling the model. See examples/gpt/weight.py:
tensorrt_llm_gpt.layers[i].mlp.fc.weight.value = fromfile(...)
tensorrt_llm_gpt.layers[i].mlp.fc.bias.value   = fromfile(...)
```

编译后也可以更新权重，需要使用*tensorrt_llm.Builder*类的*refit_engine*方法
## Pattern-Matching and Fusion

算子融合可以减少内存（DRAM）转移和调用计算核心（Tensor Core）的次数，也会减少launch kernel的次数。经典的例子就是矩阵乘法或卷积+激活函数。识别可以融合的算子列表称为模式匹配
## Plugins（cpp/tensorrt_llm/plugins)

有一些更高级的融合算法（例如FlashAttention），TensorRT不能通过模式匹配识别出来，需要用户自己添加，这个功能叫Plugins

自定义量化Tensor的例子：
```c++
// In cpp/tensorrt_llm/plugins/quantizeTensorPlugin/quantizeTensorPlugin.cpp:

int QuantizeTensorPlugin::enqueue(...) {
    if (inputDesc[0].type == DataType::kFLOAT) {
        invokeQuantization<float>(...);
    } else {
        invokeQuantization<half>(...);
    }
    return 0;
}

// In cpp/tensorrt_llm/kernels/quantization.cu:

template <typename T>
void invokeQuantization(...) {
    // The standard <<< >>> construct to launch CUDA kernels
    quantizedKernel<<<grid, block, 0, stream>>>(...);
}
```
## Runtime

## Multi-GPU and Multi-Node Support

## In-flight Batching