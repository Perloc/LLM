# 功能

- 以动态库形式插入到网络中用以实现某些算子
- 实现TensorRT不原生支持的层或结构
- 替换性能不足的层或结构
- 手动合并TensorRT没有自动融合的层或结构

>Plugin不能和其他层进行融合，Plugin节点前后可能会插入reformat节点增加开销

# Workflow
## 实现步骤

1. 继承*IPluginV2DynamicExt*类实现一个Plugin类
2. 继承*IPluginCreator*类实现一个PluginCreator类
3. 实现用于计算的CUDA C++ kernel
4. 将Plugin编译为动态库（.so）保存
5. 在TensorRT中加载和使用Plugin
## 导入步骤

1. 加载编译好的Plugin动态库
2. 在Plugin Registry中找到需要的Plugin
3. 通过Plugin Creator构造需要的Plugin
4. 将Plugin插入网络中/Parser自动识别
# Plugin和TensorRT的交互
## compile time
- TensorRT向Plugin传递**参数**和**权重**
- Plugin向TensorRT报告其输入输出的张量信息，包括**数量**、**Shape**、**DataType**和**Layout**组合
- Plugin向TensorRT报告其需要的**Workspace大小**
- TensorRT尝试各种可能的组合，选择性能最佳的输入输出组合
## runtime

TensorRT向Plugin提供输入输出的张量地址、Workspace地址和所在的stream
# Plugin的类型

- IPluginV2：支持单一input/output格式
- IPluginV2Ext：支持单一input格式和混合output格式
- IPluginV2IOExt：支持混合input/output格式，Implict Batch模式
- IPluginV2DynamicExt：支持混合input/output格式，Dynamic Shape模式
![[Pasted image 20231203162916.png]]
# 关键API

```c++
/*** Plugin ***/

// 获得每个输出张量的形状，Dynamic Shape模式输出张量不一定是编译期常量，因此需要用表达式计算
DimsExprs getOutputDimensions(...);

// 该Plugin是否支持当前尝试的组合
bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
	WHERE_AM_I();
	if (inOut[pos].format != TensorFormat::kLINEAR)
		return false;

	switch (pos) {
	case 0:
		return inOut[0].type == DataType::kFloat || inOut[0].type == DataType::kINT32;
	case 1:
		return inOut[1].type == inOut[0].type;
	case 2:
		inOut[2].type == DataType::kINT32;
	default:
		return false;
	}
	return false;
}

// 推理前调用，Dynamic Shape模式下，每当输入数据形状发生变化（调用context.set_binding_shape）时，调用该函数
void configurePlugin(...);

// 获得中间计算结果的存储空间，TensorRT用于显存优化
size_t getWorkspaceSize(...);

// 调用CUDA C++ kernel的入口
int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {...}

// 初始化Plugin和释放资源
int32_t initialize(...) noexcept;
void terminate(...) noexcept;

// 创建多个与源对象共享本engine资源的多个context
IPluginV2DynamicExt *clone(...) const noexcept;

// 申请/销毁使用context独占的cudnn或cublas资源
// attachToContext/detachToContext

// context/engine销毁时调用
void destroy() noexcept;

// 序列化
size_t getSerializationSize() const noexcept {
	return sizeof(nK_) + sizeof(nH_) + sizeof(float) * weight_.count;
}
void serialize(void *buffer) const noexcept; {
	char *data = reinterpret_cast<char*>(buffer);
	size_t offset = 0;
	memcpy(data + offset, &nK_, sizeof(nK_));
	offset += nK_;
	memcpy(data + offset, &nH_, sizeof(nH_));
	offset += nH_;
	size_t size = sizeof(float) * nK_ * nH_;
	memcpy(data + offset, weight_.values, size);
}

// 反序列化
IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
	return new XXXPlugin(name, serialData, serialLength);
}

/*** PluginCreator ***/
// 根据传入参数调用Plugin构造函数
IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept;
// 注册PluginCreator
REGISTER_TENSORRT_PLUGIN(XXXPluginCreator);
```
# FP16/INT8

- FP16：Plugin需要允许float16的输入输出张量类型，并实现float16的CUDA C++ kernel
- INT8：Plugin需要支持FP32，否则要手动指定所有输入输出张量的dynamic range，并且内部张量的dynamic range也要手动指定

# 经典案例

1. 整合零散算子和memory bound操作
2. 整合Self-Attention结构
3. 融合其他高度优化的CUDA kernel