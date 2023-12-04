![[Pasted image 20231203131932.png]]
![[Pasted image 20231203134606.png]]
## Logger

![[Pasted image 20231203135209.png]]
## Builder

![[Pasted image 20231203135230.png]]
![[Pasted image 20231203135332.png]]
## Network

![[Pasted image 20231203135351.png]]
![[Pasted image 20231203135444.png]]
![[Pasted image 20231203135636.png]]
## Layer and Tensor

![[Pasted image 20231203135815.png]]
# Checkpoint

![[Pasted image 20231203140052.png]]
# Quantization
![[Pasted image 20231203140234.png]]
![[Pasted image 20231203140412.png]]
# Runtime

![[Pasted image 20231203140507.png]]
## Engine

![[Pasted image 20231203140539.png]]
## Context

![[Pasted image 20231203140623.png]]
## Input and Output

![[Pasted image 20231203140709.png]]
## Buffer

![[Pasted image 20231203140739.png]]
# Serialization and deserialization

![[Pasted image 20231203140846.png]]
# Paser (from ONNX)

![[Pasted image 20231203141235.png]]

# 辅助工具

- trtexec：TensorRT命令行工具，端到端性能测试工具
- Netron：网络可视化
- onnx-graphsurgeon：onnx计算图编辑
- polygraphy：结果验证与定位，图优化
- Nsight Systems：性能分析
## trtexec

### 功能

1. 由ONNX文件生成TensorRT引擎并序列化为Plan文件
2. 查看ONNX文件或Plan文件的网络逐层信息
3. 模型性能测试
### 从ONNX构建TensorRT引擎并推理

```shell
trtexec \
	--onnx=model.onnx \
	--minShapes=x:0:1x1x28x28 \
	--optShapes=x:0:4x1x28x28 \
	--maxShapes=x:0:16x1x28x28 \
	--workspace=1024 \
	--saveEngine=model-FP32.plan \
	--shapes=x:0:4x1x28x28 \
	--verbose \
	> result-FP32.txt
	
# 读取result-FP32.plan并进行推理
trtexec \
	--loadEngine=./model-FP32.plan \
	--shapes=x:0:4x1x28x28 \
	--verbose \
	> result-load-FP32.txt
```
## onnx-graphsurgeon
### 功能

- **修改计算图**：图属性/节点/张量/节点和张量的连接/权重
- **修改子图**：添加/删除/替换/隔离
- **优化计算图**：常量折叠/拓扑排序/去除无用层
## polygraphy

深度学习模型调试器
### 功能

- 使用多种后端运行推理计算
- 比较不同后端的逐层计算结果
- 由模型文件生成TensorRT引擎并序列化为plan文件
- 查看模型网络的逐层信息
- 修改ONNX模型，如提取子图，计算图化简
- 分析ONNX转TensorRT失败原因，将原计算图中可以/不可以转TensorRT的子图分割保存
- 隔离TensorRT中的错误tactic
