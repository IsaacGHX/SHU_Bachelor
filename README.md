# 上海大学本科学位论文

## 一、课题来源、意义与主要内容
### 课题来源：
- 本课题研究旨在解决图神经网络（Graph Neural Networks, GNNs）在测试阶段面临的分布偏移（Out-of-Distribution, OOD）问题，通过引入一种自监督的适配方法提高模型的泛化性能。

### 研究意义：
- 随着图神经网络在推荐系统、知识图谱补全、社交网络分析等领域的广泛应用，其在测试数据分布偏移情况下的性能退化问题显得尤为重要。
- 本研究希望通过提出一种名为GOAT（Graph Out-of-distribution Augmentation-to-Augmentation at Test-time）的全新的自监督框架，填补现有方法在测试时无标签数据适配中的空白。

### 主要内容：
1. **现状分析**：分析现有GNN模型在分布偏移场景下的局限性。
2. **方法设计**：提出一种基于数据增强与自监督学习的测试时适配方法（GOAT），包括拓扑感知特征偏置适配器的设计。
3. **实验验证**：在多个数据集上验证GOAT在节点分类任务中的性能提升。
4. **框架推广**：探索GOAT方法在其他图任务（如图分类、边预测）中的适用性。

---

## 二、目的的要求和主要技术指标
### 研究目标：
- 提出一种新颖的测试时适配方法，能够有效减少训练数据与测试数据分布差异对模型性能的影响。

### 技术指标：
1. **准确性指标**：在至少三个不同的分布偏移的场景下的基准数据集上，节点分类准确率较基础的方法 —— 经验风险最小化（*Empirical Risk Minimization*），有所提升。
2. **效率指标**：在资源受限环境下，使得模型能够在百万级别的边数量的图结构数据上进行应用。
3. **鲁棒性指标**：在不同分布偏移场景下，性能能够一致提升，并且达到统计显著性。

---

## 三、进度计划
1. **第一阶段**（第1-4周）：文献调研与相关技术梳理，撰写研究背景及综述。
2. **第二阶段**（第5-8周）：GOAT框架设计与核心算法实现。
3. **第三阶段**（第9-12周）：在公开数据集上进行实验并分析结果。
4. **第四阶段**（第13-15周）：论文撰写、修改与完善。
5. **第五阶段**（第16周）：论文定稿与答辩准备。

---

## 四、主要文献、资料和参考书
1. 文献：
   - Jin, Wei, et al. "Empowering Graph Representation Learning with Test-Time Graph Transformation." The Eleventh International Conference on Learning Representations.
   - Wu, Qitian, et al. "Handling Distribution Shifts on Graphs: An Invariance Perspective." International Conference on Learning Representations.
   - Hu, W. et al. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. *NeurIPS*.
   - Wang, Dequan, et al. "Tent: Fully test-time adaptation by entropy minimization." arXiv preprint arXiv:2006.10726 (2020).
   - ...
2. 工具：
   - PyTorch: 用于模型实现和实验。
   - DGL: 深度图学习框架。
   - ...
3. 数据集：
   - OGB-ArXiv
   - Elliptic
   - FB100
   - Cora
   - ...
4. 书籍：
   - 《图神经网络：理论与实践》
   - ...
