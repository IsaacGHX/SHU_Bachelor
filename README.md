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
#### Paradigm & Losses
- *全名*(**简称 publish**) | [PAPER](...) | [CODE](...) | [CITE](...)
- *Graph Contrastive Learning with Augmentations*(**GraphCL NIPS2020**) | [PAPER](https://arxiv.org/abs/2010.13902) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2010.13902)
- *Invariant Risk Minimization*(**IRM ArXiv2019**) | [PAPER](https://arxiv.org/abs/1609.02907) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=13187382171790179664&hl=en&as_sdt=0,5)
 - *Domain Generalization via Invariant Feature Representation*(**DomainInvariant PMLR2013**) | [PAPER](...) | [CODE](...) | [CITE](...)

#### GNN Backbones & Pre-train
- *Semi-Supervised Classification with Graph Convolutional Networks*(**GCN ILCR2022**) | [PAPER](https://arxiv.org/abs/1609.02907) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=9692529718922546949&hl=en&as_sdt=0,5)
- *Inductive representation learning on large graphs*(**SAGE NIPS2017**) | [PAPER](https://arxiv.org/abs/1710.10903) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=10802896480404413344&hl=en&as_sdt=0,5)
- *Graph Attention Networks*(**GAT ICLR2018**) | [PAPER](https://arxiv.org/abs/1706.02216) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=5609128480281463225&hl=en&as_sdt=0,5)
- *Adaptive Universal Generalized PageRank Graph Neural Network*(**GPR ICLR2021**) | [PAPER](https://arxiv.org/abs/2006.07988) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=17989054169887872189&hl=en&as_sdt=0,5)
- *Graph Transformer Networks*(**GTN NIPS2021**) | [PAPER](https://arxiv.org/abs/1911.06455) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1911.06455)
- *GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders*(**GraphVAE ICANN2018**) | [PAPER](https://arxiv.org/abs/1802.03480) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1802.03480)
  - FOR SMALL GRAPH. MAY NOT BE CITED 
- *Learning on Graphs with Out-of-Distribution Nodes*(**OODGAT KDD2022**) | [PAPER](https://arxiv.org/abs/2308.06714) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2308.06714)
- *Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data*(**SR-GNN NIPS2021**) | [PAPER](https://arxiv.org/abs/2108.01099) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2108.01099)
- *Unsupervised Domain Adaptive Graph Convolutional Networks*(**UDA-GCN PWC2020**) | [PAPER](https://dl.acm.org/doi/abs/10.1145/3366423.3380219) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Man+Wu+and+Shirui+Pan+and+Chuan+Zhou+and+Xiaojun+Chang+and+Xingquan+Zhu&btnG=)
  - NEED TO TAKE A LOOK!!!
- *Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks*(**Cluster-GCN KDD2019**) | [PAPER](https://arxiv.org/abs/1905.07953) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1905.07953)
- *Strategies for Pre-training Graph Neural Networks*(**GNNPretrain ICLR2020**) | [PAPER](https://arxiv.org/abs/1905.12265) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=8697784444104397361&hl=en&as_sdt=0,5)
- *Data Augmentation for Graph Neural Networks*(**GNNPretrainAugmentation AAAI2021**) | [PAPER](https://arxiv.org/abs/2006.06830) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2006.06830#d=gs_cit&t=1735391297843&u=%2Fscholar%3Fq%3Dinfo%3A-g5cAlwDzWIJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)


#### Datasets
- **(**EERM-codebase ICLR2023**) | [PAPER](https://arxiv.org/abs/2202.02466) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar?cluster=15550862662340330123&hl=en&as_sdt=0,5)
- *Revisiting Semi-Supervised Learning with Graph Embeddings*(**Cora PLMR2016**) | [PAPER](https://arxiv.org/abs/1603.08861) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1603.08861)
- *Pitfalls of Graph Neural Network Evaluation*(**Amazon Photo ICLR2023**) | [PAPER](https://arxiv.org/abs/1811.05868) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1811.05868)
- *Multi-scale Attributed Node Embedding*(**Twitch-E JCN2021**) | [PAPER](https://arxiv.org/abs/1909.13021) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1909.13021)
- *Social Structure of Facebook Networks*(**Facebook-100 Physica-A**) | [PAPER](https://arxiv.org/abs/1102.2166) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1102.2166)
- *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs*(**Elliptic AAAI2021**) | [PAPER](https://arxiv.org/abs/1902.10191) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1902.10191)
- *Open Graph Benchmark: Datasets for Machine Learning on Graphs*(**OGB-AriXiv NIPS2020**) | [PAPER](https://arxiv.org/abs/2005.00687) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2005.00687)
- *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*(**OODbenchmark ArXiv2019**) | [PAPER](https://arxiv.org/abs/1903.12261) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=4440880036617273374&hl=en&as_sdt=0,5)


#### OOD Detect
- *Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning*(**CMD ICLR2022**) | [PAPER](https://arxiv.org/abs/1702.08811) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=11153161380714770222&hl=en&as_sdt=0,5)
- *Uncertainty Quantification over Graph with Conformalized Graph Neural Networks*(**StructureShifts NIPS2024 OODdetect**) | [PAPER](https://arxiv.org/abs/2305.14535) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2305.14535)
- *A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability*(**AAGOD KDD2023 OODdetect**) | [PAPER](https://dl.acm.org/doi/abs/10.1145/3580305.3599244) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Data-centric+Framework+to+Endow+Graph+Neural+Networks+with+Out-Of-Distribution+Detection+Ability&btnG=)

#### OOD Resolution
- *Handling Distribution Shifts on Graphs: An Invariance Perspective*(**EERM ICLR2023 Node Classification**) | [PAPER](https://arxiv.org/abs/2202.02466) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar?cluster=15550862662340330123&hl=en&as_sdt=0,5)
- *Discovering Invariant Rationales for Graph Neural Networks*(**ICLR2022 Graph Classification**) | [PAPER](https://arxiv.org/abs/2201.12872) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2201.12872)
- *Learning invariant graph representations for out-of-distribution generalization*(**NIPS2022 Graph Classification**) | [PAPER](https://arxiv.org/abs/2202.05441) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2202.05441)
- *Graph Learning under Distribution Shifts: A Comprehensive Survey on Domain Adaptation, Out-of-distribution, and Continual Learning*(**OOD_Survey_Wu ArXiv**) | [PAPER](https://arxiv.org/abs/2402.16374) | [CODE](相关code的超链接) | [CITE](https://scholar.google.com/scholar?cluster=7367020713387072072&hl=en&as_sdt=0,5)
- *Beyond Generalization: A Survey of Out-Of-Distribution Adaptation on Graphs*(**OOD_Survey_Liu ArXiv**) | [PAPER](https://arxiv.org/abs/2402.11153) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2402.11153)
- *From Local Structures to Size Generalization in Graph Neural Networks*(**OODgeneralizationStructure PMLR2021**) | [PAPER](https://arxiv.org/abs/2010.08853) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2010.08853)
- *Domain-invariant Graph for Adaptive Semi-supervised Domain Adaptation*(**InvariantDomain ACMTMCCA2022**) | [PAPER](https://dl.acm.org/doi/10.1145/3487194) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Domain-invariant+Graph+for+Adaptive+Semi-supervised+Domain+Adaptatio&btnG=)


#### Graph with Prompts
- *Subgraph-level Universal Prompt Tuning*(**SUPT ArXiv2024**) | [PAPER](https://arxiv.org/abs/2402.10380) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2402.10380)
- *MultiGPrompt for Multi-Task Pre-Training and Prompting on Graphs*(**MultiGPrompt ACM2024**) | [PAPER](https://arxiv.org/abs/2312.03731) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2312.03731)
- *GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks*(**GPPT KDD2022**) | [PAPER](https://dl.acm.org/doi/abs/10.1145/3534678.3539249) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Gppt%3A+Graph+pre-training+and+prompt+tuning+to+generalize+graph+neural+networks&btnG=)
- *Universal Prompt Tuning for Graph Neural Networks*(**GPF ICLR2024**) | [PAPER](https://arxiv.org/abs/2209.15240) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2209.15240)
- *An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations*(**Prompt_Recommandation_System NIPS2023**) | [PAPER](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c6af791af7ef0f3e02bccef011211ca5-Abstract-Conference.html) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=An+Empirical+Study+Towards+Prompt-Tuning+for+Graph+Contrastive+Pre-Training+in+Recommendations&btnG=)
- *All in One: Multi-task Prompting for Graph Neural Networks*(**All_in_One KDD2023**) | [PAPER](https://arxiv.org/abs/2307.01504) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=17322478293736418371&hl=en&as_sdt=0,5)
- *GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks*(**GraphPrompt ACM2023**) | [PAPER](https://arxiv.org/abs/2302.08043) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2302.08043)

#### Test-time & Graphs
- *Source Free Unsupervised Graph Domain Adaptation*(**SOGA ACMICWSDM2024**) | [PAPER](https://arxiv.org/abs/2112.00955) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2112.00955)
- *Empowering Graph Representation Learning with Test-Time Graph Transformation*(**GTrans ICLR2023**) | [PAPER](https://arxiv.org/abs/2210.03561) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=12479531068894109978&hl=en&as_sdt=0,5)
- *DropEdge: Towards Deep Graph Convolutional Networks on Node Classification*(**DropEdge ICLR2020**) | [PAPER](https://arxiv.org/abs/1907.10903) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1907.10903)
- *Collaborate to Adapt: Source-Free Graph Domain Adaptation via Bi-directional Adaptatio*(**GraphCTA WWW2024**) | [PAPER](https://arxiv.org/abs/2403.01467) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2403.01467)
- *MEMO: Test Time Robustness via Adaptation and Augmentation*(**MEMO NIPS2022**) | [PAPER](https://arxiv.org/abs/2110.09506) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2110.09506#d=gs_cit&t=1735208209453&u=%2Fscholar%3Fq%3Dinfo%3ANvFr-mM_fMQJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)
- *SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks*(**SLAPS NIPS2021**) | [PAPER](https://arxiv.org/abs/2102.05034) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2102.05034)
- *TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?*(**TTT++ NIPS2023**) | [PAPER](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html) | [CODE](...) | [CITE](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=TTT%2B%2B%3A+When+Does+Self-Supervised+Test-Time+Training+Fail+or+Thrive%3F&btnG=)
- *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*(**TTT PMLR2020**) | [PAPER](https://arxiv.org/abs/1909.13231) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=1909.13231)
- *Test-Time Training for Graph Neural Networks*(**GT3 ArXiv2022**) | [PAPER](https://arxiv.org/abs/2210.08813) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=4459392671520994454&hl=en&as_sdt=0,5)
- *GraphTTA: Test Time Adaptation on Graph Neural Networks*(**GraphTTA ArXiv2022**) | [PAPER](https://arxiv.org/abs/2208.09126) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=3397674391742244376&hl=en&as_sdt=0,5)
- *Tent: Fully Test-time Adaptation by Entropy Minimization*(TENT ICLR2021) | [PAPER](https://arxiv.org/abs/2006.10726) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=2996193136579278806&hl=en&as_sdt=0,5)
- *Virtual Node Tuning for Few-shot Node Classification*(**VNodes KDD2023**) | [PAPER](https://arxiv.org/abs/2306.06063) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2306.06063)


#### Test-time & Prompt in CV & NLP
- *Learning to Prompt for Vision-Language Models*(**CoOp IJCV2023**) | [PAPER](https://arxiv.org/abs/2109.01134) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2109.01134)
- *DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting*(**DenseCLIP CVPR2022**) | [PAPER](https://arxiv.org/abs/2112.01518) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2112.01518)
- *Conditional Prompt Learning for Vision-Language Models*(**CoCoOp CVPR2022**) | [PAPER](https://arxiv.org/abs/2203.05557) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2203.05557)
- *Learning to Prompt for Continual Learning*(**L2P CVPR2022**) | [PAPER](https://arxiv.org/abs/2112.08654) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2112.08654)
  - Whether it should be cited need to be considered.
- *Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models*(**TPT NIPS2022 VLMprompt**) | [PAPER](https://arxiv.org/abs/2209.07511) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2209.07511)
- *Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning*(**DiffTPT ICCV2023**) | [PAPER](https://arxiv.org/abs/2308.06038) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2308.06038)
- *Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification*(**Tip-Adapter ECCV2022**) | [PAPER](https://arxiv.org/abs/2207.09519) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=3256821949763414308&hl=en&as_sdt=0,5)
- *Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing*(**PromptNLP_survey ACMCS2023**) | [PAPER](https://arxiv.org/abs/2107.13586) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=3155602780841366325&hl=en&as_sdt=0,5)
- *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*(**P-Tuning v2 ACL2022Spotlight**) | [PAPER](https://arxiv.org/abs/2110.07602) | [CODE](...) | [CITE](https://scholar.google.com/scholar?cluster=2013484515801163267&hl=en&as_sdt=0,5)
- *LoRA: Low-Rank Adaptation of Large Language Models*(**LORA ICLR2022**) | [PAPER](https://arxiv.org/abs/2106.09685) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2106.09685)
- *CLIP-Adapter: Better Vision-Language Models with Feature Adapters*(**CLIP-Adapter IJCV2024**) | [PAPER](https://arxiv.org/abs/2110.04544) | [CODE](...) | [CITE](https://scholar.google.com/scholar_lookup?arxiv_id=2110.04544)
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
