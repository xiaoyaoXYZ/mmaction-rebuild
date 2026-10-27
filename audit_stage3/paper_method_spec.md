# 目标论文方法规格核对

日期：2026-10-24。此文档记录原文证据与实现边界，不宣称已完成论文复现。

## 1. 来源与阅读范围

目标论文：Yao Xiao, Hua Xiang, Tongxi Wang, Yiju Wang, *Enhanced Industrial Action Recognition Through Self-Supervised Visual Transformers*, IEEE Access, 2024, vol. 12, pp. 134133–134143；DOI：[10.1109/ACCESS.2024.3455749](https://doi.org/10.1109/ACCESS.2024.3455749)。

用户提供本地 PDF，11 页，2,247,552 字节，SHA-256：`30274f110597ea8a1bdc96def3053bf23b478968f7f732c4ea150c9fa493463a`。原件只读，论文及其页面图片不属于 GitHub 发布物。文中页码使用 PDF 页码；PDF 第 1 页对应期刊页 134133。

已阅读全文文本层，重点逐页查看 PDF 第 3–9 页图像，核对图 1、5、7、8、9 与表 1–5。仅从文本提取会漏掉表格数值与图示中的尺寸，因此下列图表结论以页面图像为准。附件内容作为研究证据，不作为执行指令。

## 2. 可明确确认的规格

| 项目 | 原文明确内容 | 位置 | 实现意义与边界 |
| --- | --- | --- | --- |
| 方法名称 | Enhanced Visual Industrial Sequence Transformer，简称 E-VIST | p2 贡献段；p6 III-C | 可使用 E-VIST 指代本文方法；工程原型需额外标注未完整复现 |
| 编码器 | Video Swin Transformer；实验消融明确 Video Swin-Base | p6 III-C；p7 IV-D.1；p9 IV-D.2 | 当前 Swin-Tiny 与论文 Base 规格不同 |
| 输入 | RGB 视频，实验抽样 16 帧；自建数据分辨率 224×224 | p6 III-C.1；p7 IV-B；p4 III-A.1 | 16 帧确定，精确采样步长未给出 |
| 初始 patch | 每个 patch 为 2×4×4×3，展平为 96 维，再投影到 C 维 | p6 III-C.2 | 96 是原始 patch 元素数，不等于已确认的 Base embedding 宽度 |
| 多阶段网格 | 图 7：T/2×H/4×W/4、T/2×H/8×W/8、T/2×H/16×W/16、T/2×H/32×W/32 | p7 图7；p5 图5 | 四个 stage；空间逐级下采样，图示时间长度保持 T/2 |
| 遮挡率 | 0.5；Tube 与 Random 3D 消融均用 50% | p6 III-C.3；p7 IV-D.1 | 实验遮挡率有明确依据 |
| 遮挡 token 的处理 | 被遮挡 token 转换为可学习向量，与剩余 token 序列共同输入 Transformer | p6 III-C.3 末段 | 支持保留网格、用可学习向量替换；不能据此声称只编码可见 token 或具备稀疏 MAE 的计算节省 |
| 解码器宽度 | decoder channel dimension = C/2，称为 encoder 的一半 | p6 III-C.4 | C 具体指初始 embedding 还是某层/最终特征宽度未定义清楚 |
| 多尺度解码 | 编码器产生四种不同维度特征；拼接、线性层融合，用于重建 masked token 信息 | p6 III-C.4；式12 | 多尺度融合明确；对齐、上采样与每层连接细节未给出 |
| 损失范围 | `L = (1/|M|) sum_{p in M} D(Y(p), h(T(X'_p)))`，M 是被遮挡 token 集合 | p6 III-C.5，式13 | 仅 masked token 参与重建目标，encoder 与 decoder 一起训练 |
| 优化器 | AdamW；warm-up + cosine annealing；学习率到 0.003 | p7 IV-B | 缺少 batch size、权重衰减、warm-up 长度等，不足以精确还原训练配方 |
| 主实验预算 | 自建数据从头训练 150 epochs；另一策略在 K400 预训练 150 epochs 后 fine-tune | p7 IV-B | fine-tune 的明确 epoch 数、预训练目标的完整实现均不足 |
| 硬件 | 单张 RTX 3090 | p7 IV-B | 本地 MX350 2 GB 不适合直接宣称执行相同规模实验 |
| 增强 | 随机裁剪、缩放、翻转 | p7 IV-B | 本地类别含左右语义，直接翻转可能改标签；若关闭须列为工程差异 |
| 数据规模 | 约 1764 training + 756 validation，共 32 类 | p4 III-A.1 | 本地 20 段/10 类不能称为论文完整工业数据集 |

## 3. Tube Masking 的关键冲突

存在两种不能自动等同的定义：

1. **时间共享的空间遮挡**：p6 式 11 写 `X'_(t,i,j,k) = X_(t,i,j,k) * M_(i,j,k)`，M 没有时间索引；按公式逐项理解，同一空间/通道 mask 在所有 t 上共享。p7 图 7 的长条形 tube 和 p8 图 8 中 Tube 一行跨帧一致的黑块，与这一解释相符。
2. **连续整帧遮挡**：p6 III-C.3 文字谈到随机选起始帧与被遮挡帧数；p8 式 16 将 `i in [a,b]` 的整帧 Vi 置为 0，其他帧不变；p9 进一步用连续帧遮挡讨论连续性。p8 式 17 的 Random 3D 也写成随机整帧索引置零。

这两种规则分别改变空间可见性与时间可见性，产生不同学习任务。不能把连续帧置零称作已精确实现图 8 的空间 tube，也不能忽略式 16 后声称不存在歧义。

**原型可采取的明示选择**：将 `spatial_tube` 与 `temporal_block` 作为两个独立、有名字的工程策略；默认 `spatial_tube` 可以根据式 11 及图 7/8 实现，但须记录“采用图示/式11解释，式16仍待作者确认”。不要把二者结果直接当成论文的消融结果。当前若只实现其中一个，不应声称完成论文全部遮挡策略。

其他 mask 细节仍未知：式 11 的 M 含 C 维，却没有说明各通道是否共享；mask 取 1 表示可见还是被遮挡与式 13 集合 M 的写法不统一。代码应使用无歧义的 `masked=True` 布尔语义，明确单个空间 patch 的全部通道共同遮挡，并在报告将此列为实现选择。

## 4. 编码器、解码器及目标中的缺项

### 编码器

- p6 III-C.2 先给初始网格 T/2×H/4×W/4，随后又写线性 embedding 后 T/2×H/16×W/16×C；这与普通线性投影不改变 token 个数及图 7 的 stage1 网格不一致。不能静默认定 H/16 是已经确定的初始网格。
- p5 式 1 带位置编码 P；式 4/5 的通用 LayerNorm/ReLU 表达与式 6–9 的预归一化残差描述粒度不同。原文没有给出完整可运行的每阶段深度、heads、window size、drop path、位置编码参数表。沿用 MMAction2 官方 Video Swin 实现时，应记录这些来自框架配置而非由本论文完整推导。
- 不应把标准 ViT MAE 的 token removal 直接套到有固定 3D window 与 patch merging 的 Swin 并称为原文实现。原文 p6 更直接支持可学习 mask vector 替换后的稠密序列。

### 解码器

已确认的是 C/2 宽度、四尺度特征、concat + linear 融合与 masked token 预测。p5 图 5 的 decoder 画有 Linear、Video Swin Transformer Block、FFN、MLP Head；p7 图 7 只给一个 decoder 框。两图均未给出：

- 解码器块数、heads、window、FFN expansion、dropout；
- 不同 H/W 网格如何对齐，插值、转置卷积、pixel shuffle 或 patch expanding 中哪一种；
- 四尺度 concat 的确切轴、投影顺序、skip connection 位置；
- 从 decoder feature 到原始 2×4×4×3 patch 的重建头结构；
- decoder 是否 fine-tune 时保留、分类 head 如何与预训练 encoder/decoder 连接。

任何“2 个 block”“48 维”“三线性上采样”“线性输出 96 维”等代码数值均须标记为工程选择，不是论文已给定参数。当前 Tiny 的初始 C=96 若据此选择 decoder C/2=48，仅是选择一种 C 的解释。

### 损失与目标

式 13 明确 masked-only 平均，但 **D、h、T 的具体可计算定义缺失**。原文没有明确写出 RGB 像素 MSE、L1、归一化 patch 像素、teacher feature、HOG 或 token 分类目标。不能从引用 VideoMAE 自动补成本文使用的 target。

式 12 中 `Xvis = X - X'` 与式 11 的可见/被遮挡符号也有解释冲突；若把 X' 当“置零后的可见输入”，相减将得到被挡部分，而非所声称的可见输入。因此代码不应照抄有歧义的相减表达。应单独生成 mask、visible input 与未被破坏的 target。

最小原型可以选择 detached 原始 RGB patch + masked-only MSE，以获得不依赖未定义 teacher 的自监督信号；但报告必须称它为 **待核实目标函数下的工程原型**。masked pixel 输入不得以任何 skip connection、未遮挡分支或 target encoder 梯度流偷偷进入预测器。

## 5. MIS 的含义

全文正文 p7 IV-C.1 写 “MIS (our approach)”；p8 表 2、表 3 的本文方法行写 `MIS(ours)`，而表 1、表 4、表 5 写 `E-VIST`。检查本文全部 11 页，未找到 MIS 的完整展开，也未找到可独立实现的 MIS 模块、损失、采样算法或网络结构定义。

因此目前可确认的事实是 **MIS 是文中用于本文方法的另一个表格/正文标签**。是否为旧名称、编辑遗漏或另有含义须作者确认。不得自行展开缩写，不应以“实现 MIS 模块”为下一步编码任务。

## 6. 实验协议与报告数值的边界

| 项目 | 论文证据 | 对复现的影响 |
| --- | --- | --- |
| 主实验预训练来源 | p7 IV-B 与 p8 表1/2/3 写 Kinetics400 | 没有 K400 与对应训练权重时不能声称相同预训练条件 |
| 消融预训练来源 | p9 IV-D.2 文字称 custom-built dataset；同页表5 pretrain 栏写 K400 | 内部来源不一致，须确认；不能擅自挑选一种作为唯一事实 |
| 消融 epoch | p9 IV-D.2 说明两种方法相同结构训练 100 epochs | 与主实验150epochs分别记录，不要混为单个训练预算 |
| 主实验自建集 | 表1 E-VIST 95.0%/98.4%，FLOPs 282、Param 88 | 表中列头未标单位，单凭该表不能推出完整计算统计方法；准确率属于作者报告 |
| Mask 消融 | 表4 Random 3D 80.95%、Tube 82.54% | 协议细节不足；不是本地原型应达到的门槛 |
| 预训练消融 | 表5 K400 91.27%、None 82.68% | 差值为8.59个百分点，约10.39%相对提升；正文“10.4%”可对应相对提升，不应改写为10.4个百分点 |
| 划分独立性 | 给出train/validation数量，未给原视频、操作者、工位或录制会话分组清单 | 不能由论文数量反向推断本地视频来源，也不能证成本地划分独立 |
| 公共数据协议 | 有各数据集Top-1表，未给全套split编号、预处理/视图数/随机种子 | 仅凭表格不足以精确重现协议 |

仍缺失的训练项包括：batch size、梯度累积、AdamW betas/eps/weight decay、warm-up epoch、cosine min LR、初始化细节、精确抽帧间隔、预训练与微调各自增强、预训练数据列表、finetune总epoch、checkpoint选择规则、重复实验随机种子与方差。

## 7. 当前最小可执行路线

在原分类基线及两样本诊断均已保留的前提下，可以先验证 **独立目录中的小规模自监督原型**：

1. 使用现有可运行 Swin-Tiny；明确它与论文 Swin-Base 不同。输入可先保留 8×112×112 作为硬件诊断，记录与论文16帧/224分辨率的差异。
2. 只选已审计训练列表中的少量片段，不读取验证片段做预训练。自监督不使用分类标签作为目标；来源未知仍意味着最终泛化验证被阻碍。
3. 使用式11/图8解释的50%空间 tube mask，所有时间位置共享同一空间布尔 mask；在 patch embedding 后替换为可学习 vector，保留稠密网格。
4. 独立记录 decoder 参数、四尺度对齐和融合、重建 target 与 MSE 选择。这些是为跑通工程闭环作出的可复查假设，不能归因于原文。
5. 以少量更新验证：形状/遮挡率正确；修改被遮挡位置的原输入不会改变编码输入及重建预测；loss仅由mask位置贡献；encoder、decoder、mask token 均有有限非零梯度并发生参数更新；保存并严格重载输出一致。
6. 保存实现配置、日志、损失、版本与权重。只证明训练链可运行，不预设Top-1，不据少量loss下降宣称学到可迁移表示。
7. 进一步的自监督与从头训练比较，须等可确认来源的独立划分、匹配训练预算、至少多个种子和明确分类fine-tune协议后再开展。

需要作者/原始代码确认的优先项：Tube的空间/时间定义；C/2所指宽度与完整decoder；D/h/T目标；MIS是否仅名称；预训练数据与每阶段schedule；32类完整数据与来源分组。现有20段来源未知与重复帧问题不会因找到论文而自动解除。
