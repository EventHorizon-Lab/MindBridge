# 端侧人物一致性感知与 FunASR 收敛决策

> 文档性质：截至 2026-08-26 的实现与决策快照；研究链接提供选型背景，不代表持续更新的 SOTA 排名
>
> 状态：Phase 3 可执行基线，目标端侧平台真值验收前不得宣称人物一致性 SOTA
>
> 更新日期：2026-08-26
>
> 范围：原生视听流、人脸、VAD/ASR/标点/diarization、声纹、face↔voice 和端侧遗忘

## 1. 结论

MindBridge 当前默认链路只保留一套语音生态：

| 能力 | 当前默认 | 说明 |
| --- | --- | --- |
| 原生视频流 | GStreamer appsink（DeepStream 可选）→ `InsightFaceVideoEncoder.encode_frame()` | 直接接受带毫秒时间戳的 BGR frame，不重新打开视频；采集后端可替换，帧契约不变 |
| 原生音频流 | 16 kHz mono PCM16 → FunASR causal Paraformer | 任意输入 chunk；内部使用官方 600ms 窗口和 cache |
| Event-close 语音理解 | `SpeechAnalyzer` 契约 + 两个 FunASR 引擎：CUDA 上的 Nano vLLM，其余平台的 `AutoModel`（默认 Fun-ASR-Nano） | 契约只要求带时间范围的语音段和 speaker centroid；换模型是换 recipe，换引擎是换环境，都不换管线；两个引擎都产 centroid，所以引擎可由环境自动选 |
| 人脸 | InsightFace SCRFD + ArcFace | 在各目标端侧平台上再比较检测器档位；模板留在设备 |
| 声纹 | FunASR 返回的 CAM++ speaker centroid | 不再二次解码波形或重复加载 speaker encoder |
| 活跃说话人 | 带 face anchor 且保留音轨的 Omni/VLM 复核 | 只产出可撤销证据；LR-ASD 仍是本地质量候选 |
| 本地身份 | AES-256-GCM 加密 SQLite prototype | 绝对阈值 + runner-up margin；显式遗忘同步删除 |

因此，**当前产品路径应以 FunASR 替换 NeMo**。这次替换删除了第二套 diarization runtime、
ASR/turn 融合器、波形临时文件和独立 ERes2NetV2 重推理，开发者只需提供同步 AV、一个 FunASR
pipeline 和设备校准阈值。NeMo/Sortformer 不再是安装依赖；只有带真值的相同硬件 bake-off 证明其
DER/false-link 收益足以覆盖额外依赖和编排成本时，才作为质量挑战者重新评估。

这条链路的产品目标包含**全平台可用**：MindBridge 的 Edge 不限定 Jetson，同一套身份感知最终需要覆盖
地瓜 RDK、Rockchip RK、Intel/OpenVINO x86、通用 ARM 主机，以及把 4090/5090/A100 直接当作“端”的
GPU 主机。当前交付基线是 InsightFace ONNX Runtime 与 FunASR `AutoModel`，`edge` extra 只声明标准
x86_64 和 Apple Silicon 环境；RDK、RK 与其他 ARM/NPU 仍是待验收目标，不是已支持平台。平台差异只
允许出现在 runtime 与编译工件上，bbox 归一化、embedding 维度、身份门禁和遗忘语义在所有平台保持
一致。任何只能在单一厂商 SDK 上成立的做法都不进默认路径。

这不表示 FunASR 已经包办所有实时语义。在线 Paraformer 的 partial transcript 是 provisional；
punctuation、稳定 speaker label 和长期 voiceprint 在 Event/window 关闭后才确认。当前 realtime
diarization 上游会随窗口增长重新聚类，不能把早期 speaker number 当成稳定身份。

## 2. 为什么这比原组合更适合 MindBridge

原组合需要 FunASR 产 ASR/VAD、NeMo 产 speaker turns、MindBridge 自写 overlap/margin 融合，再把
每个 turn 解码为 WAV 交给独立 ERes2NetV2。它有三个问题：

1. 两个模型栈给出的边界需要自定义启发式合并，错误归属会直接污染声纹；
2. 相同音频被重复读取和推理，显存、冷启动和每一个端侧平台的镜像都变重；
3. 开发者必须理解四个中间类型，却仍无法获得真正的 microphone chunk API。

这次收敛正是 **Code is the Product** 的直接结果：它删除的是一整套 MindBridge 自维护的融合代码，
而不是新增一层封装。判断标准不是“支持的能力更多”，而是“更少的代码做同一件事且更难出错”。任何
反向提案——重新引入第二套 runtime、为未来平台预建 provider 抽象——都必须先证明它删掉的比加入的多。

[FunASR](https://github.com/modelscope/FunASR) 的官方 `AutoModel` 已能组合 VAD、ASR、标点和
speaker model，并在 `sentence_info` 中返回 `start/end/text/spk/timestamp`。当前上游的
`return_spk_center=True` 还直接返回与 `spk` 索引对应的 centroid。MindBridge 只解析这一份结果，
保留自己的领域职责：加密模板、匹配门槛、证据、撤销和云端安全输出。

减重边界必须诚实：FunASR Python 发行本身仍包含 Torch、ModelScope/Transformers 和音频科学计算
依赖，因此它不进入 Core/SDK/server。`mindbridge[edge]` 在 Linux/Windows x86_64 和 macOS 14+
Apple Silicon 安装 PyPI 通用模型栈；Linux ARM 镜像各自安装与平台 SDK 匹配的 Torch/ONNX
Runtime、FunASR 和 ModelScope（Jetson 用 JetPack/CUDA 工件，地瓜 RDK 用 OpenExplorer，RK 用
RKNN Toolkit）。减掉的是**重复模型栈和 MindBridge 代码**，不是把 FunASR 描述成小型纯 Python
包。

## 3. 与 M3-Agent、TaskMem 的差异

### 3.1 M3-Agent

依据 M3-Agent 公开实现：

| 环节 | M3-Agent | MindBridge |
| --- | --- | --- |
| 人脸 | InsightFace `buffalo_l`，CPU、5 FPS、HDBSCAN 与固定阈值 | 同样复用 InsightFace，但自动选择并核验 CUDA provider；有质量门、绝对阈值和 runner-up margin |
| 说话区间 | Gemini 看整段视频生成秒级 ASR 区间 | PCM causal ASR 给低延迟 partial；FunASR 质量路径给毫秒 VAD/ASR/punc/spk |
| 声纹 | 自行构造 ERes2NetV2，逐区间再次推理 | 直接使用同一次 FunASR/CAM++ 调用返回的 speaker centroid |
| face↔voice | Prompt 可直接写 `Equivalence` 并刷新图 | Omni 只能写可撤销 ASD 证据；跨 Observation 双向互为最佳后才解析 ID |
| 隐私/遗忘 | Benchmark 图预处理 | 生物模板只在加密设备库；Observation/identity tombstone 真实删除 |

MindBridge 复用了它的原子事件、稳定人物 ID、外观变化、对话、关系与因果要求，但没有复制固定阈值、
全模板均值匹配或让 Prompt 直接合并身份。错误合并人物的产品代价远高于暂时保留两个匿名 ID。

### 3.2 TaskMem

[TaskMem](https://github.com/ByteDance-Seed/TaskMem) 的公开路径以 10 秒 AV block 和约 50 秒滚动
上下文处理连续经历，并组合 ASR/diarization、face/voice 与 episode generation。MindBridge 采用其
滚动上下文和可视身份锚点思路，但不采用依赖训练/RL 的 memorization policy：模型必须冻结，学习
发生在记忆、索引、反馈和生命周期状态。

## 4. 当前可执行数据流

```mermaid
flowchart LR
    CAM["CSI / USB camera"] --> GST["GStreamer (DeepStream optional)"]
    GST --> FRAME["timestamped BGR frame"]
    FRAME --> FACE["InsightFace encode_frame"]

    MIC["microphone / AEC"] --> PCM["16 kHz mono PCM16"]
    PCM --> ONLINE["FunASR causal cache\nprovisional text"]
    PCM --> ROLL["bounded rolling fragment"]

    GST --> ROLL
    ROLL --> GATE["VAD / scene / task Event close"]
    GATE --> SPEECH["FunASR AutoModel\nVAD + ASR + punc + spk"]
    SPEECH --> CENTROID["speaker centroids"]

    FACE --> LOCAL["encrypted SQLite identity"]
    CENTROID --> LOCAL
    FACE --> ASD["anchored native-AV ASD evidence"]
    SPEECH --> ASD
    ASD --> LOCAL
    LOCAL --> OUTBOX["cloud-safe timed identities only"]
```

采集线程不等待模型。模型通过有界队列消费 frame/PCM；积压时先降低人脸/ASD 采样，不能破坏原始
时间轴或丢掉 rolling fragment。`FunASRStreamingTranscriber` 对可变 chunk 做缓存和异步背压，
模型失败后关闭该 session，避免上游 cache 已变化时重放同一 PCM。最后一个非空 chunk 必须携带
`is_final=True`；断流则同时丢弃 cache 和 provisional 文本。

## 5. 模型与门禁

### 5.1 人脸

[InsightFace](https://github.com/deepinsight/insightface) 的 SCRFD/ArcFace 生态提供多个检测器档位、
五点对齐和 ONNX Runtime。**ONNX 是选择它的关键原因**：同一份权重可以在 CUDA、OpenVINO、RKNN、
地瓜 BPU 和纯 CPU 上落地，档位划分因此按算力而不是厂商展开。当前代码直接复用 `buffalo_l` 验证质量
路径；产品 bake-off 比较：

- 嵌入式 NPU（Orin Nano、RDK X5、RK3588）：SCRFD-500MF 或 YuNet/SFace；
- 中算力 SoC / x86（Orin NX、高配 RK3588、Intel + OpenVINO）：SCRFD-2.5GF + ArcFace R50；
- 高算力端侧与 dGPU（AGX Orin、4090/5090/A100 主机）：SCRFD-10GF + ArcFace R50。

`encode_frame()` 只接受调用方已解码的 BGR frame、`timestamp_ms` 和 `duration_ms`，输出归一化 bbox
与 embedding。真实生产 tracking 交给平台原生组件（DeepStream/NvDCF、RKNN、OpenVINO 或平台自带
tracker）；MindBridge 不复制 decoder/tracker，也不为它们建统一抽象层。
只有清晰度、尺寸、姿态、遮挡和检测分数合格的轨迹样本才能进入 identity memory。

### 5.2 FunASR 在线与质量路径

当前固定模型如下：

```text
online_asr = iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online@v2.0.4
quality_asr = iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch@v2.0.4
vad = iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@v2.0.4
punc = iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727@v2.0.4
speaker = iic/speech_campplus_sv_zh-cn_16k-common@v2.0.2
```

在线模型使用官方 `(0, 10, 5)` chunk 配置，即 600ms 中间窗口，并保留 encoder/decoder look-back
cache。质量路径的一次 `generate()` 开启 `sentence_timestamp`、`return_spk_res` 和
`return_spk_center`。speaker centroid 只在以下条件全部满足时进入长期设备身份：

1. 同 label 的累计有效语音不少于部署门槛，当前默认 1 秒；
2. turn confidence 通过设备校准门槛；
3. 不与另一 speaker turn 重叠；
4. 最佳本地 prototype 通过绝对 cosine threshold，并以足够 margin 胜过第二名。

FunASR 当前结果没有可解释的逐 turn diarization probability，因此构造器中的 confidence 是物理环境
校准旋钮，不是模型置信度。重叠段会封顶为 0.5，只保留 observation-scoped transcript；不能让一个
全局 centroid 因混叠音频污染长期声纹。

### 5.3 llama.cpp 多端路径：已评估，不采纳

[llama.cpp](https://github.com/ggml-org/llama.cpp) 路径确实存在，而且是上游正式提供的
（`runtime/llama.cpp/`）：Fun-ASR-Nano、SenseVoiceSmall 和 Paraformer 三个模型都有 GGUF 导出脚本、
预转权重和 ggml 实现的 SAN-M encoder，单一静态二进制、量化权重（Nano 全量化约 1.3 GB）、运行期不需要
Python。上游验证也很扎实：encoder cosine 1.000000、kaldi fbank cosine 1.000000、端到端 CER 与
PyTorch 相差 0.02%。对「Torch/FunASR 基线难以部署的平台」这个诉求，它看起来正是答案。

**但它不能进这条管线，原因是能力缺口而不是工程量。** 上游 README 明确写着：standalone llama.cpp /
GGUF 二进制**既没有实现 CAM++ speaker embedding，也没有实现 speaker 聚类**；`--vad` 只切语音，不给
speaker 标签，要 speaker 就回去用 `AutoModel(spk_model="cam++")` 或 vLLM 服务。也就是说这条路径给得出
带时间轴的转写和模型自带标点，唯独给不出声纹。

而声纹不是这条管线的可选项，是它的目的：`SpeechAnalysis` 要的是「带时间范围的语音段 **加上这些语音段
所属的 speaker centroid**」，`FunASRRecipe` 甚至在构造时就拒绝缺 speaker model 的组合。一个产不出
centroid 的引擎接进来，只能让每个 segment 退成 observation 作用域，即在低算力平台上静默关掉跨
observation 的人物一致性——而低算力平台恰恰是最需要「这是同一个人」的家用与巡检场景。用一个默认就
削掉核心能力的引擎去换部署便利，不是取舍，是把问题挪到看不见的地方。

（本文档上一版写的怀疑因此被证实了一半：「没有上游证据证明同一运行时完整覆盖 punctuation、
diarization 和 CAM++ centroid」——punctuation 和时间轴有了，centroid 没有。）

因此非 CUDA 平台的语音路径就是 `AutoModel`：慢一些，但每个平台都能跑，而且回答得出「谁在说话」。
重新评估这条路径的条件是明确的：**上游在 GGUF 二进制里实现 CAM++ embedding 与聚类**。届时它是一个薄
适配器，不需要为它预留任何抽象——`SpeechAnalyzer` 契约已经是那个位置。

## 6. face↔voice 闭环

音频 diarization 只能回答“哪个匿名 speaker 在说话”，不能知道画面中的哪张脸属于该声纹。
MindBridge 当前将 face box 标为 `F0/F1/...`，保留原视频音轨，并通过异步 OpenAI SDK 让 Omni/VLM
联合检查口型、语音起止和可见行为。它只返回 `FaceVoiceAssociationEvidence`，不能直接合并模板。

一对 face↔voice 必须满足：

1. 时间区间真实重叠，排除画外音、屏幕人物、机器人 TTS 和明显音画不同步；
2. 证据绑定 Observation、精确区间和产出它的模型；
3. 跨多个 Observation 达到累计时长和加权 confidence；
4. face→voice 与 voice→face 双向互为最佳，且都以校准 margin 胜过第二名；
5. 新竞争证据或 tombstone 能立即让解析退回未绑定。

`SQLiteIdentityMemory` 不创建不可逆 equivalence。它保存有界证据，并在读取时解析后续上传的匿名
ID；加密 face/voice templates 仍各自存在。云端当前不会回写已经导入的历史 voice Entity；只有
产品明确需要历史回补时，才增加版本化、可撤销的 cloud alias 关系。

## 7. 自适应算力

`device=auto` 在 CUDA 可用时选择 GPU；显式请求 CUDA 而 runtime/provider 回落到 CPU 会立即失败。
同一条规则适用于其他加速器：显式请求 OpenVINO、RKNN 或 BPU 而实际回落到 CPU 同样必须报错，
静默降级会让功耗和延迟验收失去意义。本地入口在空闲加速器内存达到配置门槛时并发人脸与统一
speech pipeline，否则顺序复用同一加速器。CPU 承担 SQLite、AES、哈希、FFmpeg 控制与调度；没有
可用加速器时才执行模型降级。8 GiB 当前只是 5090 软件验证门槛，每个目标平台都必须与机器人主任务
共同校准出自己的数值。

2026-08-13 的 RTX 5090 功能验证使用真实仓库音频，不是 mock：

| 路径 | 输入 | 结果 | 实测 |
| --- | --- | --- | --- |
| FunASR 质量路径 | 20 秒 16kHz mono WAV | 6 个毫秒 speech segment、标点、1 个 speaker、192 维 centroid | 推理 1.283s；峰值 CUDA allocation 1.93 GiB |
| FunASR 在线路径 | 6 秒 PCM，60 次 100ms push | 6 个 partial/final 文本 delta，最终 offset 6000ms | 推理 0.369s；峰值 872 MiB |
| 加密声纹 | 同一真实 centroid，两个 Observation | 两次均为 device scope 且命中同一匿名 ID | 通过 |

这些数字证明当前代码真实使用 GPU 和完整数据流，只适用于这台 5090 与该样本；没有 DER/WER/真值
身份标注，也没有任何非 CUDA 平台的数据，因此不能据此宣称模型精度或端侧 SOTA。

## 8. 安装与依赖边界

- Core/SDK：`uv pip install .`；
- 任意端侧/机器人编排（Jetson、地瓜 RDK、RK、OpenVINO x86、ARM 主机、dGPU 工作站）：
  `uv pip install '.[edge]'`；Linux/Windows x86_64 与 macOS 14+ Apple Silicon 主机同时获得通用
  模型运行栈，Linux ARM 设备保留 JetPack、RKNN、OpenVINO 或 BPU 提供的 wheel；
- 云服务：`uv pip install '.[server]'`；
- Jina SentenceTransformers 服务：再叠加 `'.[cloud-models]'`；
- `edge` 在 Linux/Windows x86_64 和 macOS 14+ Apple Silicon 声明 InsightFace、ONNX Runtime、
  OpenCV、FunASR、ModelScope（由 FunASR 传递安装）、Torch 和 TorchAudio；Linux ARM 厂商镜像
  自行提供这些模型运行时；Intel macOS 因 PyTorch 2.13 无 wheel 而只安装编排栈；NumPy 因为被
  MindBridge 直接导入，在所有平台都由 `edge` 声明。

通用 lock 与发布 wheel 都从 PyPI 解析同一组版本；随包的 `onnxruntime` 是 CPU 人脸 provider，
CUDA/TensorRT 人脸 provider 仍由平台镜像提供。lock 不包含 Jetson Torch、RKNN Toolkit、OpenVINO
runtime 或 BPU 工具链。CI 用独立 venv 对 Core、Edge、Server 分别 build/install，并在
隔离解释器中导入对应入口，防止 Edge 再次拖入 FastAPI、Celery、MCP 或 PostgreSQL。FunASR 安装
成功还不等于模型路径可用；设备镜像必须在启动验收中真正加载一次模型，并核对**实际生效的
device/provider**——不能因为进程起得来就认为加速器在工作。

## 9. 真值 bake-off 与 SOTA 门槛

公开数据和一套经同意采集的机器人 replay 必须同时报告：

- 人脸：track recall、TAR@固定 FAR、false merge、跨日 IDF1；
- diarization：DER/JER、speaker confusion、overlap DER、在线标签稳定性；
- ASR：中英/噪声/远场分桶 WER/CER；
- 声纹：EER、minDCF、TAR@固定 FAR，按 2s/3s/full、语言与距离分桶；
- ASD/闭环：mAP、off-screen FP、pair precision/recall、false-link、time-to-link、撤销延迟；
- 系统：完整链路 RTF/FPS、P50/P95、加速器/CPU/RAM/显存占用、功耗、温度、丢帧和队列深度，
  **按平台分别报告**，不做跨平台外推。

质量指标（人脸、diarization、ASR、声纹、ASD）追求的是绝对分数，评测时可以直接使用各自当前最强的
可部署模型与配置，不要求为了可比性刻意压低档位；系统指标则必须来自真实目标平台。两类指标同时
报告，不允许用其中一类掩盖另一类。

候选晋级顺序是：先满足 false-link 与隐私/遗忘约束，再最大化 recall，最后在合格候选中选择更轻的
运行时。两小时目标功耗 soak、机器人主任务并发、断电重启、幂等重放和 tombstone 都是门禁。

下一步只做三件事：固定带真值的跨日 identity replay；在至少一个 NVIDIA 平台和一个非 NVIDIA 平台
（地瓜 RDK、RK 或 OpenVINO x86）实测当前 FunASR 路径；用同一 replay 决定 LR-ASD 是否替换 Omni
ASD。没有结果前不恢复 NeMo 并行栈，也不新增 provider 框架——包括不为多平台预建抽象层。
