# Linly-Talker 项目代码分类说明

本文将当前项目按 `部署`、`测试`、`设计`、`研发`、`优化` 五类进行归类。

分类原则：

- 按“主要职责”归类，不按文件名机械归类。
- 同一路径如果同时承担多个职责，只放入一个“主分类”，并在说明中标注“兼任职责”。
- 媒体素材、模型权重、证书等非代码资源单独列出，避免和源码混在一起。

## 1. 部署

这类内容主要负责环境安装、依赖准备、模型下载、启动方式和运行配置。

| 路径 | 归类原因 |
| --- | --- |
| `scripts/` | 安装、下载、环境初始化脚本 |
| `requirements_app.txt` | 应用部署依赖 |
| `requirements_webui.txt` | WebUI 部署依赖 |
| `api/requirements.txt` | API 服务部署依赖 |
| `ASR/requirements_*.txt` | ASR 模块依赖清单 |
| `TTS/requirements_*.txt` | TTS 模块依赖清单 |
| `VITS/requirements_*.txt` | VITS/语音克隆依赖清单 |
| `TFG/requirements_*.txt` | 数字人生成模块依赖清单 |
| `AutoDL部署.md` | AutoDL 部署文档 |
| `colab_webui.ipynb` | Colab 启动与体验入口 |
| `configs.py` | 运行端口、模式、证书等部署配置 |
| `https_cert/` | HTTPS 运行所需证书 |
| `api/README.md` | API 启动与调用说明 |

补充说明：

- `app.py`、`webui.py`、`api/*.py` 也参与“启动”，但它们的主职责仍然是业务研发，因此归到“研发”。

## 2. 测试

这类内容主要负责接口验证、功能校验、单元测试或推理测试。

| 路径 | 归类原因 |
| --- | --- |
| `api/tts_client.py` | TTS API 测试客户端 |
| `api/llm_client.py` | LLM API 测试客户端 |
| `api/talker_client.py` | Talker API 测试客户端 |
| `src/test_audio2coeff.py` | 音频到系数流程测试脚本 |
| `src/facerender/sync_batchnorm/unittest.py` | 单元测试脚本 |
| `NeRF/data_utils/face_parsing/test.py` | NeRF 相关测试脚本 |
| `Musetalk/configs/inference/test.yaml` | MuseTalk 推理测试配置 |
| `LLM/__init__.py` 中的 `test_*` 方法 | 模型可用性快速验证 |

补充说明：

- 项目当前的测试代码比较分散，更多是“样例验证/推理验证”，不是集中式测试工程。

## 3. 设计

这里的“设计”包含产品说明、模块说明、界面展示、架构说明和配置设计。

| 路径 | 归类原因 |
| --- | --- |
| `README.md` | 项目总体设计与功能说明 |
| `README_zh.md` | 中文设计说明 |
| `docs/` | 架构图、界面图、说明文档 |
| `常见问题汇总.md` | 运行方案与问题设计沉淀 |
| `SECURITY.md` | 安全设计说明 |
| `ASR/README.md` | ASR 模块设计说明 |
| `TTS/README.md` | TTS 模块设计说明 |
| `VITS/README.md` | 语音克隆模块设计说明 |
| `TFG/README.md` | 数字人生成模块设计说明 |
| `LLM/README.md` | LLM 模块设计说明 |
| `face_detection/README.md` | 人脸检测模块说明 |
| `src/config/` | 核心生成流程配置设计 |
| `Musetalk/configs/` | MuseTalk 配置设计 |
| `GPT_SoVITS/configs/` | GPT-SoVITS 配置设计 |

补充说明：

- 如果你把“设计”理解为“文档 + 配置 + 界面方案”，这部分就是当前项目里最接近设计资产的内容。

## 4. 研发

这部分是项目的主体，负责核心业务逻辑、模型接入、推理流程和交互实现。

| 路径 | 归类原因 |
| --- | --- |
| `app.py` | 早期 Gradio 主入口 |
| `webui.py` | 当前多模块 WebUI 主入口 |
| `app_img.py` | 图片数字人相关入口 |
| `app_multi.py` | 多模块入口 |
| `app_musetalk.py` | MuseTalk 入口 |
| `app_talk.py` | 对话生成入口 |
| `app_vits.py` | 语音克隆入口 |
| `api/tts_api.py` | TTS 服务实现 |
| `api/llm_api.py` | LLM 服务实现 |
| `api/talker_api.py` | Talker 服务实现 |
| `ASR/` | 语音识别研发代码 |
| `TTS/` | 文本转语音研发代码 |
| `VITS/` | 语音克隆研发代码 |
| `TFG/` | 数字人驱动/视频生成研发代码 |
| `LLM/` | 大模型接入与对话研发代码 |
| `src/` | 底层推理、渲染、音视频与工具链核心代码 |
| `face_detection/` | 人脸检测核心代码 |
| `GPT_SoVITS/` | GPT-SoVITS 集成与训练/推理代码 |
| `Musetalk/` | MuseTalk 集成代码 |
| `NeRF/` | ER-NeRF/NeRF 相关核心代码 |

补充说明：

- 这一类占比最大，说明当前仓库是“以功能集成为中心”的研发型项目。
- `src/`、`GPT_SoVITS/`、`Musetalk/`、`NeRF/` 里有不少上游算法代码和二次集成代码混在一起，后续如果要做工程化治理，这部分最值得继续拆分。

## 5. 优化

这类内容主要服务于性能、显存、推理速度、资源占用和工程运行体验优化。

| 路径 | 归类原因 |
| --- | --- |
| `src/cost_time.py` | 统计耗时，便于性能分析 |
| `webui.py` 中的 `clear_memory()` | 显存回收与运行稳定性优化 |
| `src/utils/model2safetensor.py` | 模型格式转换优化 |
| `src/utils/safetensor_helper.py` | safetensor 使用优化 |
| `src/utils/face_enhancer.py` | 结果质量增强 |
| `scripts/install_pytorch3d.py` | 依赖安装适配与工程效率优化 |
| `ASR/OmniSenseVoice.py` | 更快的语音识别方案 |
| `TFG/Wav2Lipv2.py` | 相比旧版 Wav2Lip 的效果/性能优化方向 |
| `TTS/CosyVoice.py` | 高质量 TTS/克隆方案优化 |
| `VITS/CosyVoice.py` | 语音生成能力优化方向 |

补充说明：

- 这些文件并不一定只做优化，它们很多也是研发代码；这里只是按“最明显的工程价值”单列出来，方便你从项目管理角度识别优化资产。

## 6. 非代码资源

以下内容建议不要硬塞进上述五类“代码分类”，更适合作为运行资源或展示资源单独管理。

| 路径 | 建议归属 |
| --- | --- |
| `docs/*.png` `docs/*.jpg` | 设计素材/展示资源 |
| `examples/` | 示例素材 |
| `inputs/` | 输入样例与运行资源 |
| `src/config/*.mat` | 模型配置资源 |
| `checkpoints/` | 模型权重资源 |
| `https_cert/` | 运行证书资源 |

## 7. 当前项目的总体判断

如果从项目管理角度看，这个仓库当前更像下面这种结构：

1. `部署层`：`scripts/`、`requirements*.txt`、`AutoDL部署.md`
2. `设计层`：`README*`、`docs/`、各模块 `README`、配置目录
3. `研发层`：`app*.py`、`webui.py`、`api/*.py`、`ASR/`、`TTS/`、`TFG/`、`LLM/`、`src/`
4. `测试层`：分散在 `api/`、`src/`、`NeRF/`、`LLM/` 中
5. `优化层`：散落在 `webui.py`、`src/utils/`、模型替代方案代码中

## 8. 如果后续要真正重组目录

可以考虑按下面的方式做物理拆分：

- `deploy/`：部署脚本、依赖文件、启动说明
- `design/`：README、架构图、模块设计文档、配置模板
- `research/` 或 `src/`：核心业务与模型接入代码
- `tests/`：接口测试、单元测试、推理验证
- `optimize/`：性能分析、格式转换、显存/速度优化工具

这一步我还没有替你实际移动目录，只先完成“现状分类说明”，这样风险最小，也方便你下一步决定是否继续重构。
