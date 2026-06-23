# Linly-Talker

项目已按新的物理结构重组：

- `deploy/`：部署脚本、依赖、证书、Colab、AutoDL 文档
- `design/`：README、API 文档、设计说明、界面和架构资料
- `tests/`：API 客户端和测试脚本
- `src/`：核心源码、应用入口、API 实现、算法模块
- `optimize/`：性能与工程优化相关代码

主要文档入口：

- 中文说明：`design/README_zh.md`
- 英文说明：`design/README.md`
- API 文档：`design/api/README.md`

兼容说明：

- 根目录保留了 `webui.py`、`app.py`、`app_talk.py` 等兼容入口。
- 旧命令大多仍可继续使用，但真实实现已迁移到新目录结构。
