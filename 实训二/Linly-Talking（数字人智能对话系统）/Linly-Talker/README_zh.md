# Linly-Talker

项目已经按照新的物理目录结构重组：

- `deploy/`：部署脚本、依赖、证书、Colab、AutoDL 文档
- `design/`：说明文档、API 文档、界面图和架构资料
- `tests/`：接口测试客户端和测试脚本
- `src/`：核心源码、应用入口、API 实现、算法模块
- `optimize/`：性能与工程优化代码

主要文档入口：

- 中文文档：`design/README_zh.md`
- 英文文档：`design/README.md`
- API 文档：`design/api/README.md`

兼容说明：

- 根目录保留了 `webui.py`、`app.py`、`app_talk.py` 等兼容启动入口。
- 旧启动命令大多仍可使用，但真实代码已经迁移到新的结构中。
