# 部署 (Deployment)

本项目中的文件归类及说明：

## infra/ (基础设施)
项目路径: `infra/milvus-standalone/`
- docker-compose.yml - Milvus + MinIO + etcd
- volumes/ - 数据持久化目录

## logs/ (日志)
项目路径: `logs/`
- app.log - 应用运行日志

## scripts/ (部署脚本)
项目路径: `scripts/`
- ingest_pdf.py - PDF导入脚本
