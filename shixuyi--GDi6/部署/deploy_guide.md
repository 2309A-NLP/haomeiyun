# 部署指南

## 环境要求
- Python 3.9+
- Milvus (Docker)
- 依赖: pip install -r requirements.txt

## 启动步骤
1. 启动 Milvus: cd infra/milvus-standalone && docker-compose up -d
2. 启动服务: python run.py
3. 前端访问: http://localhost:8001

## 结构说明
- infra/ - Milvus + MinIO + etcd Docker 配置
- logs/ - 应用日志
- scripts/ - 部署运维脚本
