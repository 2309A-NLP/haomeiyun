# 研发指南

## 项目结构
- api/routes.py - API 路由
- core/config.py - 配置管理
- core/logging.py - 日志管理  
- ingest/pdf_parser.py - PDF 解析
- llm/client.py - LLM 客户端
- models/schemas.py - 数据模型
- services/*.py - 业务服务层

## 开发流程
1. 修改配置: .env
2. 修改路由: app/api/routes.py
3. 修改服务: app/services/*.py
4. 重启服务: python run.py --reload
