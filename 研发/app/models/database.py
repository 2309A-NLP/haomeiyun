"""
数据库模型定义模块
功能：定义所有数据库表结构（ORM模型）、数据库连接、会话管理和表迁移
"""
from datetime import datetime
from sqlalchemy import (
    JSON,       # JSON类型字段，用于存储非结构化数据
    Boolean,    # 布尔类型字段
    Column,     # 数据库列定义
    DateTime,   # 日期时间类型
    Float,      # 浮点数类型（用于评分）
    ForeignKey, # 外键约束
    Integer,    # 整数类型
    String,     # 字符串类型（需要指定长度）
    Text,       # 长文本类型（无长度限制）
    create_engine,  # 创建数据库引擎
    inspect,        # 检查数据库结构（用于迁移）
    text,           # 执行原生SQL语句
)
from sqlalchemy.ext.declarative import declarative_base    # 创建ORM基类
from sqlalchemy.orm import sessionmaker                    # 创建会话工厂
from ..core.config import settings                       # 导入配置（数据库连接信息）
from ..utils.logger import logger                        # # 导入日志记录器

# ==================== 数据库连接配置 ====================

# 构建MySQL连接URL
# 格式: mysql+pymysql://用户名:密码@主机:端口/数据库名
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
    f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
)

# 创建数据库引擎
# pool_pre_ping=True: 每次使用连接前检查是否可用（防止连接已断开）
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

# 创建会话工厂
# autocommit=False: 不自动提交，需要手动commit
# autoflush=False: 不自动刷新，需要手动flush
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建ORM基类（所有模型都继承自它）
Base = declarative_base()


# ==================== 数据模型定义 ====================

class User(Base):
    """用户表 - 存储系统用户信息"""
    __tablename__ = "users"  # 数据库表名

    id = Column(Integer, primary_key=True, index=True)  # 用户ID（主键、自增、索引）
    username = Column(String(50), unique=True, index=True)  # 用户名（唯一、索引）
    email = Column(String(100), unique=True, index=True)  # 邮箱（唯一、索引）
    hashed_password = Column(String(255))  # 加密后的密码（bcrypt/sha256）
    phone = Column(String(20))  # 手机号
    is_active = Column(Boolean, default=True)  # 账户是否启用（软删除标记）
    created_at = Column(DateTime, default=datetime.now)  # 创建时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # 更新时间（自动更新）


class Role(Base):
    """角色表 - 存储系统角色和用户自定义角色"""
    __tablename__ = "roles"

    # ===== 基础字段 =====
    id = Column(Integer, primary_key=True, index=True)  # 角色ID
    name = Column(String(50), unique=True, index=True)  # 角色名称（如：lawyer、doctor）
    display_name = Column(String(100))  # 显示名称（如：专业律师）
    description = Column(Text)  # 角色描述
    avatar = Column(String(255))  # 头像URL或路径
    specialties = Column(JSON)  # 专长领域（JSON数组，如：["民法","刑法"]）
    prompt_template = Column(Text)  # 提示词模板
    is_active = Column(Boolean, default=True)  # 是否启用
    created_at = Column(DateTime, default=datetime.now)  # 创建时间

    # ===== 自定义角色字段（系统角色为NULL）=====
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)  # 创建者ID（外键关联用户表）
    is_public = Column(Boolean, default=False)  # 是否公开（公开角色可被其他用户使用）
    usage_count = Column(Integer, default=0)  # 使用次数（用于排序和推荐）
    rating = Column(Float, default=0.0)  # 平均评分（0-5分）
    rating_count = Column(Integer, default=0)  # 评分人数（用于计算平均分）
    tags = Column(JSON)  # 标签（JSON数组，如：["法律咨询","合同纠纷"]）
    system_prompt = Column(Text)  # 系统提示词（覆盖默认系统提示）
    answer_style = Column(String(50))  # 回答风格（如：professional, friendly, concise）


class ChatSession(Base):
    """聊天会话表 - 存储用户与角色的对话会话"""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)  # 会话ID
    user_id = Column(Integer, index=True)  # 用户ID（外键关联用户表）
    role_id = Column(Integer, index=True)  # 角色ID（外键关联角色表）
    title = Column(String(200))  # 会话标题（自动生成或用户命名）
    legal_field = Column(String(50))  # 法律领域（用于知识库检索过滤）
    status = Column(String(20), default="active")  # 会话状态：active（活跃）、archived（归档）、deleted（已删除）
    created_at = Column(DateTime, default=datetime.now)  # 创建时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # 更新时间


class KnowledgeDocument(Base):
    """知识文档表 - 存储上传的法律文档元数据"""
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)  # 文档ID
    title = Column(String(500))  # 文档标题
    doc_type = Column(String(50))  # 文档类型（如：law、regulation、case）
    legal_field = Column(String(50))  # 法律领域（如：contract、criminal）
    source = Column(String(500))  # 来源（文件名或用户提供的来源信息）
    file_path = Column(String(500))  # 服务器上的文件存储路径
    chunk_count = Column(Integer, default=0)  # 切分后的文本块数量
    status = Column(String(20), default="pending")  # 处理状态：pending（待处理）、processing（处理中）、completed（已完成）、failed（失败）
    metadata_json = Column(JSON)  # 元数据JSON（存储可变信息：文件大小、原始文件名、错误信息等）
    created_at = Column(DateTime, default=datetime.now)  # 创建时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # 更新时间


# ==================== 数据库操作函数 ====================

def get_db():
    """
    获取数据库会话（依赖注入使用）
    用法: db: Session = Depends(get_db)
    功能:
    1. 创建数据库会话
    2. 将会话传递给API端点
    3. 请求结束后自动关闭会话
    """
    db = SessionLocal()  # 创建会话
    try:
        yield db  # 返回会话供使用
    finally:
        db.close()  # 确保会话被关闭（释放连接）


def init_db():
    """
    初始化数据库
    功能：
    1. 创建所有表（如果不存在）
    2. 执行表结构迁移（添加新字段）
    """
    # 创建所有表（根据ORM模型定义）
    Base.metadata.create_all(bind=engine)

    # 执行roles表的迁移（添加自定义角色相关字段）
    _migrate_roles_table()


def _migrate_roles_table():
    """
    迁移roles表（添加自定义角色字段）
    为什么需要迁移？
    - 系统初期可能没有这些字段
    - 后续版本添加了自定义角色功能
    - 使用ALTER TABLE添加新字段，避免删表重建（保护已有数据）
    """
    # 检查roles表是否存在
    inspector = inspect(engine)
    if "roles" not in inspector.get_table_names():
        return  # 表不存在，无需迁移

    # 获取现有字段名集合
    existing_columns = {column["name"] for column in inspector.get_columns("roles")}

    # 需要添加的字段及其SQL语句
    alter_statements = {
        "owner_id": "ALTER TABLE roles ADD COLUMN owner_id INTEGER NULL",  # 创建者ID
        "is_public": "ALTER TABLE roles ADD COLUMN is_public BOOLEAN DEFAULT FALSE",  # 是否公开
        "usage_count": "ALTER TABLE roles ADD COLUMN usage_count INTEGER DEFAULT 0",  # 使用次数
        "rating": "ALTER TABLE roles ADD COLUMN rating FLOAT DEFAULT 0.0",  # 平均评分
        "rating_count": "ALTER TABLE roles ADD COLUMN rating_count INTEGER DEFAULT 0",  # 评分人数
        "tags": "ALTER TABLE roles ADD COLUMN tags JSON NULL",  # 标签
        "system_prompt": "ALTER TABLE roles ADD COLUMN system_prompt TEXT NULL",  # 系统提示词
        "answer_style": "ALTER TABLE roles ADD COLUMN answer_style VARCHAR(50) NULL",  # 回答风格
    }

    # 执行迁移（在事务中）
    with engine.begin() as conn:
        # 添加缺失的字段
        for column_name, statement in alter_statements.items():
            if column_name in existing_columns:
                continue  # 字段已存在，跳过
            conn.execute(text(statement))  # 执行ALTER TABLE
            logger.info("Migrated roles table: added column %s", column_name)

        # 创建owner_id索引（加速按创建者查询）
        refreshed_indexes = {index["name"] for index in inspect(engine).get_indexes("roles")}
        if "ix_roles_owner_id" not in refreshed_indexes:
            conn.execute(text("CREATE INDEX ix_roles_owner_id ON roles (owner_id)"))
            logger.info("Migrated roles table: created index ix_roles_owner_id")

# Role表设计亮点
# 支持两种角色类型
# 系统默认角色：owner_id = NULL, is_public = True
# 用户自定义角色：owner_id = 用户ID, is_public = 可选
