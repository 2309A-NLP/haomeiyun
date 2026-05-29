# app/models/schemas.py
# 定义了系统中所有数据验证、序列化和反序列化的Pydantic模型
# 📚 什么是Pydantic Schema？
# Pydantic Schema 就是利用 Pydantic 库定义的数据模型蓝图，用来规定数据结构应该长什么样
# 可以把它理解成一个严格的表格模板或数据合同，明确告诉程序：传入或传出的数据，每个字段叫什么名字、是什么类型、是否必需、取值范围是什么。
# # Pydantic的作用：
# 1. 验证请求数据格式（类型、长度、范围等）
# 2. 自动生成API文档
# 3. 序列化Python对象 → JSON
# 4. 反序列化JSON → Python对象
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class LegalField(str, Enum):
    """法律领域枚举"""
    CIVIL = "civil"  # 民法
    CRIMINAL = "criminal"  # 刑法
    LABOR = "labor"  # 劳动法
    CONTRACT = "contract"  # 合同法
    FAMILY = "family"  # 婚姻家庭
    IP = "ip"  # 知识产权
    COMPANY = "company"  # 公司法
    ADMINISTRATIVE = "administrative"  # 行政法


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """聊天消息"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    citations: Optional[List[Dict]] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, max_length=10000, description="用户消息")
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    role_id: str = Field(default="lawyer", description="角色ID")
    legal_field: Optional[LegalField] = Field(None, description="法律领域")
    location: Optional[str] = Field(None, description="用户所在地（用于地方法规）")
    stream: bool = Field(default=False, description="是否流式输出")


class Citation(BaseModel):
    """引用来源"""
    source: str = Field(..., description="来源")
    article: Optional[str] = Field(None, description="条文编号")
    content: Optional[str] = Field(None, description="引用内容")
    score: Optional[float] = Field(None, description="相似度得分")


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str = Field(..., description="AI回复")
    citations: List[Citation] = Field(default=[], description="引用来源")
    suggested_questions: List[str] = Field(default=[], description="推荐追问")
    risk_level: str = Field(default="low", description="风险等级：low, medium, high")
    disclaimer: str = Field(..., description="免责声明")
    session_id: str = Field(..., description="会话ID")
    response_time: Optional[float] = Field(None, description="响应时间（秒）")


class KnowledgeQueryRequest(BaseModel):
    """知识库查询请求"""
    query: str = Field(..., description="查询内容")
    legal_field: Optional[LegalField] = Field(None, description="法律领域过滤")
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeUploadJsonRequest(BaseModel):
    title: str = Field(..., min_length=1, description="文档标题")
    doc_type: str = Field(default="general", description="文档类型")
    legal_field: str = Field(default="general", description="法律领域")
    source: Optional[str] = Field(None, description="来源")
    filename: str = Field(..., min_length=1, description="原始文件名")
    content_base64: str = Field(..., min_length=1, description="Base64 编码的文件内容")


class KnowledgeDocumentResponse(BaseModel):
    """知识库文档响应"""
    id: int
    title: str
    doc_type: str
    legal_field: str
    source: Optional[str]
    chunk_count: Optional[int] = 0
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class UserRegisterRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    email: Optional[str] = Field(None, description="邮箱")
    phone: Optional[str] = Field(None, description="手机号")


class UserLoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    phone: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class RoleCreateRequest(BaseModel):
    """创建角色请求"""
    name: str = Field(..., description="角色标识")
    display_name: str = Field(..., description="显示名称")
    description: Optional[str] = None
    specialties: List[str] = Field(default=[])
    prompt_template: Optional[str] = None


class RoleResponse(BaseModel):
    """角色响应"""
    id: int
    name: str
    display_name: str
    description: Optional[str]
    specialties: Optional[List[str]]
    is_active: bool

    class Config:
        from_attributes = True


# ============ 自定义角色相关模型 ============

class AnswerStyle(str, Enum):
    """回答风格枚举"""
    FORMAL = "formal"  # 正式严谨
    FRIENDLY = "friendly"  # 友好亲切
    PROFESSIONAL = "professional"  # 专业权威
    CONCISE = "concise"  # 简洁明了
    DETAILED = "detailed"  # 详细全面


class CustomRoleCreateRequest(BaseModel):
    """创建自定义角色请求"""
    name: str = Field(..., min_length=2, max_length=50, description="角色名称")
    display_name: str = Field(..., min_length=2, max_length=100, description="显示名称")
    description: str = Field(..., min_length=10, max_length=500, description="角色描述")
    specialties: List[str] = Field(..., min_items=1, max_items=10, description="专业领域")
    prompt_template: Optional[str] = Field(None, description="提示词模板")
    system_prompt: Optional[str] = Field(None, description="系统级提示词")
    answer_style: AnswerStyle = Field(default=AnswerStyle.PROFESSIONAL, description="回答风格")
    tags: Optional[List[str]] = Field(default=[], description="标签")
    is_public: bool = Field(default=False, description="是否公开")
    avatar: Optional[str] = Field(None, description="头像URL")


class CustomRoleUpdateRequest(BaseModel):
    """更新自定义角色请求"""
    display_name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, min_length=10, max_length=500)
    specialties: Optional[List[str]] = Field(None, min_items=1, max_items=10)
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    answer_style: Optional[AnswerStyle] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    avatar: Optional[str] = None


class CustomRoleResponse(BaseModel):
    """自定义角色响应"""
    id: int
    name: str
    display_name: str
    description: str
    specialties: List[str]
    prompt_template: Optional[str]
    system_prompt: Optional[str]
    answer_style: Optional[str]
    tags: Optional[List[str]]
    is_public: bool
    avatar: Optional[str]
    owner_id: Optional[int]
    usage_count: int
    rating: float
    rating_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class RoleRatingRequest(BaseModel):
    """角色评分请求"""
    rating: int = Field(..., ge=1, le=5, description="评分1-5")
    comment: Optional[str] = Field(None, max_length=200, description="评论")
