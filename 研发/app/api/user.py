"""
用户管理API路由模块
功能：处理用户注册和登录认证
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.schemas import UserRegisterRequest, UserLoginRequest, UserResponse
from ..services.user_service import UserService

router = APIRouter(prefix="/users", tags=["用户"])


@router.post("/register", response_model=UserResponse)
async def register_user(
    request: UserRegisterRequest,  # 请求体：包含用户名、密码、邮箱等注册信息
    db: Session = Depends(get_db)  # 依赖注入：自动获取数据库会话
):
    service = UserService(db)  # 创建用户服务实例
    """
    用户注册接口
    功能：创建新用户账户
    流程：
    接收用户注册信息（用户名、密码、邮箱等）
    验证用户名/邮箱是否已存在
    对密码进行哈希加密（安全性）
    保存用户信息到数据库
    返回用户信息（不包含密码）

    异常处理：
    ValueError: 业务逻辑错误（如用户名已存在）→ 400
    Exception: 其他未知错误 → 500
    """
    try:
        user = service.create_user(request)  # 调用服务层创建用户，服务层会处理：密码加密、字段验证、数据库保存
        return user  # 自动序列化为 UserResponse 格式
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        # 业务逻辑错误（如：用户名已被注册、邮箱格式错误）
        # 返回400状态码，告诉客户端请求参数有问题
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        # 系统级错误（如：数据库连接失败、未知异常）
        # 返回500状态码，表示服务器内部错误


@router.post("/login", response_model=UserResponse)
async def login_user(
    request: UserLoginRequest,  # 请求体：包含用户名和密码
    db: Session = Depends(get_db)  # 依赖注入数据库会话
):
    service = UserService(db)
    """
    用户登录接口
    功能：验证用户身份并返回用户信息
    流程：
    根据用户名查找用户
    验证密码是否匹配（使用哈希比较）
    返回用户信息（可用于后续生成JWT token）
    注意：当前版本返回用户信息，生产环境应返回JWT token

    异常处理：
    认证失败（用户名不存在或密码错误）→ 401 Unauthorized
    """
    user = service.authenticate_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return user


# 📖 代码详细注释说明
#1. **用户注册流程详解**
# 注册请求示例
# {
#     "username": "lawyer_zhang",  # 用户名（必须唯一）
#     "password": "secure123",     # 明文密码（会被加密存储）
#     "email": "zhang@law.com",    # 邮箱（可选）
#     "phone": "13800138000"       # 手机号（可选）
# }
# # 内部处理流程
# UserService.create_user(request)
#     ├─ 验证用户名是否已存在
#     ├─ 验证邮箱格式（如果提供）
#     ├─ 密码哈希加密（bcrypt/sha256）
#     ├─ 创建User模型实例
#     ├─ 保存到数据库
#     └─ 返回UserResponse（不含password字段）
