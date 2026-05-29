# app/api/role.py
"""
角色管理API路由模块
功能：管理系统默认角色和用户自定义角色的CRUD操作
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from ..models.database import get_db, User
from ..models.schemas import (
    RoleCreateRequest,
    RoleResponse,
    CustomRoleCreateRequest,
    CustomRoleUpdateRequest,
    CustomRoleResponse,
    RoleRatingRequest
)
from ..services.role_service import RoleService
from ..services.custom_role_service import CustomRoleService

router = APIRouter(prefix="/roles", tags=["角色"])
# 创建路由实例，所有角色相关接口的前缀为 /roles
# tags=["角色"] 用于在API文档中分组

# ============ 系统默认角色接口 ============

@router.post("/init-defaults")
async def init_default_roles(db: Session = Depends(get_db)):
    """初始化默认法律角色"""
    """
    初始化默认法律角色
    用途：系统首次部署时，预置法律领域的标准角色
    调用时机：系统初始化或数据迁移时
    """
    service = RoleService(db)  # 创建角色服务实例
    service.init_default_roles()   # 执行初始化（如果角色已存在则跳过）
    return {"message": "Default roles initialized"}


@router.get("/system", response_model=list[RoleResponse])
async def list_system_roles(
    skip: int = 0,  # 分页：跳过的记录数（默认从第0条开始）
    limit: int = 100,  # 分页：每页最大记录数（默认100条）
    db: Session = Depends(get_db)
):
    """
    列出系统默认角色
    用途：获取所有预定义的法律角色列表
    返回：角色信息列表（包含id、名称、描述、技能等）
    """
    service = RoleService(db)
    return service.list_roles(skip, limit)  # 支持分页查询


@router.get("/system/{role_id}", response_model=RoleResponse)
async def get_system_role(
    role_id: int,  # 角色ID（路径参数）
    db: Session = Depends(get_db)
):
    """
    获取系统默认角色详情
    用途：查看特定角色的完整信息（包括技能树、示例对话等）
    """
    service = RoleService(db)
    role = service.get_role(role_id)
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
        # 角色不存在时返回404
    return role


# ============ 自定义角色接口 ============
# 注意：以下接口需要用户认证，暂时使用user_id参数模拟

@router.post("/custom", response_model=CustomRoleResponse)
async def create_custom_role(
    request: CustomRoleCreateRequest,   # 请求体：包含角色名称、描述、技能配置等
    user_id: int = Query(..., description="用户ID"),  # 查询参数：创建者ID, 必须提供用户ID
    db: Session = Depends(get_db)
):
    """
    创建自定义角色
    流程：
        1.验证角色名称是否重复
        2.创建自定义角色记录
        3.关联创建者信息
    """
    service = CustomRoleService(db)
    try:
        role = service.create_custom_role(request, user_id) # 调用服务层创建
        return role
    except ValueError as e:  # 捕获业务逻辑错误（如名称已存在）
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/custom/my", response_model=list[CustomRoleResponse])
async def get_my_custom_roles(
    user_id: int = Query(..., description="用户ID"),  # 当前用户ID
    skip: int = 0,  # 分页偏移,偏移量（用于滚动加载）
    limit: int = 20,  # 每页数量（默认20条，适合移动端）,移动端友好（小批量）
    db: Session = Depends(get_db)
):
    """
    获取我的自定义角色列表
    用途：用户查看自己创建的所有角色
    特点：只返回当前用户创建的角色，不包括系统默认角色
    """
    service = CustomRoleService(db)
    return service.get_user_roles(user_id, skip, limit)


@router.get("/custom/public", response_model=list[CustomRoleResponse])
async def get_public_custom_roles(
    skip: int = 0,
    limit: int = 20,
    tag: Optional[str] = Query(None, description="按标签筛选"),  # 可选：按标签过滤（如"合同纠纷"）
    sort_by: Optional[str] = Query("created_at", description="排序字段: created_at, rating, usage"),
    db: Session = Depends(get_db)
):
    """
    获取公开的自定义角色列表
    用途：用户浏览其他用户分享的角色
        特点：
        只返回 is_public=True 的角色
        支持标签筛选和多种排序方式
        可用于角色市场功能
    """
    service = CustomRoleService(db)
    return service.get_public_roles(skip, limit, tag, sort_by)


@router.get("/custom/{role_id}", response_model=CustomRoleResponse)
async def get_custom_role(
    role_id: int,  # 角色ID
    user_id: Optional[int] = Query(None, description="用户ID（用于权限检查）"),  # 可选，用于公开角色
    db: Session = Depends(get_db)
):
    """
    获取自定义角色详情
    权限控制：
        公开角色：所有人可见
        私有角色：只有创建者可见
    """
    service = CustomRoleService(db)
    role = service.get_custom_role(role_id, user_id)  # user_id用于权限验证
    if not role:
        raise HTTPException(status_code=404, detail="Role not found or no permission")
    return role


@router.put("/custom/{role_id}", response_model=CustomRoleResponse)
async def update_custom_role(
    role_id: int,
    request: CustomRoleUpdateRequest,  # 更新请求体（所有字段可选）
    user_id: int = Query(..., description="用户ID"),  # 必须提供，验证是否为创建者
    db: Session = Depends(get_db)
):
    """
    更新自定义角色
    权限：只有创建者可以更新自己的角色
    可更新字段：名称、描述、技能配置、是否公开等
    """
    service = CustomRoleService(db)
    role = service.update_custom_role(role_id, request, user_id)
    if not role:
        raise HTTPException(status_code=404, detail="Role not found or no permission")
    return role


@router.delete("/custom/{role_id}")
async def delete_custom_role(
    role_id: int,
    user_id: int = Query(..., description="用户ID"),  # 必须提供，验证是否为创建者
    db: Session = Depends(get_db)
):
    """
    删除自定义角色
    权限：只有创建者可以删除自己的角色
    注意：删除操作是物理删除（永久删除）
    """
    service = CustomRoleService(db)
    success = service.delete_custom_role(role_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Role not found or no permission")
    return {"message": "Role deleted successfully"}


@router.post("/custom/{role_id}/rate")
async def rate_custom_role(
    role_id: int,
    request: RoleRatingRequest,  # 包含评分(1-5)和评论
    user_id: int = Query(..., description="用户ID"),   # 评分的用户
    db: Session = Depends(get_db)
):
    """
    评分自定义角色
    用途：用户对其他用户创建的角色进行评价
        功能：
        记录用户的评分（1-5星）
        自动更新角色的平均评分
        可选添加文字评论
    """
    service = CustomRoleService(db)
    success = service.rate_role(role_id, request.rating, request.comment, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Role not found")
    return {"message": "Rating submitted successfully"}


@router.post("/custom/{role_id}/use")
async def use_custom_role(
    role_id: int,
    user_id: int = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """
    使用自定义角色（增加使用计数）
    用途：追踪角色的流行度
    调用时机：用户开始与该角色对话时
    效果：role.usage_count += 1（用于"最常用角色"排序）
    """
    service = CustomRoleService(db)
    success = service.increment_usage(role_id)  # 只增加计数，不验证用户
    if not success:
        raise HTTPException(status_code=404, detail="Role not found")
    return {"message": "Usage recorded"}

