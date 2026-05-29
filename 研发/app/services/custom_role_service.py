# app/services/custom_role_service.py
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models.database import Role
from ..models.schemas import (
    CustomRoleCreateRequest,
    CustomRoleUpdateRequest,
)
from ..utils.logger import logger


class CustomRoleService:
    """自定义角色服务"""

    def __init__(self, db: Session):
        self.db = db

    def create_custom_role(
        self, 
        request: CustomRoleCreateRequest, 
        user_id: int
    ) -> Role:
        """创建自定义角色"""
        # 检查名称是否已存在
        existing = self.db.query(Role).filter(
            and_(
                Role.name == request.name,
                Role.owner_id == user_id
            )
        ).first()

        if existing:
            raise ValueError(f"角色名称 '{request.name}' 已存在")

        # 生成系统提示词（如果未提供）
        system_prompt = request.system_prompt
        if not system_prompt:
            system_prompt = self._generate_system_prompt(
                request.answer_style.value,
                request.specialties
            )

        db_role = Role(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            specialties=request.specialties,
            prompt_template=request.prompt_template,
            system_prompt=system_prompt,
            answer_style=request.answer_style.value,
            tags=request.tags or [],
            is_public=request.is_public,
            owner_id=user_id,
            avatar=request.avatar
        )

        self.db.add(db_role)
        self.db.commit()
        self.db.refresh(db_role)

        logger.info(f"用户 {user_id} 创建了自定义角色: {request.name}")
        return db_role

    def get_user_roles(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Role]:
        """获取用户的自定义角色列表"""
        return (
            self.db.query(Role)
            .filter(Role.owner_id == user_id)
            .order_by(desc(Role.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_public_roles(
        self,
        skip: int = 0,
        limit: int = 20,
        tag: Optional[str] = None,
        sort_by: str = "created_at"
    ) -> List[Role]:
        """获取公开的自定义角色列表"""
        query = self.db.query(Role).filter(Role.is_public == True)

        # 标签筛选
        if tag:
            query = query.filter(Role.tags.contains([tag]))

        # 排序
        if sort_by == "rating":
            query = query.order_by(desc(Role.rating))
        elif sort_by == "usage":
            query = query.order_by(desc(Role.usage_count))
        else:  # created_at
            query = query.order_by(desc(Role.created_at))

        return query.offset(skip).limit(limit).all()

    def get_custom_role(
        self, 
        role_id: int, 
        user_id: Optional[int] = None
    ) -> Optional[Role]:
        """获取自定义角色详情"""
        role = self.db.query(Role).filter(Role.id == role_id).first()

        if not role:
            return None

        # 检查访问权限
        if role.owner_id != user_id and not role.is_public:
            return None

        return role

    def update_custom_role(
        self,
        role_id: int,
        request: CustomRoleUpdateRequest,
        user_id: int
    ) -> Optional[Role]:
        """更新自定义角色"""
        role = self.db.query(Role).filter(
            and_(
                Role.id == role_id,
                Role.owner_id == user_id
            )
        ).first()

        if not role:
            return None

        # 更新字段
        update_data = request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if field == "answer_style" and value:
                setattr(role, field, value.value)
            else:
                setattr(role, field, value)

        self.db.commit()
        self.db.refresh(role)

        logger.info(f"用户 {user_id} 更新了自定义角色 {role_id}")
        return role

    def delete_custom_role(
        self, 
        role_id: int, 
        user_id: int
    ) -> bool:
        """删除自定义角色"""
        role = self.db.query(Role).filter(
            and_(
                Role.id == role_id,
                Role.owner_id == user_id
            )
        ).first()

        if not role:
            return False

        self.db.delete(role)
        self.db.commit()

        logger.info(f"用户 {user_id} 删除了自定义角色 {role_id}")
        return True

    def increment_usage(self, role_id: int) -> bool:
        """增加角色使用次数"""
        role = self.db.query(Role).filter(Role.id == role_id).first()
        if not role:
            return False

        role.usage_count += 1
        self.db.commit()
        return True

    def rate_role(
        self,
        role_id: int,
        rating: int,
        comment: Optional[str],
        user_id: int
    ) -> bool:
        """评分角色"""
        role = self.db.query(Role).filter(Role.id == role_id).first()
        if not role:
            return False

        # 计算新的平均评分
        total_score = role.rating * role.rating_count + rating
        role.rating_count += 1
        role.rating = total_score / role.rating_count

        self.db.commit()

        logger.info(f"用户 {user_id} 对角色 {role_id} 评分: {rating}")
        return True

    def _generate_system_prompt(
        self,
        answer_style: str,
        specialties: List[str]
    ) -> str:
        """根据回答风格和专业领域生成系统提示词"""
        style_prompts = {
            "formal": "请使用正式、严谨的语言风格，保持专业性和权威性。",
            "friendly": "请使用亲切、友好的语言风格，让用户感到温暖和被理解。",
            "professional": "请使用专业、清晰的语言风格，准确传达法律信息。",
            "concise": "请使用简洁明了的语言风格，直接回答核心问题，避免冗余。",
            "detailed": "请提供详细全面的回答，包括相关背景、法律依据和注意事项。",
        }

        base_prompt = f"""你是一位专业的法律咨询顾问，专业领域包括：{', '.join(specialties)}。

{style_prompts.get(answer_style, style_prompts['professional'])}

回答要求：
1. 优先依据明确的法律规范进行分析
2. 准确识别案件事实中的法律关系、权利义务与争议焦点
3. 在结论之外补充风险提示、证据要求与可执行的处理建议
4. 使用规范的法律术语，保持正式、严谨、审慎的表达方式
5. 如果当前问题与前文有关，必须结合历史对话持续分析，不得忽略用户此前已经说明的事实、诉求和时间线

输出结构要求：
优先使用清晰的 Markdown 层级结构输出，必须按照以下三个部分组织答案：
- 核心说明：先用通俗中文概括结论，让不懂法律的人也能看懂
- 细分说明：解释关键概念、判断标准、常见情形和例外
- 法律依据：必须放在最后"""

        return base_prompt
