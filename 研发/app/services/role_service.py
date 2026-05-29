# app/services/role_service.py
from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from ..models.database import Role
from ..models.schemas import RoleCreateRequest
from ..core.prompts import get_prompt_template, get_role_specialties
from ..utils.logger import logger


class RoleService:
    """角色管理服务"""

    def __init__(self, db: Session):
        self.db = db

    def create_role(self, request: RoleCreateRequest) -> Role:
        """创建角色"""
        # 如果提供了模板就使用，否则根据name获取默认模板
        prompt_template = request.prompt_template or get_prompt_template(request.name)

        db_role = Role(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            specialties=request.specialties,
            prompt_template=prompt_template
        )

        self.db.add(db_role)
        self.db.commit()
        self.db.refresh(db_role)

        logger.info(f"Created role: {request.name}")
        return db_role

    def get_role(self, role_id: int) -> Optional[Role]:
        """获取角色"""
        return self.db.query(Role).filter(Role.id == role_id).first()

    def get_role_by_name(self, name: str) -> Optional[Role]:
        """通过名称获取角色"""
        return self.db.query(Role).filter(Role.name == name).first()

    def list_roles(self, skip: int = 0, limit: int = 100) -> List[Role]:
        """列出所有角色"""
        return self.db.query(Role).filter(Role.is_active == True).offset(skip).limit(limit).all()

    def init_default_roles(self):
        """初始化默认角色"""
        default_roles = [
            {
                "name": "social_npc",
                "display_name": "社交NPC",
                "description": "提供情感陪伴与日常交流的虚拟朋友",
                "specialties": ["情感陪伴", "日常交流", "心理支持"]
            },
            {
                "name": "doctor",
                "display_name": "医生/心理医生",
                "description": "提供医疗健康咨询与心理支持建议",
                "specialties": ["高血压", "糖尿病", "心理咨询", "健康管理"]
            },
            {
                "name": "lawyer",
                "display_name": "综合律师",
                "description": "精通各领域的综合法律顾问",
                "specialties": ["民法", "刑法", "商法", "劳动法"]
            },
            {
                "name": "criminal_lawyer",
                "display_name": "刑事辩护律师",
                "description": "专注刑事辩护，保护当事人合法权益",
                "specialties": ["职务犯罪", "经济犯罪", "暴力犯罪", "毒品犯罪"]
            },
            {
                "name": "labor_lawyer",
                "display_name": "劳动法律师",
                "description": "解决劳动争议，维护劳动者权益",
                "specialties": ["劳动合同", "工伤赔偿", "经济补偿", "竞业限制"]
            },
            {
                "name": "family_lawyer",
                "display_name": "婚姻家事律师",
                "description": "处理婚姻家庭纠纷",
                "specialties": ["离婚诉讼", "财产分割", "抚养权", "遗产继承"]
            },
            {
                "name": "contract_lawyer",
                "display_name": "合同律师",
                "description": "专业合同审查与起草",
                "specialties": ["合同审查", "合同纠纷", "商事谈判", "企业合规"]
            }
        ]

        for role_data in default_roles:
            existing = self.get_role_by_name(role_data["name"])
            if not existing:
                request = RoleCreateRequest(**role_data)
                self.create_role(request)
                logger.info(f"Initialized default role: {role_data['name']}")
