import hashlib
from sqlalchemy.orm import Session
from typing import Optional

from ..models.database import User
from ..models.schemas import UserRegisterRequest, UserLoginRequest
from ..utils.logger import logger


class UserService:
    """用户服务"""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def create_user(self, request: UserRegisterRequest) -> User:
        existing = self.db.query(User).filter(User.username == request.username).first()
        if existing:
            raise ValueError("用户名已存在")

        email = request.email.strip() if request.email else None
        if email:
            same_email = self.db.query(User).filter(User.email == email).first()
            if same_email:
                raise ValueError("该邮箱已被使用")

        user = User(
            username=request.username,
            email=email,
            phone=request.phone or "",
            hashed_password=self._hash_password(request.password)
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        logger.info(f"Created user {user.username} id={user.id}")
        return user

    def authenticate_user(self, request: UserLoginRequest) -> Optional[User]:
        user = self.db.query(User).filter(User.username == request.username).first()
        if not user:
            return None
        if user.hashed_password != self._hash_password(request.password):
            return None
        return user
