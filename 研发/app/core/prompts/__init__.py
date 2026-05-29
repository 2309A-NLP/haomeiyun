# app/core/prompts/__init__.py
from .lawyer_prompts import (
    LAWYER_BASE_PROMPT,
    CRIMINAL_LAWYER_PROMPT,
    LABOR_LAWYER_PROMPT,
    FAMILY_LAWYER_PROMPT,
    CONTRACT_LAWYER_PROMPT,
    SOCIAL_NPC_PROMPT,
    DOCTOR_PROMPT,
    ROLE_PROMPTS,
    ROLE_SPECIALTIES,
    get_prompt_template,
    get_role_specialties,
    get_system_prompt,
    get_custom_role_prompt,
    get_custom_role_system_prompt,
)

__all__ = [
    "LAWYER_BASE_PROMPT",
    "CRIMINAL_LAWYER_PROMPT",
    "LABOR_LAWYER_PROMPT",
    "FAMILY_LAWYER_PROMPT",
    "CONTRACT_LAWYER_PROMPT",
    "SOCIAL_NPC_PROMPT",
    "DOCTOR_PROMPT",
    "ROLE_PROMPTS",
    "ROLE_SPECIALTIES",
    "get_prompt_template",
    "get_role_specialties",
    "get_system_prompt",
    "get_custom_role_prompt",
    "get_custom_role_system_prompt",
]
