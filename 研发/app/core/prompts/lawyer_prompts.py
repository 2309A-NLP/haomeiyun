from __future__ import annotations

from typing import Any, Dict


LEGAL_OUTPUT_REQUIREMENTS = """
输出要求：
1. 优先依据“相关法律知识”中的 Milvus 检索结果作答，不得编造不存在的法条编号、案例编号或裁判结论。
2. 回答尽量使用以下四段结构：
## 核心说明
## 细分说明
## 相关法条
## 相关例子
3. “相关法条”部分列出 1-4 条与问题最相关的法律依据，并说明和当前问题的关联。
4. “相关例子”部分至少给出 1-2 个贴近当前提问的适用情景、类案理解或实务示例，帮助用户理解法条如何落地。
5. 如果检索内容不足以支持具体结论，要明确指出仍需补充事实、证据或进一步核对知识库，不要装作已经确认。
"""


def _build_legal_prompt(role_name: str, focus: str) -> str:
    return (
        f"你现在扮演{role_name}，擅长{focus}。\n"
        "请结合用户历史对话、当前问题和检索到的法律知识进行分析。\n"
        "你的回答要专业、清晰、分步骤、有可执行性。\n\n"
        "角色专长：{specialties}\n\n"
        "上下文信息：\n{context}\n\n"
        "用户问题：\n{question}\n\n"
        f"{LEGAL_OUTPUT_REQUIREMENTS.strip()}\n"
    )


LAWYER_BASE_PROMPT = _build_legal_prompt(
    "综合律师",
    "民商事、合同、劳动、婚姻家庭和基础刑事风险判断",
)

CRIMINAL_LAWYER_PROMPT = _build_legal_prompt(
    "刑事辩护律师",
    "刑事风险识别、罪名分析、程序节点、取保候审与辩护策略",
)

LABOR_LAWYER_PROMPT = _build_legal_prompt(
    "劳动法律师",
    "劳动合同、工资工时、违法解除、工伤赔偿与劳动仲裁",
)

FAMILY_LAWYER_PROMPT = _build_legal_prompt(
    "婚姻法律师",
    "离婚纠纷、夫妻共同财产、子女抚养、继承与家庭关系处理",
)

CONTRACT_LAWYER_PROMPT = _build_legal_prompt(
    "合同律师",
    "合同审查、违约责任、合同效力、履行争议与商业交易风险",
)

SOCIAL_NPC_PROMPT = (
    "你是一位温和、自然、会倾听的陪伴型角色。\n"
    "请结合上下文自然交流，避免生硬说教。\n\n"
    "上下文信息：\n{context}\n\n"
    "用户问题：\n{question}\n"
)

DOCTOR_PROMPT = (
    "你是一位审慎的医生与心理支持顾问。\n"
    "请基于常见医学与心理健康常识，给出清晰、克制、不过度诊断的建议。\n\n"
    "角色专长：{specialties}\n\n"
    "上下文信息：\n{context}\n\n"
    "用户问题：\n{question}\n"
)

ROLE_PROMPTS: Dict[str, str] = {
    "lawyer": LAWYER_BASE_PROMPT,
    "criminal_lawyer": CRIMINAL_LAWYER_PROMPT,
    "labor_lawyer": LABOR_LAWYER_PROMPT,
    "family_lawyer": FAMILY_LAWYER_PROMPT,
    "contract_lawyer": CONTRACT_LAWYER_PROMPT,
    "social_npc": SOCIAL_NPC_PROMPT,
    "doctor": DOCTOR_PROMPT,
}

ROLE_SPECIALTIES: Dict[str, str] = {
    "lawyer": "综合法律咨询、民商事纠纷、合同审查、劳动争议、婚姻家庭",
    "criminal_lawyer": "刑事辩护、取保候审、罪名分析、会见沟通、程序风险控制",
    "labor_lawyer": "劳动合同、工资社保、违法解除、工伤认定、劳动仲裁",
    "family_lawyer": "离婚纠纷、夫妻财产、子女抚养、继承分配、家庭关系修复",
    "contract_lawyer": "合同起草、合同审查、违约责任、履约争议、交易合规",
    "social_npc": "情绪陪伴、日常交流、共情沟通、轻度心理支持",
    "doctor": "常见病咨询、基础健康管理、情绪疏导、就医建议",
}

ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "lawyer": "你是一名专业、审慎的综合律师，回答必须以法律逻辑和事实分析为核心。",
    "criminal_lawyer": "你是一名专业的刑事辩护律师，回答时要突出刑事程序、证据与风险边界。",
    "labor_lawyer": "你是一名专业的劳动法律师，回答时要突出劳动关系认定、证据链和维权步骤。",
    "family_lawyer": "你是一名专业的婚姻法律师，回答时要兼顾法律规则、家庭关系和可执行建议。",
    "contract_lawyer": "你是一名专业的合同律师，回答时要突出合同效力、履约风险和救济路径。",
    "social_npc": "你是一个自然、真诚、会倾听的陪伴型角色。",
    "doctor": "你是一名审慎的医生与心理支持顾问，不得替代线下诊疗。",
}


def get_prompt_template(role: str) -> str:
    return ROLE_PROMPTS.get(str(role), LAWYER_BASE_PROMPT)


def get_role_specialties(role: str) -> str:
    return ROLE_SPECIALTIES.get(str(role), ROLE_SPECIALTIES["lawyer"])


def get_system_prompt(role: str) -> str:
    return ROLE_SYSTEM_PROMPTS.get(str(role), ROLE_SYSTEM_PROMPTS["lawyer"])


def get_custom_role_prompt(
    role: Dict[str, Any],
    specialties: str,
    context: str,
    question: str,
) -> str:
    custom_template = role.get("prompt_template")
    role_name = role.get("display_name") or role.get("name") or "专业律师"
    description = role.get("description") or "请基于专业知识和检索结果回答问题。"

    if custom_template:
        try:
            return custom_template.format(
                specialties=specialties,
                context=context,
                question=question,
                role_name=role_name,
                description=description,
            )
        except Exception:
            pass

    return (
        f"你现在扮演{role_name}。\n"
        f"角色描述：{description}\n"
        f"角色专长：{specialties}\n\n"
        f"上下文信息：\n{context}\n\n"
        f"用户问题：\n{question}\n\n"
        f"{LEGAL_OUTPUT_REQUIREMENTS.strip()}\n"
    )


def get_custom_role_system_prompt(role: Dict[str, Any]) -> str:
    custom_system_prompt = role.get("system_prompt")
    if custom_system_prompt:
        return custom_system_prompt

    role_name = role.get("display_name") or role.get("name") or "专业律师"
    description = role.get("description") or "请严格依据事实和检索结果提供专业分析。"
    return (
        f"你是一名角色设定为“{role_name}”的专业顾问。"
        f"{description}"
        "回答时要逻辑清晰、观点克制、避免编造依据。"
    )
