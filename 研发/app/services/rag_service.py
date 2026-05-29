'''
这是一个法律咨询AI助手后端服务，通过智能判断问题类型并结合RAG检索法律知识库，为不同法律角色提供带条文引用和风险提示的法律问答。
'''
from __future__ import annotations
import re
import time
from typing import AsyncGenerator, List, Optional, Tuple
from ..core.config import settings
from ..core.prompts import (
    get_prompt_template,
    get_role_specialties,
    get_system_prompt,
    get_custom_role_prompt,
    get_custom_role_system_prompt,
)
from ..models.database import Role
from ..models.schemas import ChatRequest, ChatResponse, Citation
from ..rag.pipeline import LegalRAGPipeline
from .llm_service import LLMService
from .memory_service import MemoryService
from ..utils.logger import logger

# 系统预定义的法律角色ID集合
LEGAL_ROLE_IDS = {
    "lawyer",           # 通用律师
    "criminal_lawyer",  # 刑事律师
    "labor_lawyer",     # 劳动律师
    "family_lawyer",    # 家事律师
    "contract_lawyer",  # 合同律师
}

# 字段映射（Field Mapping） 字典，常见于编程（尤其是 Python）中，用于将一种命名体系（如角色名）转换为另一种体系（如法律领域代码）
ROLE_LEGAL_FIELD_MAP = {
    "criminal_lawyer": "criminal",  # 刑法：涉及犯罪、辩护、公诉、量刑等。
    "labor_lawyer": "labor",        # 劳动法：劳动合同、工伤、社保、劳动争议仲裁等。
    "family_lawyer": "family",      # 家事法/婚姻家庭法：离婚、抚养权、继承、婚前协议等。
    "contract_lawyer": "contract",  # 合同法：合同的订立、履行、违约、解除、纠纷处理。
}
'''
ROLE_LEGAL_FIELD_MAP：常量名，表示“角色 ↔ 法律领域 映射表”。
键（key）："criminal_lawyer" 等，表示具体的律师角色。
值（value）："criminal" 等，表示对应的法律领域代码。

为什么要做这种映射？
在实际系统（如法律咨询平台、案件管理系统）中，不同模块或接口可能使用不同命名习惯：
前端/用户侧：倾向使用 "criminal_lawyer"（易懂、含角色后缀）
后端/数据库/算法侧：倾向使用短代码 "criminal"（节省存储、便于枚举）
'''


'''
总结一句话
你这段代码不是一个随意的配置，而是一个完整的“法律咨询 AI 的决策树配置文件”——它决定了系统“听懂”用户的话之后，应该做什么级别的处理、如何回应、以及如何保证安全和稳定。
'''
# 核心目的：
# 快速过滤或识别用户输入中的“非实质性内容”，避免将其送入复杂的问题理解/法律咨询分析模块。
GREETING_PATTERNS = (  # “问候模式集合”
    "你好",
    "您好",
    "在吗",
    "谢谢",
    "多谢",
    "辛苦了",
    "再见",
)

# 这是一个简单解释/基础信息请求模式集合，用于识别用户希望获得概念性、流程性、时间性等相对基础的法律信息，而非复杂的个案分析。
SIMPLE_EXPLANATION_PATTERNS = (  # 保存的数据类型是元组，存储固定的关键词/短语模式
    "是什么",
    "什么意思",
    "啥意思",
    "定义",
    "区别",
    "有什么区别",
    "怎么理解",
    "流程",
    "多久",
    "几天",
    "几年",
)

SIMPLE_JUDGMENT_PATTERNS = (  # 简单的法律判断  --- 直接给出法条结论
    "犯法吗",
    "违法么",
    "违法吗",
    "合法吗",
    "有效吗",
    "可以吗",
    "能报警吗",
    "能起诉吗",
    "要坐牢吗",
    "会判刑吗",
)

COMPLEX_FACT_PATTERNS = (  # 复杂事实型案件  --- 需要分析+建议行动
    "我想",
    "我们",
    "对方",
    "公司",
    "单位",
    "已经",
    "现在",
    "因为",
    "之前",
    "之后",
    "证据",
    "材料",
    "起诉",
    "仲裁",
    "离婚",
    "赔偿",
    "合同",
    "工伤",
    "拘留",
    "逮捕",
    "报警",
    "执行",
)

# 法律免责声明
DEFAULT_DISCLAIMER = "本回答仅供参考，不构成正式法律意见。涉及重大权益事项时，建议及时咨询执业律师。"
GENERATION_FAILED_REPLY = "抱歉，这次回答生成失败，请稍后重试。"  # 降级兜底
# 遇到LLM API超时、LLM返回空/乱码、内容审核拦截、系统资源不足触发
# 设计原则
# 用户体验兜底：不能什么都不返回，也不能返回技术报错（如 KeyError、Timeout）
# 引导重试：明确告诉用户“稍后重试”，而不是“系统错误”
# 不暴露内部细节：避免用户看到堆栈信息或敏感配置

# 单例模式，用于确保 LegalChatService 类在整个程序运行期间只有一个实例。
class LegalChatService:
    _instance = None  # 类变量，用于存储唯一的实例

    def __new__(cls, *args, **kwargs):  # __new__ 在 __init__ 之前被调用，负责创建实例
        if cls._instance is None:       # 判断是否已经存在实例
            cls._instance = super().__new__(cls)  # 调用父类 object 的 __new__ 创建实例
        return cls._instance            # 返回唯一的实例
# 核心思想：昂贵的初始化只做一次，但允许轻量级的属性更新（如更换数据库连接）在后续调用中生效。
    # 单例模式中防止重复初始化的核心技巧
    def __init__(self, db=None):  # 查一下这个实例有没有‘已初始化’的标签
        if getattr(self, "_initialized", False):  # 如果已经有标签了（说明之前初始化过）
            self.db = db   # 更新数据库连接
            return

        self.rag = LegalRAGPipeline()  # 创建 RAG 检索增强生成管道，负责检索法律知识库、找相关法条和案例
        self.llm = LLMService()  # 创建大语言模型服务，负责生成回答、理解用户问题
        self.memory = MemoryService()  # 创建记忆/上下文管理服务，负责记住对话历史、维护上下文
        self.db = db  # 设置数据库连接，用于查询用户信息、保存对话记录
        self._initialized = True  # 标记为已初始化，标记已完成初始化，防止重复执行
#-------------------------- 一、非流式输出结果 ---------------------------------------------------------
    # 异步方法chat，这就是一个聊天后端API的核心处理函数，负责接收消息、管理对话、调用AI、返回回复，并记录性能日志。
    async def chat(self, request: ChatRequest) -> ChatResponse:  # 定义了一个异步方法，接收 ChatRequest 类型的请求，返回 ChatResponse 响应
        start_time = time.time()   # 记录请求开始处理的时间戳（用于计算响应耗时）
        session_id = request.session_id or f"{request.user_id}_{int(start_time)}"
        # 设置会话ID：如果请求中已有 session_id 就用它，否则用 {用户ID}_{时间戳} 格式生成一个新的
        # eg："user123_1699123456"

        response = await self._chat_impl(request, session_id, start_time)
        # 调用实际的聊天处理逻辑 _chat_impl（异步等待结果），传递请求、会话ID和开始时间
        self._log_soft_target(response.response_time, request.user_id, stream_mode=False)  # 日志
        # 记录日志或监控指标（可能是性能监控）
        # response.response_time：接口响应耗时
        # stream_mode=False：表示非流式模式
        return response   # 返回处理后的聊天响应
    '''
    非流式输出的特点：
        优点：逻辑简单，容易处理
        缺点：用户需要等待全部生成完毕才能看到内容（对于长回复可能较慢）
    '''
    # 流式输出
    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        # async def - 异步方法定义， -> AsyncGenerator[str, None]：返回一个异步生成器，每次yield一个字符串，不返回最终值
        start_time = time.time()  # 记录请求开始时间，用于性能监控
        session_id = request.session_id or f"{request.user_id}_{int(start_time)}"  # 会话ID管理
        # 如果请求带了session_id就用它（保持对话连续性）
        # 否则用用户ID_时间戳格式生成新会话
        simple_mode = self._should_reply_directly(request.message)  # 判断是不是使用了简单模式
        # 可能检查：问候语（"你好"）、简单问题（"今天星期几"）等
        # 如果返回True，可能不需要调用RAG或复杂处理
        citations: List[Citation] = []  # 引用列表，用于记录信息来源，例如：AI回答中引用的文档，网页，知识库条目
        # 格式可能：[{"source": "文档A.pdf", "page": 10, "content": "..."}]
        full_response = ""  # 存储完整响应（）,用途：最后可能用于日志记录、缓存、或后续处理
        final_response = "" # 存储最终响应,可能用于：去除思考过程、只保留最终答案等
        rag_used = False  # RAG（检索增强生成）使用标志,FALSE未使用RAG（简单模式），True，使用了知识库检索

        # 获取对话上下文，用于维持多轮对话的连贯性
        try:
            context = self.memory.get_conversation_context(
                request.user_id, # 用户标识
                session_id,      # 会话标识
                max_turns=8,     # 最多获取8轮对话 ，每一轮 = 用户问 + AI答
                # 关于 max_turns=8
                    # 限制8轮：避免上下文过长（节省token、降低成本、提高速度）
                    # 通常保留最近N轮，早期的对话会被丢弃
                    # 8轮是个平衡点：既保持连贯性，又不会超长
            )
            # 作用：从内存/存储中检索该用户在当前会话中的历史对话记录。让AI记住"用户刚才说的话"，实现真正的多轮对话能力

            # 根据问题，从知识库中检索相关文档
            retrieved_docs, rag_used = await self._retrieve_docs(request, simple_mode)
            # request - 用户请求对象，提供查询内容（用户问什么），simple_mode - 简单模式标志

            # 布尔值，表示是否处于简单模式：
            # simple_mode	含义	                    检索行为
            # True	        简单模式（如问候、闲聊）	不检索，直接返回 [], False
            # False	        正常模式（需要知识）	    执行检索，返回相关文档

            # 构建发送给AI的完整提示词，把所有相关信息整合成一个结构化的prompt
            prompt = self._build_prompt(
                request=request,  # 用户原始请求
                context=context,  # 历史对话上下文
                retrieved_docs=retrieved_docs,  # 检索到的相关文档
                simple_mode=simple_mode,  # 简单模式标志
                rag_used=rag_used,  # 是否使用了RAG
            )

            # 调用LLM（大语言模型）生成流式响应
            stream = self.llm.generate_stream(
                prompt=prompt,         # 用户提示词
                system_prompt=self._build_system_prompt(  # 系统提示词
                    request.role_id,   # 可能是 "123" 或 123，指定AI应该扮演什么角色
                    request.user_id,   # 如 "user_12345"，获取用户的自定义角色（如果是用户自己创建的）
                    simple_mode=simple_mode, # True --- 简单模式 ，False --- 正常回答模式
                    rag_used=rag_used,    # 是否使用了RAG使用标志
                ),
                temperature=0.3,              # 温度参数（控制创造性）
                max_tokens=self._resolve_max_tokens(simple_mode),   # 最大生成token数
            )

            # 处理LLM流式输出的核心循环，负责对每个输出块进行加工、累积和转发
            # 实现AI"打字机效果"的代码：边生成边输出，边输出边累积。
            # 把AI生成的内容，一边加工一边实时地逐块返回给用户，同时把所有块拼成完整的回答保存起来。
            async for chunk in stream:  # 异步生成器：遍历LLM生成的每个流式数据块
                # 1. 加工处理当前块，把这个片段加工一下（比如去掉没用的符号、格式化一下）
                polished_chunk = self._polish_stream_chunk(chunk, simple_mode=simple_mode)
                # 2. 过滤空块，如果加工后是空的，就跳过这次（比如AI产生了一个空白或特殊标记）
                if not polished_chunk:
                    continue
                # 3. 累积完整响应，把这个片段拼到完整回答里（慢慢攒成完整的一句话）
                full_response += polished_chunk
                # 4. 实时转发给调用方，把这个片段实时返回给用户（用户马上就能看到）
                yield polished_chunk

            # 在处理完AI生成的完整回复后，进行收尾工作：构建引用来源和最终格式化
            citations = [] if (simple_mode or not rag_used) else self._build_citations(retrieved_docs)
            # 什么时候 citations = []（空列表）？
            # 条件	                   说明	                原因
            # simple_mode = True	   简单模式（问候/闲聊）	没有检索文档，不需要引用
            # rag_used = False	       未使用RAG	            没有参考资料，不需要引用
            # 其他情况	               正常问答+使用了RAG	    有检索文档，构建引用
            final_response = self._finalize_response_text(
                request=request,  # 原始请求
                response_text=full_response,  # AI生成的完整回答
                citations=citations,  # 引用来源
                simple_mode=simple_mode,  # 是否简单模式
            )
        except Exception as exc:  # 捕获任何异常
            logger.error("Streaming chat failed for user %s: %s", request.user_id, exc)
            # 记录错误日志，包含用户ID和异常详情，方便排查问题
            citations = []  # 清空引用列表（异常情况下没有有效的引用来源）
            final_response = self._polish_model_output(full_response) if full_response.strip() else GENERATION_FAILED_REPLY
            # 决定最终返回什么内容：
            # 如果 full_response 已经有部分内容（非空），用 _polish_model_output 加工后返回
            # 如果 full_response 是空的，返回预设的GENERATION_FAILED_REPLY的失败回复
            if not full_response.strip():
                yield final_response
            # 如果完全没有生成任何内容，直接 yield 返回错误提示给用户

        # 保存对话记录到数据库
        await self._save_conversation(
            request=request,  # 用户请求对象（包含用户ID、消息内容等）
            session_id=session_id,  # 会话ID（用于关联同一会话的多轮对话）
            response_text=final_response,  # AI生成的最终回复文本
            citations=citations,  # 引用来源列表（RAG检索到的文档引用）
        )
        # 增加自定义角色的使用次数统计
        self._increment_custom_role_usage(
            request.role_id,  # 角色ID（如果是自定义角色就统计，系统角色不统计）
            request.user_id  # 用户ID（用于权限验证，只有角色创建者或公开角色才统计）
        )

        # 记录性能监控日志（用于分析响应时间和系统负载）
        self._log_soft_target(
            time.time() - start_time,  # 响应耗时（当前时间 - 开始时间）
            request.user_id,  # 用户ID（用于追踪特定用户的性能）
            stream_mode=True  # 流式模式标志（True表示这是流式输出）
        )
    # 非流式聊天实现的核心逻辑
    # chat 方法调用的实际实现函数，负责处理非流式聊天的核心业务逻辑
    async def _chat_impl(
        self,
        request: ChatRequest,  # 用户请求对象
        session_id: str,  # 会话ID（标识对话会话）
        start_time: float,  # 请求开始时间戳（用于性能计算）
    ) -> ChatResponse:  # 返回完整的聊天响应对象
        # _chat_impl - 下划线表示私有方法，impl 表示"实现"（implementation）
        # 这是实际干活的方法，chat 方法只是包装器

        # 准备阶段的两步操作，用于确定回复策略和获取对话历史
        # 判断是否应该使用简单模式直接回复（例如：问候、简单闲聊等无需复杂推理的场景）
        simple_mode = self._should_reply_directly(request.message)

        # 从记忆中获取对话上下文
        context = self.memory.get_conversation_context(
            request.user_id,  # 用户ID，用于区分不同用户的对话历史
            session_id,  # 会话ID，用于区分同一用户的不同会话
            max_turns=8,  # 最大轮数，只获取最近的8轮对话（每轮包含用户提问和助手回复）
        )

        # 根据请求内容和简单模式标识，检索相关文档（RAG检索）
        # 返回值：
        #   - retrieved_docs: 检索到的相关文档列表
        #   - rag_used: 是否实际使用了RAG检索（可能因simple_mode等原因跳过检索）
        retrieved_docs, rag_used = await self._retrieve_docs(request, simple_mode)

        # 构建发送给LLM的提示词（prompt）
        # 整合了对话上下文、检索到的文档、以及模式控制等信息
        prompt = self._build_prompt(
            request=request,  # 原始请求对象，包含用户消息、参数等
            context=context,  # 对话历史上下文（最多8轮）
            retrieved_docs=retrieved_docs,  # RAG检索到的相关文档，用于增强回答
            simple_mode=simple_mode,  # 是否简单模式标记，影响prompt的复杂程度
            rag_used=rag_used,  # 是否实际使用了RAG，用于提示模型是否参考文档
        )
        response_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self._build_system_prompt(
                request.role_id,
                request.user_id,
                simple_mode=simple_mode,
                rag_used=rag_used,
            ),
            temperature=0.3,
            max_tokens=self._resolve_max_tokens(simple_mode),
        )

        citations = [] if (simple_mode or not rag_used) else self._build_citations(retrieved_docs)
        response_text = self._finalize_response_text(
            request=request,
            response_text=response_text,
            citations=citations,
            simple_mode=simple_mode,
        )

        await self._save_conversation(
            request=request,
            session_id=session_id,
            response_text=response_text,
            citations=citations,
        )
        self._increment_custom_role_usage(request.role_id, request.user_id)

        suggested = [] if simple_mode else [
            self._polish_model_output(item)
            for item in self._generate_suggestions(request.message, response_text)
        ]
        risk_level = "low" if simple_mode else self._assess_risk(response_text)

        return ChatResponse(
            reply=response_text,
            citations=citations,
            suggested_questions=suggested,
            risk_level=risk_level,
            disclaimer=DEFAULT_DISCLAIMER,
            session_id=session_id,
            response_time=round(time.time() - start_time, 2),
        )

    async def _retrieve_docs(
        self,
        request: ChatRequest,
        simple_mode: bool,
    ) -> Tuple[List[dict], bool]:
        if not self._should_query_milvus(request):
            return [], False

        effective_legal_field = self._resolve_legal_field(request)
        logger.info(
            "Querying Milvus for user=%s role=%s legal_field=%s simple_mode=%s",
            request.user_id,
            request.role_id,
            effective_legal_field,
            simple_mode,
        )
        try:
            docs = await self.rag.retrieve(
                query=request.message,
                legal_field=effective_legal_field,
            )
        except Exception as exc:
            logger.warning(
                "Milvus retrieval unavailable for user=%s role=%s legal_field=%s, "
                "falling back to direct LLM reply: %s",
                request.user_id,
                request.role_id,
                effective_legal_field,
                exc,
            )
            return [], False
        logger.info(
            "Milvus retrieval completed for user=%s role=%s legal_field=%s docs=%s",
            request.user_id,
            request.role_id,
            effective_legal_field,
            len(docs),
        )
        return docs, True

    async def _save_conversation(
        self,
        request: ChatRequest,
        session_id: str,
        response_text: str,
        citations: List[Citation],
    ) -> None:
        await self.memory.save_message(
            request.user_id,
            session_id,
            "user",
            request.message,
        )
        await self.memory.save_message(
            request.user_id,
            session_id,
            "assistant",
            response_text,
            [c.model_dump() for c in citations],
        )

    # 私有方法的定义，用于构建提示词
    def _build_prompt(   # 单下划线表示这是一个私有方法
        self,  # 实例方法
        request: ChatRequest,  # 参数1：用户请求对象
        context: str,  # 参数2：历史对话（字符串）
        retrieved_docs: List[dict],  # 参数3：检索到的文档列表
        simple_mode: bool = False,  # 参数4：简单模式，默认False
        rag_used: bool = False,  # 参数5：是否使用RAG，默认False
    ) -> str:  # 返回值：字符串类型的prompt
        # 检查是否为自定义角色（role_id为数字）
        try:
            role_id_int = int(request.role_id)  # 将role_id转换为整数
            custom_role = self._get_custom_role(role_id_int, request.user_id)  # 获取自定义角色
            is_custom_role = custom_role is not None  # 判断是否为自定义角色
        # 这是一个防御性编程
        except (ValueError, TypeError):  # ValueError--值错误, TypeError--类型错误
            # ValueError：值错误
                # 例如：尝试将字符串 "abc" 转换为整数时
                # 或者传入了不合法的枚举值
            # TypeError：类型错误
                # 例如：对 None 类型的对象调用 .value 属性
                # 或者类型不匹配的操作
            is_custom_role = False   # 将"是否为自定义角色"标记设为 False，表示当前不是自定义角色（可能是标准角色或角色无效）
            custom_role = None  # 将自定义角色对象设为None，表示没有有效的自定义角色数据
        '''
        容错性：即使解析失败，程序也不会崩溃
        状态重置：将相关变量重置为安全的默认值
        降级处理：保证后续代码仍能正常运行（使用标准流程而非自定义角色）
        '''

        # 简单模式的提示词
        simple_prompt = (
            f"历史对话：\n{context or '无'}\n\n"   # 对话历史，有内容就显示内容，没内容就返回无，作用：给AI提供对话上下文，保持连贯性
            f"用户问题：\n{request.message}\n\n"   # 当前问题，直接嵌入用户当前发送的消息，明确标识这是需要回答的核心问题
            "请直接回答用户问题，控制在1到3段内，语言简洁自然。"  # 回答格式约束
            "如果只是问候、定义、区别、时间、流程或简短判断，不要使用Markdown标题和列表。"  # 样式约束，对于简单的问题，Markdown标题和列表会显得冗余、不自然。
        )

        # 无知识库上下文时的提示词，用于当RAG 检索没有返回相关文档时的场景
        no_context_prompt = (
            f"历史对话：\n{context or '无'}\n\n"  # 对话历史
            f"用户问题：\n{request.message}\n\n"  # 当前问题
            "当前没有可直接引用的知识库条文。请结合已有法律专业知识进行审慎分析，"  # 情景说明，避免了大模型的幻觉，防止 AI 因为"不知道"而编造法条
            "明确提示用户仍需继续核对知识库内容、最新规范和案件事实，"  # 责任声明
            "不要虚构具体法条编号。若关键信息不足，请明确指出需要补充的事实或证据。"  # 行为约束
        )

        if is_custom_role and custom_role:
            # 使用自定义角色
            role_specialties = ', '.join(custom_role.get("specialties", []))

            if simple_mode:
                return f"角色专长：{role_specialties}\n\n{simple_prompt}"

            if not rag_used:
                return f"角色专长：{role_specialties}\n\n{no_context_prompt}"

            knowledge_context = self.rag.format_context(retrieved_docs)
            return get_custom_role_prompt(
                role=custom_role,
                specialties=role_specialties,
                context=f"历史对话：\n{context or '无'}\n\n相关法律知识：\n{knowledge_context}",
                question=request.message
            )

        # 使用系统默认角色
        role_specialties = get_role_specialties(request.role_id)
        prompt_template = get_prompt_template(request.role_id)

        if simple_mode:
            return f"角色专长：{role_specialties}\n\n{simple_prompt}"

        if not rag_used:
            return f"角色专长：{role_specialties}\n\n{no_context_prompt}"

        knowledge_context = self.rag.format_context(retrieved_docs)
        return prompt_template.format(
            specialties=role_specialties,
            context=f"历史对话：\n{context or '无'}\n\n相关法律知识：\n{knowledge_context}",
            question=request.message,
        )

    # 将检索到的文档列表转换为标准化的引用格式
    # 列表推导式，遍历 retrieved_docs 中的每个文档，把每个文档转换成 Citation 对象
    def _build_citations(self, retrieved_docs: List[dict]) -> List[Citation]:
        return [
            Citation(                           # 创建引用对象
                source=doc.get("source", ""),   # 文档来源（文件名/URL）
                article=doc.get("article_number", ""),    # 文章编号
                content=(doc.get("content", "") or "")[:settings.CITATION_CONTENT_MAX_CHARS],    # 内容片段
                score=doc.get("score", 0),      # 相关性分数
            )
            for doc in retrieved_docs           # 遍历每个文档
        ]

    def _resolve_legal_field(self, request: ChatRequest) -> str | None:  # 解析法律领域
        # 作用：确定当前对话所属的法律领域（如劳动法、婚姻法、刑法等）
        # request: ChatRequest：参数名为 request，类型注解为 ChatRequest（一个自定义的请求数据类）
        # 逻辑：
            # 如果请求中直接指定了 legal_field，直接返回其值
            # 否则根据 role_id（角色ID）从映射表 ROLE_LEGAL_FIELD_MAP 中查找对应的法律领域
            # 找不到则返回 None
        if request.legal_field:  # 如果解析的法律邻域
            return request.legal_field.value   # 返回该法律邻域字段的.value属性，eg：LegalField.LABOR.value → "劳动法"
        return ROLE_LEGAL_FIELD_MAP.get(str(request.role_id))
        # 如果请求中没有直接指定法律领域，执行这行
            # ROLE_LEGAL_FIELD_MAP：一个预定义的映射字典，键是角色ID，值是对应的法律领域
            # str(request.role_id)：将角色ID转为字符串，用作字典的键
            # .get() 方法：如果找到对应的角色ID，返回其法律领域；找不到则返回 None
        # 业务逻辑流程图解：
        # 开始
        #   ↓
        # 请求中有 legal_field 吗？
        #   ↓ 是              ↓ 否
        # 返回 legal_field.value   根据 role_id 从映射表中查找
        #                               ↓
        #                          找到？→ 返回对应领域
        #                               ↓
        #                          没找到 → 返回 None

    def _should_query_milvus(self, request: ChatRequest) -> bool:  # 判断是否需要查询向量数据库
        return bool((request.message or "").strip())

    def _build_system_prompt(
        self,
        role_id: str,
        user_id: str | None = None,
        simple_mode: bool = False,
        rag_used: bool = False,
    ) -> str:
        # 检查是否为自定义角色（role_id为数字）
        try:
            role_id_int = int(role_id)
            custom_role = self._get_custom_role(role_id_int, user_id)
            is_custom_role = custom_role is not None
        except (ValueError, TypeError):
            is_custom_role = False
            custom_role = None

        if is_custom_role and custom_role:
            # 使用自定义角色的系统提示词
            base_prompt = get_custom_role_system_prompt(custom_role)
        else:
            # 使用系统默认角色的系统提示词
            base_prompt = get_system_prompt(role_id)

        if simple_mode:
            return (
                f"{base_prompt}\n\n"
                "最终输出要求：直接回答，语气自然，不要使用Markdown标题或列表。"
            )
        if not rag_used:
            return (
                f"{base_prompt}\n\n"
                "最终输出要求：在缺少可引用条文时，明确提示需要继续核对知识库内容与案件事实，"
                "不要虚构具体法条编号。"
            )
        return (
            f"{base_prompt}\n\n"
            "最终输出要求：优先使用“## 核心说明”“## 细分说明”“## 法律依据”三段结构。"
        )

    def _finalize_response_text(
        self,
        request: ChatRequest,
        response_text: str,
        citations: List[Citation],
        simple_mode: bool,
    ) -> str:
        polished = self._polish_model_output(response_text)
        if simple_mode:
            return self._ensure_simple_reply(polished)
        if self._is_legal_role(request.role_id):
            return self._ensure_legal_sections(polished, citations)
        return polished

    def _resolve_max_tokens(self, simple_mode: bool) -> int:
        if simple_mode:
            return settings.SIMPLE_CHAT_MAX_TOKENS
        return settings.LLM_MAX_TOKENS

    def _is_legal_role(self, role_id: str) -> bool:
        if role_id in LEGAL_ROLE_IDS:
            return True

        try:
            int(str(role_id))
            return True
        except (TypeError, ValueError):
            return False

    def _get_custom_role(self, role_id: int, user_id: str | None = None) -> Optional[dict]:
        """获取自定义角色信息

        Args:
            role_id: 角色ID（整数类型）
            user_id: 用户ID（可选，字符串或None），用于权限验证

        Returns:
            角色字典，包含角色的所有字段信息，如果角色不存在或无权限则返回None
        """
        # 检查数据库连接是否存在
        # 如果没有数据库连接，无法查询角色信息
        if not self.db:
            return None

        # 从数据库中查询指定ID的角色
        # Role 是数据库模型类，对应角色表
        # filter(Role.id == role_id) 过滤出ID匹配的角色
        # first() 返回第一条结果，如果没有找到则返回None
        role = self.db.query(Role).filter(Role.id == role_id).first()

        # 如果数据库中不存在该角色，返回None
        if not role:
            return None

        # 准备请求者ID（用于权限验证）
        # 初始化为None，表示未登录或ID无效的用户
        requester_id: int | None = None

        # 如果提供了user_id，尝试将其转换为整数
        if user_id is not None:
            try:
                # 将user_id转换为字符串再转为整数，确保兼容性
                # 这样既支持字符串如"123"，也支持已经是整数的情况
                requester_id = int(str(user_id))
            except (TypeError, ValueError):
                # 如果转换失败（如user_id="abc"或None），保持为None
                # 转换失败意味着用户未登录或ID格式无效
                requester_id = None

        # 权限判断逻辑：
        # 1. is_system_role: 系统角色的 owner_id 为 None，所有人可用
        # 2. role.is_public: 角色是公开的，所有人可用
        # 3. 角色是用户自己创建的（owner_id == requester_id）
        is_system_role = role.owner_id is None  # 判断是否为系统预设角色
        has_permission = (
                is_system_role  # 系统角色：有权限
                or role.is_public  # 公开角色：有权限
                or (requester_id is not None and role.owner_id == requester_id)  # 自己的角色：有权限
        )

        # 如果没有权限，返回None（拒绝访问）
        if not has_permission:
            return None

        # 权限验证通过，返回角色的完整信息（字典格式）
        return {
            "id": role.id,  # 角色唯一标识
            "name": role.name,  # 角色名称（如"python_expert"）
            "display_name": role.display_name,  # 显示名称（如"Python专家"）
            "description": role.description,  # 角色描述
            "specialties": role.specialties or [],  # 专业领域列表，如["Python", "数据分析"]
            "prompt_template": role.prompt_template,  # 提示词模板
            "system_prompt": role.system_prompt,  # 系统提示词
            "answer_style": role.answer_style,  # 回答风格（如"简洁"、"详细"）
            "tags": role.tags or [],  # 标签列表，用于分类
            "is_public": role.is_public,  # 是否公开（True=所有人可见）
            "avatar": role.avatar,  # 角色头像URL
            "owner_id": role.owner_id  # 创建者ID（系统角色为None）
        }

    # 私有方法，用于增加自定义角色的使用次数统计，目前只做了数据库连接的检查
    def _increment_custom_role_usage(self, role_id: str, user_id: str | None = None) -> None:
        if not self.db:  # 判断有没有连接上数据库，如果没连数据库，无法进行统计操作
            return       # 直接返回，什么都不做

        try:
            role_id_int = int(str(role_id))  # 将字符串形式的role_id转换为整数
        except (TypeError, ValueError):
            # 异常处理：
            # TypeError - role_id类型错误（如None、列表）
            # ValueError - 无法转换（如"abc"、"12.5"）
            return   # 直接返回，不统计

        role = self.db.query(Role).filter(Role.id == role_id_int).first()
        # 从数据库中查询角色，查询逻辑：Role.id = role_id_int --- 匹配角色ID
        # 过滤条件：
            # not role - 角色不存在 → 返回
            # role.owner_id is None - 系统角色（不是自定义角色）→ 返回
            # 说明：只统计自定义角色（有owner_id的），系统角色不统计
        if not role or role.owner_id is None:
            return

        # 将用户ID从字符串或整数类型统一转换为整数类型，如果转换失败则设为None，表示“无效用户”或者“未登录用户”
        requester_id: int | None = None  # 初始化：没有有效的用户ID
        if user_id is not None:          # 如果传入了用户ID
            try:
                requester_id = int(str(user_id))  # 尝试转换为整数
            except (TypeError, ValueError):     # 转换失败
                requester_id = None  # 未登录或无效

        if not role.is_public and requester_id != role.owner_id:
            # 权限检查，决定是否有资格增加使用统计
            # 逻辑分解：
                # not role.is_public - 角色是私有的（非公开）
                # requester_id != role.owner_id - 当前用户不是角色的创建者
                # 两个条件同时成立时，拒绝统计
            return

        role.usage_count = (role.usage_count or 0) + 1
        # 将角色的使用次数加1
        # 处理空值：
            # 如果 role.usage_count 是 None，(None or 0) 返回 0
            # 如果 role.usage_count 是 5，(5 or 0) 返回 5
        self.db.commit() # 提交事务，将更新永久保存到数据库中
        # 为什么需要 commit：
            # 数据库操作默认在事务中
            # 只有 commit 后更改才生效
            # 如果不 commit，其他查询看不到这个更新

    # 这是一个模型输出抛光方法，用于清理和标准化AI生成的原始文本，移除Markdown标记、多余空格和换行，使输出更干净整洁。
    def _polish_model_output(self, text: str) -> str:
        if not text:  # 作用：如果输入文本为空，直接返回空字符串
            return ""

        polished = text.replace("\r", "\n")  # 统一换行符  作用：将Windows风格的换行符 \r\n 或单独的 \r 统一替换为Unix风格的 \n
        polished = polished.replace("```markdown", "").replace("```", "")
        # 移除代码块标记   作用：移除Markdown代码块的开始和结束标记
        polished = re.sub(r"(?m)^(#{1,6})\s*\*\*(.*?)\*\*\s*$", r"\1 \2", polished)
        # 处理标题中的加粗
        r'''
        正则分解：
            (?m) - 多行模式
            ^ - 行首
            (#{1,6}) - 捕获组1：1-6个井号
            \s* - 可选空白
            \*\* - 两个星号（加粗开始）
            (.*?) - 捕获组2：非贪婪匹配任意字符
            \*\* - 两个星号（加粗结束）
            \s*$ - 行尾可选空白
            替换：\1 \2 - 井号 + 空格 + 内容（移除加粗标记）
        '''
        polished = re.sub(r"(?m)^(\s*[-*]\s+)\*\*(.*?)\*\*\s*$", r"\1\2", polished)
        # 处理无序列表中的加粗
        r'''
        正则分解：
            (\s*[-*]\s+) - 捕获组1：列表标记（如 - 或 *）
            \*\*(.*?)\*\* - 加粗内容
        '''
        polished = re.sub(r"(?m)^(\s*\d+\.\s+)\*\*(.*?)\*\*\s*$", r"\1\2", polished)
        # 处理有序列表中的加粗
        polished = re.sub(r"[ \t]+", " ", polished)
        # 合并多个空格，作用：将连续的空格或制表符合并为一个空格
        polished = re.sub(r" *\n *", "\n", polished)
        # 处理换行周围的空格  作用：移除换行符前后的空格
        polished = re.sub(r"\n{3,}", "\n\n", polished)
        # 合并多余换行  作用：将3个及以上连续换行符替换为2个
        return polished.strip()  # 去除开头和结尾的空白字符

    # 这是一个流式输出块的抛光方法，用于实时处理每个流式数据块，确保输出格式干净
    # 方法概述：对每个流式输出的文本块进行实时清理，移除特殊字符、统一空格格式，并根据是否简单模式决定清理程度
    def _polish_stream_chunk(self, chunk: str, simple_mode: bool = False) -> str:
        # 作用：防御性编程，空块不处理
        if not chunk:   # 如果 chunk 是空的
            return ""   # 就返回空字符串，直接返回，不执行后续任何操作

        polished = chunk.replace("\r", "\n")  # 统一换行符，作用：将 \r（回车符）统一替换为 \n（换行符）
        if simple_mode:  # 简单模式
            polished = polished.replace("```markdown", "").replace("```", "")
            polished = polished.replace("#", "").replace("*", "")
            # 作用：简单模式下移除Markdown标记，保持纯文本
            # 移除内容：
                # ```markdown 和 ``` - 代码块标记
                # - Markdown标题标记
                # * - Markdown加粗/斜体标记
        polished = re.sub(r"[ \t]+", " ", polished)  # 将连续的空格或制表符合并为一个空格
        return polished  # 返回处理结果

    # 智能判断用户消息是否为简单模式的方法，通过多种规则来确定是否可以快速回复而不需要复杂处理
    # 通过分析消息的长度、内容模式、标点符号等多个维度，判断用户的问题是否简单到可以直接回复
    def _should_reply_directly(self, message: str) -> bool:
        text = re.sub(r"\s+", "", message or "").lower()
        # 作用：清理和标准化文本
            # re.sub(r"\s+", "", ...) - 移除所有空白字符（空格、换行、制表符）
            # message or "" - 如果message是None，用空字符串代替
            # .lower() - 转换为小写
        if not text:  # 作用：如果清理后文本为空，返回False（不视为简单模式）
            return False

        if text in GREETING_PATTERNS:  # 如果文本完全匹配预设的问候模式，返回True
            return True

        if len(text) <= 10 and any(pattern in text for pattern in GREETING_PATTERNS):
            # 作用：短文本（<=10字符）且包含问候词，返回True
            return True

        has_complex_facts = any(pattern in text for pattern in COMPLEX_FACT_PATTERNS)
        # 检测是否包含复杂事实查询的关键词

        # 判断短文本是否包含简单解释模式，如果符合条件且没有复杂事实需求，就认为是简单问题
        if len(text) <= 24 and any(pattern in text for pattern in SIMPLE_EXPLANATION_PATTERNS):
            # 短文本（≤24字符）匹配解释模式时，如果没有复杂事实需求才返回True
            return not has_complex_facts  # 取反复杂事实标志

        # 判断是否为简短的判断/确认类消息，如果是直接返回True
        if len(text) <= 20 and any(pattern in text for pattern in SIMPLE_JUDGMENT_PATTERNS):
            return True

        # 判断是否为简短的疑问句，且不包含复杂事实需求
        if len(text) <= 14 and text.endswith(("吗", "么", "呢", "?")) and not has_complex_facts:
            # 条件拆解
                # len(text) <= 14 - 文本长度不超过14个字符
                # text.endswith(("吗", "么", "呢", "?")) - 以疑问词结尾
                # not has_complex_facts - 不包含复杂事实关键词
            return True

        return False

    # 这是一个确保简单回复格式干净的方法，用于移除Markdown格式标记，使回复保持纯文本的简洁形式
    # 方法概述：将AI生成的回复转换成干净的纯文本，移除所有Markdown格式（标题、列表等），适合在简单模式下返回给用户
    def _ensure_simple_reply(self, text: str) -> str:
        cleaned = self._polish_model_output(text)  # 作用：先对原始文本进行基础抛光（去除多余空格、修正格式等）
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)  # 移除Markdown标题
        # 正则表达式分解：(?m) - 多行模式（每行开头独立匹配），^ - 行首
        # \s{0,3} - 0到3个空白字符（缩进）
        # #{1,6} - 1到6个井号（Markdown标题标记）
        # \s* - 后面的空白字符
        cleaned = re.sub(r"(?m)^\s*(?:[-*]|\d+\.)\s+", "", cleaned)  # 移除Markdown列表标记
        r'''
        正则表达式分解：
            (?m) - 多行模式
            ^ - 行首
            \s* - 0个或多个空白字符
            (?:[-*]|\d+\.) - 非捕获组，匹配：
            [-*] - 短横线或星号（无序列表）
            | - 或
            \d+\. - 数字加小数点（有序列表）
            \s+ - 至少一个空白字符
        '''
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # 合并多余换行
        return cleaned.strip()  # 移除开头和结尾的空白字符

    # 这是一个确保法律回答具有标准章节结构的方法，用于格式化法律咨询的回复
    # 这个方法的作用是：确保法律相关的回答包含"核心说明"和"法律依据"两个标准章节，如果缺失则自动补充
    def _ensure_legal_sections(self, text: str, citations: List[Citation] | None = None) -> str:
        citations = citations or []  # 作用：如果 citations 是 None，设为空列表
        normalized = self._polish_model_output(text)  # 作用：对原始文本进行基础抛光（去除多余空格、格式化等）

        if "## 核心说明" in normalized and "## 法律依据" in normalized:  # 如果文本已经包含两个必要章节，直接返回（无需处理）
            return normalized

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
        # 作用：按两个以上换行符分割文本，提取非空段落
        # 正则表达式：\n{2,} - 匹配2个或更多连续换行符
        # 例如：
        # 输入：
        # text = "第一段\n\n第二段\n\n\n第三段"
        # 分割结果：
        # paragraphs = ["第一段", "第二段", "第三段"]
        if not paragraphs:
            paragraphs = ["请结合更具体的事实、时间线和证据，再补充一次问题。"] # 如果没有有效段落，使用默认提示文本

        core = paragraphs[0]  # 作用：第一段作为“核心说明”
        detail = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else "建议先明确争议焦点、关键证据和当前所处阶段，再进一步分析。"
        # 作用：
        # 如果有多个段落，其余段落作为"详细说明"
        # 如果只有一个段落，使用默认提示文本
        legal_basis = self._build_legal_basis_section(citations)
        # 作用：从引用列表中构建法律依据章节（代码未完整显示）

        # 分成3个部分：
        return (
            "## 核心说明\n"
            f"{core}\n\n"
            "## 细分说明\n"
            f"{detail}\n\n"
            "## 法律依据\n"
            f"{legal_basis}"
        )

    # 这是一个构建法律依据章节的方法，用于将检索到的引用文档格式化为法律回答中的"法律依据"部分
    # 作用：将检索到的引用列表（citations）转换成格式化的法律依据文本，去重并限制数量，用于法律回答的"## 法律依据"章节
    def _build_legal_basis_section(self, citations: List[Citation]) -> str:
        if not citations:  # 如果没有引用来源，返回默认提示文本
            return "- 本次回答未返回可展示的知识库引文，请结合检索结果与案件事实继续核对。"

        lines = []  # 存储格式化后的法律依据行
        seen = set()  # 去重，避免重复添加相同内容
        for citation in citations[:4]:  # 最多处理前4个引用（避免法律依据过长）
            title = " ".join(part for part in [citation.source or "", citation.article or ""] if part).strip()
            # 作用：将 source（来源）和 article（文章编号）组合成标题
            # 处理逻辑：
                # 只取非空的 part
                # 用空格连接
                # 去除首尾空格
            title = title or "相关法律依据"  # 默认的标题，使用默认值
            content = self._polish_model_output(citation.content or "") # 对引用内容进行抛光（移除多余空格、格式标记等）
            line = f"- {title}"        # 先构建基础：短横线 + 标题
            if content:                # 如果有内容
                line += f"：{content}"  # 组装法律依据的每一行
                '''
                eg: title = "民法典 第577条"
                    content = "当事人一方不履行合同义务的，应当承担违约责任"
                    
                    # 执行过程
                    line = f"- {title}"          # line = "- 民法典 第577条"
                    if content:                  # content不为空，进入
                        line += f"：{content}"   # line = "- 民法典 第577条：当事人一方不履行合同义务的，应当承担违约责任"
                    
                    # 最终结果
                    "- 民法典 第577条：当事人一方不履行合同义务的，应当承担违约责任"
                '''
            # 去重机制，用于防止添加重复的法律依据条目
            if line not in seen:  # 如果这条记录还没有出现过
                seen.add(line)  # 添加到已见集合中
                lines.append(line)  # 添加到结果列表

        return "\n".join(lines)  # 用换行符将列表中的所有元素连接成一个字符串

    # 这是一个生成推荐问题的方法，用于根据用户查询和AI回答的内容，智能推荐用户可能感兴趣的后续问题
    # 这个方法的作用是：基于用户问题和AI回答中的关键词，生成3个相关的后续问题建议，帮助用户继续深入询问
    def _generate_suggestions(self, query: str, response: str) -> List[str]:
        # query - 用户的原始问题，response - AI的回答内容，返回 - 推荐问题列表（最多3个）
        suggestions = []

        if "赔偿" in query or "赔偿" in response:
            suggestions.append("赔偿金额一般如何计算？")
            suggestions.append("我需要准备哪些关键证据？")
            # 涉及'赔偿'问题

        if "离婚" in query or "离婚" in response:
            suggestions.append("离婚时财产通常如何分割？")
            suggestions.append("子女抚养权判断重点有哪些？")
            # 涉及'离婚'话题

        if "合同" in query or "合同" in response:
            suggestions.append("合同无效的常见情形有哪些？")
            suggestions.append("违约金过高时能否请求调整？")

        if "工伤" in query or "工伤" in response:
            suggestions.append("工伤认定通常需要提交哪些材料？")
            suggestions.append("工伤赔偿标准一般怎么确定？")

        if not suggestions:
            suggestions = [
                "相关法律规定具体是什么？",
                "我下一步应该准备哪些证据？",
                "这个问题是否有时效限制？",
            ]

        return suggestions[:3]

    # 这是一个风险评估方法，用于分析AI回答中是否包含不同风险等级的法律关键词，帮助判断用户面临的法律风险程度
    def _assess_risk(self, response: str) -> str:  # 根据AI回答中出现的法律关键词，评估用户问题的风险等级，分为高风险、中风险、低风险三个级别
        high_risk_keywords = ["刑事责任", "刑事拘留", "逮捕", "判刑", "犯罪"]  # 高风险关键词  涉及刑事法律后果的关键词
        medium_risk_keywords = ["赔偿", "违约金", "解除合同", "诉讼", "仲裁"]  # 中风险关键词  涉及民事或经济纠纷的关键词

        for keyword in high_risk_keywords:
            if keyword in response:
                return "high"    # 只要有一个高风险词，立即返回high

        for keyword in medium_risk_keywords:
            if keyword in response:
                return "medium"  # 没有高风险词但有中风险词，返回medium

        return "low"  # 都没有就返回低风险

    # 性能监控日志方法，用于记录响应时间是否超过预设的软目标阈值
    # 这是一个私有方法，用于检查API响应时间是否超过了设定的“软目标”时间，如果超过就记录警告日志。
    def _log_soft_target(self, elapsed: float | None, user_id: str, stream_mode: bool) -> None:
        # 方法签名：
        # elapsed: float | None - 响应耗时（秒），可能为None
        # user_id: str - 用户ID
        # stream_mode: bool - 是否为流式模式（True=流式，False=非流式）
        # -> None - 无返回值
        if elapsed is None:  # 如果耗时为None，直接返回（不记录日志）
            return
        if elapsed > settings.SOFT_RESPONSE_TARGET_SECONDS:  # 如果响应时间超过了设定的软目标阈值
            logger.info(   # 当超时时，记录INFO级别日志
                "Chat %s response exceeded soft target: %.2fs for user %s",
                "stream" if stream_mode else "normal",  # 模式标识
                elapsed,    # 实际耗时
                user_id,    # 用户ID
            )

    # 确保法律文本输出符合特定格式要求的方法，用于规范化法律相关回答的结构
    # 这是一个私有方法，用于确保法律相关的回答包含必需的章节标题，并按照标准格式组织内容。
    def _ensure_legal_sections(self, text: str, citations: List[Citation] | None = None) -> str:
        # 方法签名：
            # text: str - 原始回答文本
            # citations: List[Citation] | None - 引用来源列表（可选）
            # -> str - 返回格式化后的文本
        citations = citations or []  # 空值处理：如果 citations 是 None，设为空列表
        normalized = self._normalize_legal_section_titles(self._polish_model_output(text))
        # 文本处理（从内到外）：
            # self._polish_model_output(text) - 抛光原始输出（去除多余空格、格式化等）
            # self._normalize_legal_section_titles() - 标准化法律章节标题
            # 结果赋值给 normalized
        required_titles = (  # 必需的章节标题：定义了法律回答必须包含的4个标准章节
            "## 核心说明",   # 法律问题的核心理论
            "## 细分说明",   # 详细分析和解释
            "## 相关法条",   # 引用的法律法规
            "## 相关例子",   # 实际案例或示例
        )
        # 检查并补全法律回答的章节结构，确保输出包含所有必需的四个章节
        # 功能：检查AI生成的回答是否包含所有必需的法律章节，如果缺少某些章节，就自动构建补充内容。
        if all(title in normalized for title in required_titles):
            # 作用：判断 normalized 中是否包含所有必需的章节标题
            return normalized
            # 逻辑分解：
                # title in normalized for title in required_titles - 遍历每个必需标题，检查是否在文本中
                # all(...) - 所有标题都存在时返回 True
                # 如果都存在，直接返回原文本（不需要补充）

        # 构建完整法律回答的核心逻辑，通过提取或生成四个标准章节，最终组装成格式化的法律咨询回复
        # 作用：将AI生成的原始回答和引用资料，整理成包含四个标准章节（核心说明、细分说明、相关法条、相关例子）的结构化法律回答
        sections = self._extract_markdown_sections(normalized)  # 从清理后的文本中提取已有的Markdown章节
        core = sections.get("## 核心说明") or self._first_meaningful_paragraph(normalized)
        # 逻辑：
        # 优先使用已有的 ## 核心说明 章节
        # 如果没有，从文本中提取第一个有意义的段落
        detail = sections.get("## 细分说明") or self._build_detail_section(normalized)
        legal_basis = sections.get("## 相关法条") or self._build_legal_basis_section(citations)
        examples = sections.get("## 相关例子") or self._build_example_section(citations, core, detail)

        return (
            "## 核心说明\n"
            f"{core}\n\n"
            "## 细分说明\n"
            f"{detail}\n\n"
            "## 相关法条\n"
            f"{legal_basis}\n\n"
            "## 相关例子\n"
            f"{examples}"
        )

    def _build_legal_basis_section(self, citations: List[Citation]) -> str:
        if not citations:
            return "- 当前没有命中可直接展示的 Milvus 法条内容，建议补充案件事实后重新检索。"

        lines = []
        seen = set()
        for citation in citations[:4]:
            title = " ".join(
                part for part in [citation.source or "", citation.article or ""] if part
            ).strip()
            title = title or "相关法律依据"
            content = self._truncate_sentence(
                self._polish_model_output(citation.content or ""),
                120,
            )
            line = f"- {title}"
            if content:
                line += f"：{content}"
            if line not in seen:
                seen.add(line)
                lines.append(line)

        return "\n".join(lines) if lines else "- 当前没有可展示的法条摘要。"

    def _build_example_section(
        self,
        citations: List[Citation],
        core: str,
        detail: str,
    ) -> str:
        lines = []
        for index, citation in enumerate(citations[:2], start=1):
            source = " ".join(
                part for part in [citation.source or "", citation.article or ""] if part
            ).strip() or f"检索结果{index}"
            content = self._truncate_sentence(
                self._polish_model_output(citation.content or ""),
                100,
            )
            if content:
                lines.append(
                    f"- 例子{index}：如果你的情况与“{source}”描述的情形接近，比如 {content}，通常就要重点核对事实是否满足该条规则的适用条件。"
                )

        if lines:
            return "\n".join(lines)

        fallback_core = self._truncate_sentence(core, 60)
        fallback_detail = self._truncate_sentence(detail, 80)
        return (
            f"- 例如：如果实际事实与“{fallback_core}”对应，且你能提供合同、聊天记录、付款凭证、劳动材料或身份关系证明，"
            "通常就可以围绕法律关系是否成立、对方是否违约、是否存在赔偿责任来展开判断。\n"
            f"- 再如：若争议重点集中在“{fallback_detail}”，就要把时间线、通知记录、对方回复和书面材料整理出来，再与 Milvus 命中的法条逐项比对。"
        )

    def _normalize_legal_section_titles(self, text: str) -> str:
        normalized = text or ""
        title_aliases = {
            "## 核心说明": ["## 核心结论", "## 结论", "## 重点结论"],
            "## 细分说明": ["## 详细分析", "## 具体分析", "## 分析", "## 细化说明"],
            "## 相关法条": ["## 法律依据", "## 相关法律依据", "## 法条依据", "## 法律条案"],
            "## 相关例子": ["## 相关案例", "## 案例示例", "## 举例说明", "## 示例"],
        }
        for canonical, aliases in title_aliases.items():
            for alias in aliases:
                normalized = normalized.replace(alias, canonical)
        return normalized

    # 这是一个提取Markdown章节的方法，用于从文本中解析出所有二级标题（##）及其对应的内容
    # 作用：遍历文本，识别所有 ## 开头的标题，将每个标题和它下面的内容提取成键值对，返回一个字典。
    def _extract_markdown_sections(self, text: str) -> dict[str, str]:  # 输入的是文本，输出的是字典
        sections: dict[str, str] = {}  # 存储结果的字典
        current_title: str | None = None  # 当前正在处理的标题
        buffer: list[str] = []  # 缓存当前标题下的内容行

        for line in (text or "").splitlines():  # 如果text为None，用空字符串，按行分割，遍历每一行进行处理
            stripped = line.strip()    # 去除首尾空白字符（空格、换行、制表符等），用于判断是否是
            if stripped.startswith("## "):  # 判断是否是Markdown二级标题（以##开头），注意：必须严格是##（两个井号+空格）
                if current_title:
                    content = "\n".join(buffer).strip()
                    if content:
                        sections[current_title] = content
                        # current_title - 上一个正在处理的标题
                        # 将buffer中缓存的内容用换行符连接
                        # strip() - 去除首尾空白
                        # 如果内容不为空，保存到字典
                current_title = stripped   # 设置当前标题为新标题
                buffer = []   # 清空buffer，准备缓存新标题下的内容
                continue
            # 处理非标题行
            if current_title:
                buffer.append(line)
                '''
                如果不是标题行，但当前正在某个标题下
                将原始行（不是strip后的）添加到buffer
                保留原始格式（缩进、空格等）
                '''

        if current_title:  # 保存最后一个章节
            content = "\n".join(buffer).strip()
            if content:
                sections[current_title] = content
                '''
                作用：循环结束后，将最后一个标题及其内容保存到字典中
                    为什么需要这段代码？
                    循环中只在遇到新标题时才保存上一个标题的内容
                    最后一个标题后面没有新标题来触发保存
                    所以需要在循环结束后单独保存
                '''

        return sections

    # 提取第一个有意义段落的方法，用于从文本中获取最重要的核心段落，如果没有则返回默认提示
    def _first_meaningful_paragraph(self, text: str) -> str: # 这个方法的作用是：从文本中提取第一个非空的段落，作为法律回答的"核心说明"章节的备用内容。
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text or "") if part.strip()]
        '''
        分解：
            text or "" - 如果text为None，用空字符串
            re.split(r"\n{2,}", ...) - 按2个及以上连续换行符分割（段落分隔符）
            part.strip() - 去除每个段落的首尾空白
            if part.strip() - 只保留非空段落
            [...] - 列表推导式
        '''
        if paragraphs:
            return paragraphs[0]   # 如果有段落，返回第一个
        return "请结合更具体的事实经过、时间节点和已有证据，再补充一次问题。"

    def _build_detail_section(self, text: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text or "") if part.strip()]
        remaining = paragraphs[1:]
        if remaining:
            return "\n\n".join(remaining)
        return "建议先明确争议焦点、证据情况、对方身份以及当前所处阶段，再进一步判断维权路径和风险。"

    def _truncate_sentence(self, text: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"
