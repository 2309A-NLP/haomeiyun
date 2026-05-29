# app/api/chat.py

# 导入异步IO库，用于处理异步操作
import asyncio
# 导入JSON库，用于处理JSON数据的序列化和反序列化
import json

# 从FastAPI框架导入所需组件
from fastapi import APIRouter, Depends, HTTPException  # APIRouter: 创建路由; Depends: 依赖注入; HTTPException: HTTP异常处理
from fastapi.responses import StreamingResponse  # StreamingResponse: 流式响应
# 从SQLAlchemy ORM导入会话类型
from sqlalchemy.orm import Session
# 导入可选类型注解
from typing import Optional
# 从项目配置模块导入应用配置
from ..core.config import settings
# 从数据库模型模块导入数据库会话获取函数
from ..models.database import get_db
# 从Pydantic模型模块导入请求和响应模型
from ..models.schemas import ChatRequest, ChatResponse
# 导入RAG（检索增强生成）服务类
from ..services.rag_service import LegalChatService
# 导入日志记录器
from ..utils.logger import logger

# 创建路由路由器，设置路径前缀为"/chat"，标签为["对话"]用于API文档分组
router = APIRouter(prefix="/chat", tags=["对话"])


# 依赖注入函数：获取聊天服务实例  作用：为了解耦、可测试、易维护
#依赖注入让你声明"我需要什么"，而不是写代码"怎么得到它"。FastAPI 负责把"怎么得到"的逻辑集中管理，自动注入给你。

# 集中管理创建逻辑
# 依赖注入让代码从"强耦合、难测试、难修改"变成"松耦合、易测试、易扩展"
'''
使用注入依赖的好处：1.避免了重复写创建、关闭代码  
2.无法替代假的服务（只模仿真实服务的行为，但不执行真实的复杂操作）,可以快速、稳定、独立、安全、便宜
3.接口既可以处理HTTP、又管理对象创建
4.难以统一修改
'''
def get_chat_service(db: Session = Depends(get_db)) -> LegalChatService:
    """通过数据库会话依赖创建并返回法律聊天服务实例"""
    return LegalChatService(db=db)   # 手动创建服务

# POST请求端点：普通对话接口（非流式）
'''
作用：声明这是一个处理 HTTP POST 请求的函数
路径："" 表示空路径，实际完整路径是 /chat（因为 router 的 prefix 是 /chat）
HTTP方法：POST 通常用于创建资源或提交数据

作用：指定这个接口返回的数据格式必须是 ChatResponse 类型
好处：
FastAPI 会自动验证返回数据结构
自动生成 API 文档中的响应示例
自动进行数据转换（如把 datetime 转成字符串）
'''
@router.post("", response_model=ChatResponse)  # 响应模型为ChatResponse
async def chat(
        request: ChatRequest,  # 聊天请求对象（包含消息、用户ID、会话ID等）
        service: LegalChatService = Depends(get_chat_service)  # 注入聊天服务依赖
):
    """多角色对话接口"""
    try:
        # 记录聊天请求日志，截取消息前50个字符
        logger.info(f"Chat request from user {request.user_id}: {request.message[:50]}...")
        '''
        logger.info()：记录一条 INFO 级别的日志
        request.user_id：从自动解析的 ChatRequest 对象中获取用户ID
        request.message[:50]：只取消息的前50个字符
        防止日志过长（用户可能发很长的消息）
        保护隐私（不记录完整消息内容）
        用途：追踪和调试，方便查看谁在什么时候发了什么请求
        '''
        # 调用服务层的异步聊天方法
        response = await service.chat(request)
        '''
        await：异步等待，因为 AI 对话可能需要时间
        service.chat(request)：调用业务逻辑层
        这里会处理具体的对话逻辑
        比如调用大模型、检索法律知识库等
        service 是通过依赖注入获取的服务实例
        '''
        # 返回响应结果
        return response  # FastAPI 会自动将其转换为 JSON 格式的 HTTP 响应
    except Exception as e:
        # 记录错误日志
        logger.error(f"Chat failed: {e}")
        # 抛出500内部服务器错误HTTP异常
        raise HTTPException(status_code=500, detail=str(e))
        #捕获所有可能的异常（Exception 是万能捕获）
        # logger.error()：记录错误日志
        # raise HTTPException(...)：抛出 FastAPI 的 HTTP 异常
        # status_code=500：HTTP 500 内部服务器错误
        # detail=str(e)：把错误详情返回给客户端


# POST请求端点：流式对话接口（Server-Sent Events）
@router.post("/stream")
async def chat_stream(
        request: ChatRequest,  # 聊天请求对象 表示参数 service 应该是 LegalChatService 类型
        service: LegalChatService = Depends(get_chat_service)  # 注入聊天服务依赖 告诉 FastAPI："这个参数的值不是直接从请求里拿的，而是通过 get_chat_service() 这个函数来提供"
):
    """流式对话接口"""

    # 定义异步事件生成器函数
    async def event_generator():
        # 创建异步队列，用于在生产者和消费者之间传递数据
        queue = asyncio.Queue()
        '''
        问题：service.chat_stream() 是一个异步生成器，而我们需要向客户端发送 SSE 格式的数据。两者速度可能不匹配：
            生产者（AI 模型）：可能时快时慢
            消费者（HTTP 响应）：需要稳定的输出流
            解决：用队列作为缓冲区，解耦生产和消费：
        '''

        # 定义异步生产者函数：生成流式响应块
        async def produce_chunks():
            try:
                # 异步遍历服务层生成的流式响应块
                async for chunk in service.chat_stream(request):
                    # 将响应块放入队列，标记类型为"chunk"
                    await queue.put(("chunk", chunk))
                    # ("chunk", chunk)：元组格式，第一个元素是类型，第二个是数据
            except Exception as exc:
                # 记录生产者异常
                logger.error(f"Streaming producer failed: {exc}")
                # 将错误信息放入队列，标记类型为"error"
                await queue.put(("error", "抱歉，流式生成中断，请稍后重试。"))
            finally:
                # 无论成功或失败，最后放入"done"标记表示生产结束
                await queue.put(("done", None))
                # 三种类型：chunk（文本块）、error（错误）、done（结束）
                # 生产者独立运行，不阻塞主流程

        # 创建生产者异步任务
        producer_task = asyncio.create_task(produce_chunks())
        """
        创建一个独立运行的后台任务
        这个任务不会被主循环阻塞
        AI 开始在后台生成文本，不断往队列里放数据
        """

        try:
            # 主循环：从队列中获取数据并发送给客户端
            while True:
                try:
                    # 从队列获取数据，设置超时时间（用于发送心跳包）
                    kind, payload = await asyncio.wait_for(
                        queue.get(),  # 等待队列里有数据（可能等很久）
                        timeout=settings.STREAM_HEARTBEAT_SECONDS,  # 心跳超时时间配置
                    )
                    '''
                    asyncio.wait_for(..., timeout=30)：最多等30秒
                    如果30秒内没等到数据 → 触发 TimeoutError
                    超时时发送心跳包，保持连接
                    '''
                except asyncio.TimeoutError:
                    # 超时时发送SSE格式的心跳包，保持连接不中断
                    yield ": keep-alive\n\n"  # 心跳包
                    continue

                # 处理数据块类型
                if kind == "chunk":
                    # 构造JSON格式的数据块
                    data = json.dumps({"type": "chunk", "content": payload or ""}, ensure_ascii=False)
                    # ensure_ascii = False 是什么意思，始终保留原始中文
                    # payload or '' 防止 None 被序列化成 null，导致前端处理出错
                    # 以SSE格式输出：data: {json}\n\n
                    # 一句话总结：chunk 类型就像是给每个数据块贴上了标签，让前后端能够清晰地知道"这是什么数据"以及"该怎样处理它"。
                    yield f"data: {data}\n\n"
                    continue

                # 处理错误类型
                if kind == "error":
                    # 构造错误消息JSON
                    data = json.dumps({"type": "chunk", "content": payload or ""}, ensure_ascii=False)
                    # 发送错误消息
                    yield f"data: {data}\n\n"
                    break  # 错误发生后退出循环

                # 处理完成标记
                if kind == "done":
                    break  # 正常结束，退出循环

                '''
                六、关键细节总结
                    类型	    是否发送数据	      是否退出循环	后续是否发 [DONE]	   用途
                    chunk	✅ 发送	❌        continue	    ✅ 循环结束后发	   正常文本块
                    error	✅ 发送（错误信息）  ✅ break	    ❌ 不会发	       异常终止
                    done	❌ 不发送	      ✅ break	    ✅ 循环结束后发	   正常结束
                    
                统一格式：错误也用 chunk 类型，简化前端
                及时退出：error/done 立即 break，停止处理
                保证输出：yield 确保数据立即发送
                健壮性：payload or "" 防止 None 值异常
                '''
        finally:
            # 确保生产者任务被正确取消（如果还在运行中）
            if not producer_task.done():
                '''
                done() 返回 True：任务已经完成（正常结束或已取消）
                done() 返回 False：任务还在运行中
                '''
                producer_task.cancel()  # 取消任务
                '''
                不是强制停止，而是发送一个"请取消"的信号
                任务会在下一个 await 点抛出 CancelledError
                类似礼貌地通知："你可以停止了"
                '''
                try:
                    await producer_task  # 等待任务完成取消
                except asyncio.CancelledError:
                    pass  # 忽略取消异常

        # 发送结束标记（SSE标准格式）
        yield "data: [DONE]\n\n"

    # 返回流式响应对象，媒体类型为Server-Sent Events
    return StreamingResponse(   # 流式传输数据
        event_generator(),      # 内容生成器
        media_type="text/event-stream"    # 媒体类型
    )


# GET请求端点：获取聊天历史记录
@router.get("/history/{user_id}")
async def get_chat_history(
        user_id: str,  # 用户ID（路径参数）
        session_id: Optional[str] = None,  # 可选的会话ID（查询参数）
        service: LegalChatService = Depends(get_chat_service)  # 注入聊天服务
):
    """获取对话历史"""
    # 如果提供了会话ID，则获取指定会话的消息列表
    # 用户点击某个历史会话，查看具体的对话内容
    if session_id:
        # 从服务的memory组件中获取会话消息
        messages = service.memory.get_session_messages(user_id, session_id)
        # 返回会话ID和消息列表
        return {"session_id": session_id, "messages": messages}
    else:
        # 如果没有提供会话ID，则获取用户的所有会话列表
        sessions = service.memory.get_session_list(user_id)
        # 返回会话列表
        return {"sessions": sessions}


# POST请求端点：创建新的聊天会话
@router.post("/history/{user_id}/{session_id}")
async def create_chat_session(
        user_id: str,  # 用户ID（路径参数）
        session_id: str,  # 会话ID（路径参数）
        service: LegalChatService = Depends(get_chat_service)  # 注入聊天服务
):
    """创建空会话记录"""
    # 调用服务的memory组件创建新会话
    session = service.memory.create_session(user_id, session_id)
    # 返回成功消息和会话信息
    return {"message": "Session created", "session": session}


# DELETE请求端点：清除聊天历史记录
@router.delete("/history/{user_id}/{session_id}")
async def clear_chat_history(
        user_id: str,  # 用户ID（路径参数）
        session_id: str,  # 会话ID（路径参数）
        service: LegalChatService = Depends(get_chat_service)  # 注入聊天服务
):
    """清除对话历史"""
    # 调用服务的memory组件清除指定会话的内存
    service.memory.clear_memory(user_id, session_id)
    # 返回成功消息
    return {"message": "History cleared"}
