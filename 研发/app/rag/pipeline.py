'''
实现了一个法律领域的RAG（检索增强生成）检索管道，专门用于从法律知识库中检索相关法律条文
'''
from __future__ import annotations  # 支持类型注解中使用类自身的引用

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..core.config import settings
from ..utils.logger import logger
from ..vector_store.milvus_client import MilvusClient
from .embedding import BGEEmbedder
from .rerank import BGEReranker


class LegalRAGPipeline:
    """法律RAG检索管道 - 负责从法律知识库中检索相关法律条文"""

    def __init__(self):
        # 避免重复初始化（单例模式）
        if getattr(self, "_initialized", False) and not getattr(self, "_closed", False):
            # 检查对象是否已经初始化过且未被关闭，如果已初始化，直接返回，避免重复创建资源（节省内存和启动时间）
            return

        # 初始化向量化模型（将文本转换为向量）
        self.embedder = BGEEmbedder()
        '''
        作用：将法律文本转换成向量（一串数字）
        比喻：把法律条文翻译成计算机能理解的"数学语言"
        用途：后续通过计算向量相似度来找相关法律条文
        '''

        # 重排序器（用于提升检索结果的准确性）
        self.reranker = None
        '''
        作用：对初步检索结果重新排序，把最相关的排前面
        为什么懒加载：重排序模型比较耗资源，等真正需要时才创建
        '''

        # Milvus向量数据库客户端
        self.milvus = MilvusClient()
        '''
        作用：连接Milvus向量数据库
        比喻：像连接MySQL一样，但这个数据库专门存向量
        用途：存储和检索法律条文的向量表示
        '''

        # 检索参数配置
        self.top_k = settings.TOP_K    # 最多返回多少个结果 --- 8
        self.rerank_top_k = settings.RERANK_TOP_K  # 重排序后保留多少个 --- 4
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD  # 相似度阈值,也就是相似度的及格线 --- 0.7

        # 线程池执行器（将同步操作放到线程中执行，避免阻塞异步事件循环）
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-retrieve")
        '''
        作用：创建4个线程的工作池
        为什么需要：检索操作可能耗时，放到线程池里避免阻塞主程序
        比喻：开4个窗口同时处理检索请求
        '''

        self._initialized = True   # 标记已初始化
        self._closed = False       # 标记未关闭

    # 把阻塞操作放到线程池中执行，让事件循环可以继续处理其他任务
    async def retrieve(  # 这是一个异步包装器，调用时返回一个协程对象，不会立即执行
            self,
            query: str,  # 用户查询文本
            legal_field: Optional[str] = None,  # 法律领域（如：劳动法、婚姻法）
            top_k: int = None,  # 返回结果数量
    ) -> List[Dict]:
        """异步检索接口 - 对外提供的检索方法"""
        loop = asyncio.get_running_loop()   # 获取当前正在运行的异步事件循环，事件循环负责调度和执行所有异步任务
        # 在线程池中执行同步检索，避免阻塞事件循环
        return await loop.run_in_executor(
            self._executor,  # 线性池（4个线程）
            self._retrieve_sync,  # 要执行的同步函数
            query,     # 用户的查询文本
            legal_field,     # 法律领域
            top_k,           # 返回结果数量
        )
    '''
    run_in_executor：把同步函数提交给线程池执行
    await：等待线程池中的任务完成，获取结果
    在等待期间，事件循环可以执行其他任务
    '''

    def _retrieve_sync(
            self,
            query: str,  # 用户原始查询，如："公司开除我怎么办"
            legal_field: Optional[str] = None,  # 法律领域过滤，如'劳动法'、'婚姻法'
            top_k: int = None,     # 返回的结果数量，如：8
    ) -> List[Dict]:
        """同步检索核心逻辑"""
        start_time = time.time()   # 记录开始时间，用于性能监控

        # ========== 第1步：查询改写/扩展 ==========
        # 目的：将日常用语转换为法律专业术语，提高检索召回率
        # 例如："公司开除我怎么办" → "公司开除我怎么办 解除劳动合同 辞退 违法解除 经济补偿"
        rewritten_query = self._rewrite_query(query)

        # 记录日志，方便调用和追踪检索效果
        logger.info("Original query: %s", query)   # 原始查询
        logger.info("Rewritten query: %s", rewritten_query)  # 扩展后的查询

        # 2. 将查询文本转换为向量（用于向量相似度搜索）
        embed_start = time.time()  # 记录开始时间
        embed_result = self.embedder.encode_query(rewritten_query, return_sparse=False)  # 将文本转换为向量，只返回密集向量，不返回稀疏向量
        dense_vector = embed_result["dense_vecs"][0].tolist()  # 密集向量
        '''
        embed_result["dense_vecs"]：获取所有向量（批次结果）
        [0]：取第一个向量（因为只有一个查询）
        .tolist()：将numpy数组转换为Python列表
        '''
        logger.info("RAG query embedding completed in %.2fs", time.time() - embed_start)
        # 计算耗时：time.time() - embed_start
        # 格式化输出：保留2位小数
        # 例如："RAG query embedding completed in 0.05s"

        # 3. 构建过滤条件（按法律领域过滤）,构建Milvus向量数据库的过滤表达式，用于按法律领域筛选检索结果
        filter_expr = f'legal_field == "{legal_field}"' if legal_field else None
        '''
        总结
            这行代码的作用：
            有legal_field → 生成 'legal_field == "具体领域"' 过滤条件
            无legal_field → 返回None，不过滤
            核心价值：让检索系统能够按法律领域精准筛选，避免跨领域干扰，提高检索准确性
        '''

        # 确定检索数量（多检索一些用于后续重排序）
        # 确定检索数量的两个关键参数，一个是最终要返回的结果数，另一个是初检时要多取多少结果用于后续重排序
        target_top_k = max(1, top_k or min(self.top_k, settings.RAG_FAST_TOP_K))  # 确定最终返回数量
        search_k = target_top_k * max(settings.RAG_SEARCH_MULTIPLIER, 3)  # 确定初检数量


        # 4. 混合检索：结合密集向量检索和稀疏向量检索（BM25）
        # 混合检索的核心执行部分，它同时使用语义向量和关键词两种方式从向量数据库中检索相关法律条文
        search_start = time.time()  # 开始的时间
        hybrid_results = self.milvus.hybrid_search(
            dense_vector=dense_vector,  # ① 语义向量（768维数字）
            query=rewritten_query,  # ② 原始查询文本（用于关键词匹配）
            top_k=search_k,  # ③ 返回结果数量（如15条）
            filter_expr=filter_expr,  # ④ 过滤条件（如：只查劳动法）
            legal_field=legal_field,  # ⑤ 法律领域（辅助过滤）
            knowledge_type="law",  # ⑥ 知识类型（只查法律条文）
        )
        logger.info("RAG Milvus search completed in %.2fs", time.time() - search_start)

        # 5. 重排序：使用BGE重排序模型对结果重新打分（提升准确性）
        rerank_start = time.time()
        if hybrid_results and settings.RAG_ENABLE_RERANK:
            # 懒加载重排序器
            if self.reranker is None:
                self.reranker = BGEReranker()

            # 执行重排序
            reranked = self.reranker.rerank(
                query=rewritten_query,
                documents=[r["content"] for r in hybrid_results],  # 文档内容
                top_k=min(self.rerank_top_k, target_top_k),
            )

            # 补充文档元信息（来源、法条编号等）
            for i, r in enumerate(reranked):
                source_index = r.get("id", i)
                if isinstance(source_index, int) and source_index < len(hybrid_results):
                    source_doc = hybrid_results[source_index]
                    r.update(
                        {
                            "source": source_doc.get("source", ""),
                            "article_number": source_doc.get("article_number", ""),
                            "legal_field": source_doc.get("legal_field", ""),
                            "knowledge_type": source_doc.get("knowledge_type", ""),
                        }
                    )
            logger.info("RAG rerank completed in %.2fs", time.time() - rerank_start)
        else:
            # 不启用重排序时直接取前K个
            reranked = hybrid_results[:target_top_k]

        # 6. 结果过滤：按相似度阈值过滤低质量结果
        filtered = [r for r in reranked if r["score"] >= self.similarity_threshold]

        # 如果全部被过滤掉，保底返回最多2个结果
        if not filtered and reranked:
            filtered = reranked[: min(target_top_k, 2)]

        # 7. 补足结果数量：确保至少有2个结果
        min_target = min(target_top_k, 2)
        if len(filtered) < min_target and reranked:
            seen_keys = {str(item.get("id", item.get("document_title", ""))) for item in filtered}
            for item in reranked:
                if len(filtered) >= min_target:
                    break
                key = str(item.get("id", item.get("document_title", "")))
                if key in seen_keys:
                    continue
                # 确保法律领域匹配
                if legal_field and item.get("legal_field") not in ("", legal_field):
                    continue
                filtered.append(item)
                seen_keys.add(key)

        # 8. 记录检索耗时和结果数量
        elapsed = time.time() - start_time
        logger.info("RAG retrieval completed in %.2fs, found %s results", elapsed, len(filtered))
        return filtered

    def close(self) -> None:
        """清理资源：关闭线程池和数据库连接"""
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=False, cancel_futures=True)  # 立即关闭线程池
        self.milvus.close()
        self._closed = True

    def _rewrite_query(self, query: str) -> str:
        """查询改写/扩展：增强法律语义，提高检索召回率"""
        """
        查询改写/扩展：增强法律语义，提高检索召回率
        核心功能：将用户的日常口语表达，扩展为法律专业术语和关键词
        例如："公司开除我怎么办" → "公司开除我怎么办 解除劳动合同 辞退 违法解除 经济补偿 法律规定"
        """
        # ========== 第1步：基础清理 ==========
        query = (query or "").strip()  # 处理空值并去除首尾空格
        if not query:
            return ""  # 空查询直接返回空字符串

        # ========== 第2步：法律同义词映射 ==========
        # 法律同义词映射（将日常用语转换为法律专业术语）
        legal_synonyms = {
            "离婚": ["婚姻关系解除", "协议离婚", "诉讼离婚", "感情破裂"],
            "赔钱": ["损害赔偿", "违约金", "经济补偿", "赔偿金额"],
            "被抓": ["刑事拘留", "逮捕", "强制措施", "刑事立案"],
            "欠钱": ["债务纠纷", "民间借贷", "债权债务", "欠款"],
            "开除": ["解除劳动合同", "辞退", "解雇", "违法解除"],
            "工伤": ["工伤认定", "职业伤害", "工伤保险", "劳动能力鉴定"],
            "合同无效": ["合同效力", "无效合同", "可撤销合同", "合同解除"],
            "房产": ["房屋买卖", "不动产登记", "产权", "房产证"],
            "继承": ["法定继承", "遗嘱继承", "遗产分割", "继承人"],
        }

        # 添加同义词扩展，只扩展用户查询中匹配到的第一个法律关键词，然后就停止检查后面的关键词。
        rewritten = query  # 保留用户的原始查询
        for term, synonyms in legal_synonyms.items():  # items（）返回字典中的键值对
            if term in query:
                rewritten = f"{rewritten} {' '.join(synonyms)}"  # 将同义词列表转换成字符串，并追加到原查询后面。
                break  # 只匹配第一个关键词，避免过度扩展

        # 针对特定查询模式的扩展，根据查询中的特定模式或关键词，追加相应的法律专业术语，进一步提高检索的准确性
        if "怎么办" in query or "怎么" in query:
            rewritten += " 法律规定 法律责任 法律后果"
        if "赔偿" in query:
            rewritten += " 赔偿标准 赔偿金额 赔偿范围"

        # 劳动法领域专门扩展，专门用于识别和处理与劳动法律相关的用户问题
        labor_markers = ("劳动合同", "劳动者", "用人单位", "违法解除", "解除劳动合同", "辞退", "开除")
        if any(marker in query for marker in labor_markers):
            rewritten += " 劳动合同法 违法解除 非法辞退 继续履行 赔偿金 经济补偿 用人单位 劳动争议"

        # 违约金领域专门扩展
        liquidated_damage_markers = ("违约金", "约定过高", "过高", "调整违约金")
        if any(marker in query for marker in liquidated_damage_markers):
            rewritten += " 违约金 违约金过高 法院调整 实际损失 过分高于损失 仲裁机构"

        # 夫妻债务领域专门扩展
        marital_debt_markers = ("夫妻", "共同偿还", "共同债务", "婚姻关系存续期间", "借款")
        if any(marker in query for marker in marital_debt_markers):
            rewritten += " 夫妻共同债务 家庭日常生活 共同签名 追认 借款用途 婚姻关系"

        # 限制长度
        rewritten = rewritten.strip()  # 去掉字符串的开头和结尾的空格
        if len(rewritten) > settings.QUERY_REWRITE_MAX_CHARS:  # 如果超过限制，就截取前 N 个字符，N=120
            rewritten = rewritten[:settings.QUERY_REWRITE_MAX_CHARS]
        return rewritten   # 只返回前N个字符

    def format_context(self, results: List[Dict]) -> str:
        """格式化检索结果为LLM可用的上下文文本"""
        if not results:
            return "未检索到相关法律知识。"

        context_parts = []
        total_chars = 0
        for i, r in enumerate(results, 1):
            source = r.get("source", "未知来源")
            article = r.get("article_number", "")
            content = r.get("content", "") or ""

            # 控制总字符数，避免超过LLM上下文限制
            remaining = settings.RAG_CONTEXT_MAX_CHARS - total_chars
            if remaining <= 0:
                break
            content = content[:remaining]
            total_chars += len(content)

            # 构造引用格式：[参考1] 中华人民共和国民法典 第1043条\n内容\n
            part = f"[参考{i}] {source}"
            if article:
                part += f" {article}"
            part += f"\n{content}\n"
            context_parts.append(part)

        return "\n".join(context_parts)
