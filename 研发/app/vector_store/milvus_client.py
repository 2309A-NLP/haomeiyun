'''
在RAG（检索增强生成）系统中，初检（如向量检索）可能召回100条相关文档，
然后在用重排序模型会对这100条文档重新计算相关性分数，挑出最相关的top_k条（比如5条）送给LLM
'''
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusException,
        connections,
        utility,
    )
    _PYMILVUS_IMPORT_ERROR = None
except ImportError as exc:
    Collection = Any
    CollectionSchema = Any
    DataType = Any
    FieldSchema = Any
    MilvusException = Exception
    connections = None
    utility = None
    _PYMILVUS_IMPORT_ERROR = exc
from ..core.config import settings
from ..utils.logger import logger
from .bm25_store import BM25Store

EXPECTED_COLLECTION_FIELDS = {
    "id",
    "dense_vector",
    "content",
    "source",
    "article_number",
    "legal_field",
    "knowledge_type",
    "document_title",
}


class MilvusClient:
    """稠密向量存储（Dense Vector Store）+ 同步的BM25词法索引"""
    _instance = None  # 单例实例

    def __new__(cls):
        """
        单例模式实现
               确保整个应用只有一个Milvus客户端实例，
               避免重复连接和资源浪费。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._connected = False
        return cls._instance
    '''
    作用：确保全局只有一个BGEReranker实例
        为什么需要单例？
        模型加载很慢（几百MB到几GB）
        避免重复加载浪费内存和时间
        多个请求共享同一个模型
    '''

    def __init__(self):
        """初始化Milvus客户端
        流程：
        1. 检查是否已初始化（单例）
        2. 读取Milvus配置（host/port/collection）
        3. 延迟连接（lazy loading）
        4. 标记为已初始化
        """
        if self._initialized:  # 单例模式中防止重复初始化的关键保护机制
            return   # 已经初始化过，直接返回

        self.host = settings.MILVUS_HOST  # Milvus 服务器地址，如 "localhost" 或 "192.168.1.100"
        self.port = settings.MILVUS_PORT  # 端口，默认 19530
        self.collection_name = settings.MILVUS_COLLECTION  # 集合名称（类似 MySQL 的表名）
        self.collection: Optional[Collection] = None  # 初始为 None，稍后连接成功会赋值
        self.bm25_store = BM25Store()  # BM25 词法索引存储
        self._bm25_bootstrapped = False  # BM25 是否已初始化（启动标记）
        self._initialized = True  # 标记这个 Milvus 客户端对象已经准备好可以使用了

    # 确保数据库连接的私有方法，采用了懒加载模式
    def _ensure_connection(self, bootstrap_bm25: bool = True):  # 在使用 Milvus 向量数据库前，确保连接已建立且环境已就绪
        if _PYMILVUS_IMPORT_ERROR is not None:  # 检索依赖是否安装
            raise RuntimeError(  # 告诉用户pymilvus没安装，需要安装依赖
                "pymilvus is not installed. Install the vector-store dependencies before using Milvus features."
            ) from _PYMILVUS_IMPORT_ERROR   # 异常链：将新异常与原始异常关联起来

        if self._connected:  # 确保连接就绪，检查连接状态：self._connected 是一个布尔值，表示是否已与 Milvus 建立连接
            # 如果需要引导BM25且尚未完成
            # True = 已经连接成功，False = 尚未链接或连接已断开
            if bootstrap_bm25 and not self._bm25_bootstrapped:
                # 两个条件同时满足时才执行：
                #      条件	                             含义
                # bootstrap_bm25	                调用方要求自动构建 BM25 索引
                # not self._bm25_bootstrapped	    BM25 索引尚未构建（False）
                self._bootstrap_bm25_from_milvus()  # 从 Milvus 构建 BM25 索引，读取 Milvus 中存储的文档，建立关键词检索索引
            return
        '''
        调用 _ensure_connection(bootstrap_bm25=True)
                ↓
        检查 pymilvus 是否安装？
                ↓ 未安装                 ↓ 已安装
            抛出 RuntimeError      检查 self._connected？
                程序终止                ↓ False             ↓ True
                     建立连接...    需要 BM25 且未构建？
                         ↓              ↓ 是               ↓ 否
                     设置标志位      构建 BM25             直接返回
        '''

        try:
            self._connect()  # 建立与 Milvus 服务器的连接
            self._ensure_collection()  # 确保所需的集合（类似SQL表）存在
            self._connected = True     # 标记连接成功
            if bootstrap_bm25:
                self._bootstrap_bm25_from_milvus()  # 从 Milvus 初始化 BM25 算法相关数据
        except Exception as exc:
            logger.error("Failed to initialize Milvus: %s", exc)
            raise

    def _connect(self):
        try:
            try:
                connections.disconnect("default")
            except Exception:
                pass

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=10,
            )
            version = utility.get_server_version()
            logger.info("Connected to Milvus at %s:%s, version: %s", self.host, self.port, version)
        except Exception as exc:
            logger.error("Failed to connect to Milvus: %s", exc)
            raise

    def _ensure_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            if self._should_rebuild_collection(self.collection):
                logger.warning(
                    "Collection '%s' uses an outdated empty schema; rebuilding it to match the current embedding setup",
                    self.collection_name,
                )
                self.rebuild_collection()
                return
            self.collection.load()
            logger.info("Collection '%s' loaded", self.collection_name)
        else:
            self._create_collection()
            logger.info("Collection '%s' created", self.collection_name)

    def _should_rebuild_collection(self, collection: Collection) -> bool:
        field_names = {field.name for field in collection.schema.fields}
        dense_vector_field = next(
            (field for field in collection.schema.fields if field.name == "dense_vector"),
            None,
        )
        dense_dim = int((getattr(dense_vector_field, "params", {}) or {}).get("dim", 0))
        missing_fields = EXPECTED_COLLECTION_FIELDS - field_names
        unexpected_schema = dense_dim != settings.EMBEDDING_DIMENSION or bool(missing_fields)
        if not unexpected_schema:
            return False

        entity_count = collection.num_entities
        if entity_count == 0:
            return True

        raise RuntimeError(
            "Milvus collection schema is incompatible with the current embedding configuration. "
            f"Expected dense_vector dim={settings.EMBEDDING_DIMENSION}, got dim={dense_dim}, "
            f"missing_fields={sorted(missing_fields)}, num_entities={entity_count}."
        )

    def _create_collection(self):
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        dense_vector_field = FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.EMBEDDING_DIMENSION,
        )
        content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500)
        article_number_field = FieldSchema(name="article_number", dtype=DataType.VARCHAR, max_length=100)
        legal_field_field = FieldSchema(name="legal_field", dtype=DataType.VARCHAR, max_length=50)
        knowledge_type_field = FieldSchema(name="knowledge_type", dtype=DataType.VARCHAR, max_length=50)
        document_title_field = FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=500)

        schema = CollectionSchema(
            fields=[
                id_field,
                dense_vector_field,
                content_field,
                source_field,
                article_number_field,
                legal_field_field,
                knowledge_type_field,
                document_title_field,
            ],
            description="Legal Knowledge Base Collection",
        )

        self.collection = Collection(name=self.collection_name, schema=schema)
        self._create_indexes()
        self.collection.load()

    def _create_indexes(self):
        self.collection.create_index(
            field_name="dense_vector",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        logger.info("Dense vector index created with metric=%s", "COSINE")

    # 用于从 Milvus 向量数据库中批量读取文档数据，为构建 BM25 索引做准备
    def _bootstrap_bm25_from_milvus(self):
        # 1.前置检查：如果已经构建过或集合不存在，直接返回
        if self._bm25_bootstrapped or self.collection is None:
            #      条件	                     含义	               行为
            # self._bm25_bootstrapped	BM25 索引已经构建过	直接返回，避免重复构建
            # self.collection is None	Milvus 集合未初始化	直接返回，无法读取数据
            # 设计意图：幂等性 - 多次调用不会产生副作用
            return
        # 初始化变量
        iterator = None
        documents: List[Dict] = []   # 用于存储所有文档
        try:
            iterator = self.collection.query_iterator(
                batch_size=1000,    # 每批1000条
                limit=-1,           # 不限制总数（-1表示全部）
                expr="id >= 0",     # 查询条件：id大于等于0（获取所有）
                output_fields=[     # 指定要返回的字段
                    "id",
                    "content",
                    "source",
                    "article_number",
                    "legal_field",
                    "knowledge_type",
                    "document_title",
                ],
            )
            # 批量读取数据的核心循环，负责从 Milvus 迭代器中逐批获取文档并转换为标准格式
            while True:    # 无限循环，直到数据读完
                batch = iterator.next()   # 获取一批数据（1000条）
                if not batch:   # 没有更多数据时
                    break       # 跳出循环
                    # 执行逻辑：
                        # 每次循环调用 iterator.next() 获取下一批数据
                        # 如果 batch 为空（None 或空列表），说明没有更多数据
                        # break 退出循环，结束读取
                    # 为什么用 while True？
                        # 因为不知道总共有多少批数据
                        # 直到读取完所有数据才退出
                for item in batch:   # 遍历当前批次中的每条文档
                    documents.append(   # 添加到结果列表
                        {
                            "id": item.get("id"),  # 文档唯一标识,用途：后续去重、缓存、引用
                            "content": item.get("content", ""),  # 文档正文内容，
                            "source": item.get("source", ""),  # 来源文件/链接，用户可查阅原始文件
                            "article_number": item.get("article_number", ""),  # 法条编号
                            "legal_field": item.get("legal_field", ""),  # 法律领域（如：劳动法）
                            "knowledge_type": item.get("knowledge_type", ""),  # 知识类型（如：法条/案例）
                            "document_title": item.get("document_title", ""),  # 文档标题
                        }
                    )
        except Exception as exc:  # 包括：网络超时、解析错误、内存不足等
            logger.warning("Failed to bootstrap BM25 index from Milvus: %s", exc)  # 日志
            return
        finally:  # 无论是否异常都会执行
            if iterator is not None:  # 保护性检查：如果迭代器创建失败，iterator 保持为 None
                try:
                    iterator.close()  # 调用 None.close() 会抛出 AttributeError
                except Exception:
                    pass
        # 为什么关闭失败也要吞掉异常？
            # finally 块中的异常会覆盖原始异常
            # 关闭失败不是关键问题，不应中断流程
            # pass 表示完全忽略关闭时的错误

        if documents:   # 如果有文档数据
            self.bm25_store.upsert_documents(documents)   # 插入/更新文档到 BM25 索引
            logger.info("Bootstrapped BM25 index from %s Milvus documents", len(documents))   # 日志
        self._bm25_bootstrapped = True   # 标记 BM25 已构建完成

    def insert(
        self,
        dense_vectors: List[List[float]],
        contents: List[str],
        sources: List[str],
        article_numbers: List[str] = None,
        legal_fields: List[str] = None,
        knowledge_types: List[str] = None,
        document_titles: List[str] = None,
        flush: bool = True,
    ) -> List[int]:
        self._ensure_connection(bootstrap_bm25=False)

        if not dense_vectors:
            return []

        num_rows = len(dense_vectors)
        article_numbers = article_numbers or [""] * num_rows
        legal_fields = legal_fields or ["general"] * num_rows
        knowledge_types = knowledge_types or ["law"] * num_rows
        document_titles = document_titles or [""] * num_rows

        try:
            batch_size = max(1, settings.MILVUS_INSERT_BATCH_SIZE)
            primary_keys: List[int] = []
            logger.info("Starting Milvus insert: rows=%s batch_size=%s", num_rows, batch_size)

            for start in range(0, num_rows, batch_size):
                end = min(start + batch_size, num_rows)
                data = [
                    dense_vectors[start:end],
                    contents[start:end],
                    sources[start:end],
                    article_numbers[start:end],
                    legal_fields[start:end],
                    knowledge_types[start:end],
                    document_titles[start:end],
                ]
                insert_result = self.collection.insert(data)
                batch_primary_keys = list(insert_result.primary_keys)
                primary_keys.extend(batch_primary_keys)

                self._sync_bm25_documents(
                    primary_keys=batch_primary_keys,
                    contents=contents[start:end],
                    sources=sources[start:end],
                    article_numbers=article_numbers[start:end],
                    legal_fields=legal_fields[start:end],
                    knowledge_types=knowledge_types[start:end],
                    document_titles=document_titles[start:end],
                )
                logger.info("Milvus insert progress: %s/%s", end, num_rows)

            if flush:
                self.collection.flush()
                logger.info("Milvus flush completed: rows=%s", num_rows)

            return primary_keys
        except MilvusException as exc:
            logger.error("Insert failed: %s", exc)
            raise

    def _sync_bm25_documents(
        self,
        *,
        primary_keys: Iterable[int],
        contents: List[str],
        sources: List[str],
        article_numbers: List[str],
        legal_fields: List[str],
        knowledge_types: List[str],
        document_titles: List[str],
    ) -> None:
        documents = []
        for index, primary_key in enumerate(primary_keys):
            documents.append(
                {
                    "id": primary_key,
                    "content": contents[index],
                    "source": sources[index],
                    "article_number": article_numbers[index],
                    "legal_field": legal_fields[index],
                    "knowledge_type": knowledge_types[index],
                    "document_title": document_titles[index],
                }
            )
        if documents:
            self.bm25_store.upsert_documents(documents)

    # 密集向量检索的具体实现，负责在Milvus向量数据库中执行语义搜索
    # 根据语义向量，在向量数据库中查找最相似的法律条文
    def dense_search(
        self,
        vector: List[float],   # 查询向量（768维的语义向量）
        top_k: int = 10,       # 返回最相似的top-k条结果，默认值10
        filter_expr: Optional[str] = None,  # milvus的过滤表达式
    ) -> List[Dict]:    # 返回检索结果列表，每个结果包含文档内容和元数据
        self._ensure_connection()   # 检查是否已经连接到Milvus数据库
        if filter_expr and not self._has_matching_documents(filter_expr):
            # 如果有法律领域的milvus的过滤条件，但没有匹配文档
            '''
            真值表
            filter_expr	    has_matching	 condition结果	执行动作
            None (无过滤)	-	             False	        执行检索
            有过滤	        True (有文档)	 False	        执行检索
            有过滤	        False (无文档)	 True	        跳过检索
            '''
            logger.info("Dense search skipped because no Milvus documents matched filter: %s", filter_expr)
            # 日志
            return []
        try:
            # 密集向量检索的核心执行部分，负责在Milvus数据库中执行实际的向量相似度搜索
            results = self.collection.search(
                data=[vector],          # 查询向量，为什么用列表，milvus支持批量搜索多个向量
                anns_field="dense_vector",   # 指定在哪个向量字段上进行搜索，anns_field="dense_vector" 表示用这个字段做相似度计算
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},   # 余弦相似度检索
                # metric_type="COSINE": 使用余弦相似度计算
                    #   - 计算两个向量夹角的余弦值
                    #   - 值越接近1越相似
                    #   - 公式：cos(θ) = (A·B)/(|A|·|B|)

                # nprobe=16: 搜索精度参数
                    #   - 控制搜索多少个聚类（近似搜索）
                    #   - 16是平衡速度和精度的常用值
                limit=top_k,  # 返回前 top_k 条最相似的结果,例如 top_k=10，返回10条最相关的法律条文
                expr=filter_expr,  # 过滤条件，先过滤再搜索
                output_fields=["content", "source", "article_number", "legal_field", "knowledge_type", "document_title"],
                # 指定返回那些字段
            )
            # ============= 格式化结果 ===============
            # 对Milvus返回的原始结果进行格式化处理，并添加检索类型标记
            # 作用：将Milvus返回的原始数据转换成统一的格式，并标记这些结果来自密集检索
            formatted = self._format_results(results[0])  # 为什么返回第一个列表，因为输入的就是列表套列表，输出的也是列表套列表
            # ============== 添加检索=====================
            for item in formatted:   # 遍历格式化后的检索结果
                item["retrieval_type"] = "dense"  # 标记检索类型为稠密向量检索
            return formatted  # 返回带标记的检索结果
        except MilvusException as exc:  # 捕获Milvus数据库异常
            logger.error("Dense search failed: %s", exc)  # 日志
            return []

    # 检查Milvus数据库中是否存在满足过滤条件的文档，用于在正式检索前做快速验证
    # 判断是否有任何文档匹配给定的过滤表达式，避免无效的向量检索
    def _has_matching_documents(self, filter_expr: str) -> bool:
        try:
            rows = self.collection.query(
                expr=filter_expr,  # 过滤表达式（如 'legal_field == "劳动法"'）
                limit=1,  # 只要1条，确认存在就行
                output_fields=["id"],  # 只返回id字段，减少数据传输
            )   # 查询是否存在至少一条满足条件的文档
            return bool(rows)
            # rows 是列表
            # 如果找到文档：rows = [{"id": 123}] → bool(rows) = True
            # 如果没找到：rows = [] → bool(rows) = False
        except Exception as exc:
            logger.warning("Failed to pre-check Milvus filter '%s': %s", filter_expr, exc)  # 日志
            return True  # 让正式检索去处理，避免因为预检查失败而漏掉结果


    def bm25_search(   # BM25检索的封装层
        self,  # 实例方法
        query: str,  # 查询字符串（必填）
        top_k: int = 10,  # 返回最多10条结果（默认值）
        legal_field: Optional[str] = None,  # 法律领域过滤（可选）
        knowledge_type: Optional[str] = None,  # 知识类型过滤（可选）
    ) -> List[Dict]:  # 返回字典列表
        self._ensure_connection()  # 确保数据库连接
        return self.bm25_store.search(  # 调用底层存储的搜索方法
            query=query,
            top_k=top_k,
            legal_field=legal_field,
            knowledge_type=knowledge_type,
        )

    def delete_by_filter(self, filter_expr: str) -> int:
        self._ensure_connection()
        deleted_titles = self._query_document_titles(filter_expr)
        try:
            result = self.collection.delete(filter_expr)
            self.collection.flush()
        except MilvusException as exc:
            logger.error("Delete failed: %s", exc)
            return 0

        for title in deleted_titles:
            self.bm25_store.delete_by_document_title(title)
        return result.delete_count

    def _query_document_titles(self, filter_expr: str) -> List[str]:
        try:
            rows = self.collection.query(
                expr=filter_expr,
                output_fields=["document_title"],
            )
        except Exception:
            return []
        return [row.get("document_title", "") for row in rows if row.get("document_title")]

    def rebuild_collection(self) -> None:
        if _PYMILVUS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "pymilvus is not installed. Install the vector-store dependencies before using Milvus features."
            ) from _PYMILVUS_IMPORT_ERROR

        try:
            self._connect()
            try:
                if self.collection is not None:
                    self.collection.release()
            except Exception:
                pass

            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info("Collection '%s' dropped", self.collection_name)

            self.collection = None
            self._bm25_bootstrapped = False
            self.bm25_store.clear()
            self._create_collection()
            self._connected = True
            logger.info("Collection '%s' rebuilt", self.collection_name)
        except Exception as exc:
            logger.error("Failed to rebuild collection: %s", exc)
            raise

    def get_collection_stats(self) -> Dict:
        self._ensure_connection()
        try:
            self.collection.flush()
            return {
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "indexes": [idx.params for idx in self.collection.indexes],
                "bm25": self.bm25_store.stats(),
            }
        except MilvusException as exc:
            logger.error("Get stats failed: %s", exc)
            return {}

    def close(self):
        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")
        except Exception as exc:
            logger.error("Disconnect failed: %s", exc)

    @staticmethod
    def _rrf_fusion(primary_results: List[Dict], secondary_results: List[Dict], k: int = 60) -> List[Dict]:
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(primary_results):
            doc_id = doc.get("id", f"primary_{rank}")
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(secondary_results):
            doc_id = doc.get("id", f"secondary_{rank}")
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
        fused = []
        for doc_id in sorted_ids:
            item = doc_map[doc_id].copy()
            item["fusion_score"] = scores[doc_id]
            fused.append(item)
        return fused

    # 实现了混合检索的核心融合逻辑，将密集向量检索（语义）和BM25检索（关键词）的结果通过RRF算法合并，最终返回最优的top_k条结果
    # 混合检索 = 密集检索 + BM25检索 + RRF融合
    def hybrid_search(
        self,
        dense_vector: List[float],  # 768维语义向量
        query: str,                 # 用户的问题
        top_k: int = 10,            # 返回的用户的文本数量
        filter_expr: Optional[str] = None,  # Milvus过滤表达式（如：'legal_field == "劳动法"'）
        legal_field: Optional[str] = None,  # 法律领域（如："劳动法"、"婚姻法"、"刑法"）
        knowledge_type: Optional[str] = None,  # 知识类型（如："law"法律条文、"case"案例、"regulation"法规）
    ) -> List[Dict]:     # 返回检索结果列表
        # 实现一个稠密向量检索（Dense Retrieval）的过程，常见于RAG（检索增强生成）、向量数据库或语义搜索系统中
        dense_results = self.dense_search(dense_vector, top_k=top_k * 2, filter_expr=filter_expr)
        '''
        组成部分	                     含义
        self.dense_search()	         调用对象自身的稠密向量检索方法
        dense_vector	             查询的稠密向量（通常是 embedding 模型输出的浮点数数组）
        top_k=top_k * 2	             请求返回 2 倍于最终需要的结果数量
        filter_expr=filter_expr	     元数据过滤条件（如 "category = 'tech'"）
        
        请求 top_k * 2 个结果 → 后续可能进行重排序（rerank） → 筛选后保留最终 top_k 个
        典型流程：
            稠密检索快速召回 2K 个候选（宽召回）
            用更精确但更慢的模型（如交叉编码器 Cross-Encoder）重排序
            取前 K 个作为最终结果
            这样可以兼顾召回率和效率
        '''
        # 这行代码的核心作用是：通过 BM25 关键词检索，从指定法律领域和知识类型中，召回比最终需求多一倍的候选文档，为后续的混合检索或精排做准备。
        bm25_results = self.bm25_search(
            query=query,   # 查询文本，用户的问题或搜索词
            top_k=top_k * 2,   # 返回结果数量（翻倍）
            # 返回结果数量：请求返回 top_k * 2 条结果
            # 为什么要翻倍？常见原因：
                # 混合检索策略：后续会与 Dense 检索结果融合，取交集或重新排序
                # 召回更多候选：先用 BM25 召回更多，再精排筛选
                # 互补性：BM25 擅长关键词匹配，Dense 擅长语义理解，两者结合效果更好
            legal_field=legal_field,  # 法律领域过滤（如：劳动法）,作用：提高检索精准度，避免跨领域干扰
            knowledge_type=knowledge_type,  # 知识类型过滤（如：法条/案例/解释）
        )
        fused = self._rrf_fusion(dense_results, bm25_results, k=60)
        return fused[:top_k]
    # 将Milvus返回的原始搜索结果转换为统一的字典格式，方便后续处理和使用。
    # 作用：将Milvus的搜索结果对象（hits）转换成标准化的Python字典列表
    @staticmethod   # 静态方法装饰器
    # 表示这是一个静态方法
    # 不需要 self 参数，可以直接通过类名调用
    # 不需要访问实例属性
    def _format_results(hits) -> List[Dict]:
        # hits: Milvus返回的搜索结果对象（可迭代的对象集合）
        # -> List[Dict]: 返回值类型提示，表示返回字典列表
        results = []   # 创建一个空列表，用来存放转换后的结果
        for hit in hits:  # 循环遍历每个搜索结果，单个搜索结果，代表一个匹配的文档
            results.append(
                {
                    "id": hit.id,  # 文档ID
                    "score": hit.score,  # 相似度分数
                    "content": hit.entity.get("content", ""),  # 文档内容
                    "source": hit.entity.get("source", ""),  # 来源
                    "article_number": hit.entity.get("article_number", ""),  # 法条编号
                    "legal_field": hit.entity.get("legal_field", ""),  # 法律领域
                    "knowledge_type": hit.entity.get("knowledge_type", ""),  # 知识类型
                    "document_title": hit.entity.get("document_title", ""),  # 文档标题
                }
            )
        return results
