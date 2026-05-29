'''
这是一个内存版的法律文档搜索引擎，能够根据用户输入的关键词，快速找到最相关的法律条文、案例等文档，并支持按法律领域（如劳动法、民法）和知识类型（法条/案例）进行筛选。
'''
import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

from ..utils.logger import logger

# BM25 索引的中文分词器，负责将原始文本切分成有意义的词单元（tokens）
def _tokenize(text: str) -> List[str]:
    normalized = (text or "").lower()  # 1. 转小写
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)  # 2. 去除非文字字符，将非字母、非数字、非汉字、非下划线的字符替换为空格
    pieces = [part.strip() for part in normalized.split() if part.strip()]  # 3. 按空格切分
    '''
    re.fullmatch：要求整个字符串完全匹配
        为什么中文要按字切分？
        中文词之间没有空格
        简单分词策略：按单个汉字切分
        例如："劳动合同法" → ["劳", "动", "合", "同", "法"]
    '''

    tokens: List[str] = []
    for piece in pieces:
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):  # 4. 判断是否为纯中文
            tokens.extend(list(piece))  # 5. 中文按字切分
        else:
            tokens.append(piece)  # 6. 非中文保持原样
    return tokens


class BM25Index:
    """In-memory BM25 index for lexical retrieval over knowledge chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # BM25 参数 k1,控制词频（TF）对分数的影响程度
        '''
        k1 的影响：
            k1 值	       效果	                              适用场景
            小（如 0.5）	   词频影响小，出现1次和10次分数差不多	  短文本、标题
            大（如 2.0）	   词频影响大，高频词分数更高	          长文本、正文
            默认1.5	       平衡选择	                          通用场景
        '''
        self.b = b  # BM25 参数 b,控制文档长度对分数的影响程度
        '''
        b 的影响：
            b 值	       效果	                          适用场景
            b=0	           忽略长度，长文档短文档一视同仁	      文档长度差异小
            b=1	           完全归一化，严格按长度调整	      文档长度差异大
            默认 0.75	   部分归一化，平衡选择	              通用场景
        '''
        self._lock = threading.RLock()  # 可重入锁,同一线程可多次获取   Lock  同一线程只能获取一次
        self._documents: Dict[str, Dict] = {}  # 文档存储
        self._term_doc_freq: Dict[str, int] = defaultdict(int)  # 词频统计
        self._total_length = 0  # 总词数
        self._version = 0  # 索引版本
        '''
        总结
            这个构造函数完成了三件事：
            参数配置：设置 BM25 算法的 k1 和 b 参数
            锁初始化：创建可重入锁保证线程安全
            数据结构准备：初始化存储文档、词频、长度、版本的数据结构
        设计特点：
            可配置：BM25 参数可由用户调整
            线程安全：使用 RLock 支持并发操作
            类型安全：完整的类型注解
            可扩展：预留版本号支持缓存
        BM25 参数含义：
            k1：控制词频饱和度（越大，高频词权重越高）
            b：控制长度归一化（越大，长文档惩罚越重）
        '''
    # 实现了真正的BM25索引构建的逻辑
    def upsert_documents(self, documents: Iterable[Dict]) -> int:
        count = 0         # 原始的文本数量为0
        with self._lock:  # 线性锁，保证线程安全
        # 保护整个索引操作，防止多线程同时修改导致数据损坏
            for document in documents:   # 遍历每个文档
                doc_id = str(document.get("id", "")).strip()   # 文本ID，不能为空，去除前后空格
                content = (document.get("content") or "").strip()  # 文本，不能为空，去除前后空格
                if not doc_id or not content:   # 如果没有这两条件就跳过
                    continue

                if doc_id in self._documents:   # 检查文档ID是否已存在
                    self._remove_document_locked(doc_id)  # 删除旧版本
                    '''
                    为什么先删除再插入？
                    这种方式实现了 upsert 语义：
                    场景	        操作	               结果
                    文档不存在	跳过删除，直接插入	   新增文档
                    文档已存在	删除旧的，插入新的	   完全替换
                    '''

                tokens = _tokenize(content)  # 对内容进行分词
                if not tokens:  # 如果分词结果为空
                    continue   # 跳过当前文档

                token_counts = Counter(tokens)    # 统计当前文档中每个词的出现次数（词频）
                unique_terms = set(token_counts.keys())   # 提取当前文档中的唯一词（自动去重）
                for term in unique_terms:        # 循环遍历，又有一个新文档包含了这个词
                    self._term_doc_freq[term] += 1   # 将文档频率加1

                # 存储文档的核心部分，负责将处理后的文档数据保存到索引中
                self._documents[doc_id] = {   # 存储到文档字典
                    "doc": {                  # 原文档
                        "id": document.get("id"),
                        "content": content,
                        "source": document.get("source", ""),
                        "article_number": document.get("article_number", ""),
                        "legal_field": document.get("legal_field", ""),
                        "knowledge_type": document.get("knowledge_type", ""),
                        "document_title": document.get("document_title", ""),
                    },
                    "terms": token_counts,  # 词频统计（用于 BM25 计算）
                    "length": len(tokens),  # 文档长度（用于 BM25 计算）
                }
                self._total_length += len(tokens)     # 更新总长度
                count += 1         # 成功计数加1

            if count:       # 如果有文档被成功插入/更新
                self._version += 1    # 版本号加1
        return count        # 返回成功处理的文档数量

    # BM25 索引的按文档标题删除方法，用于删除所有匹配特定标题的文档
    def delete_by_document_title(self, document_title: str) -> int:
        removed = 0        # 记录成功删除的文档数量
        with self._lock:   # 使用可重入锁保护整个操作,保证了删除操作的原子性
            targets = [
                doc_id
                for doc_id, payload in self._documents.items()   # 遍历原数据，_documents.items()键值对视图，每个元素是 (doc_id, payload)
                if payload["doc"].get("document_title", "") == document_title  # 过滤条件,只保留标题匹配的文档
            ]
            '''
            _documents = {
                "doc_001": {"doc": {"document_title": "劳动合同法"}, "terms": {...}},
                "doc_002": {"doc": {"document_title": "劳动合同法"}, "terms": {...}},
                "doc_003": {"doc": {"document_title": "民法典"}, "terms": {...}}
            }
            
            list(_documents.items()) = [
                ("doc_001", {"doc": {"document_title": "劳动合同法"}, ...}),
                ("doc_002", {"doc": {"document_title": "劳动合同法"}, ...}),
                ("doc_003", {"doc": {"document_title": "民法典"}, ...})
            ]
            '''
            for doc_id in targets:  # 遍历所有要删除的文档ID
                self._remove_document_locked(doc_id)   # 调用内部删除ID方法
                removed += 1
            if removed:     # 如果文档被删除了
                self._version += 1
        return removed    # 返回删除数量
    '''
    版本号是为了让系统能够检测到数据变化，从而保证缓存的有效性、数据的一致性和操作的可靠性。没有版本号，系统就无法知道什么时候该刷新缓存，用户可能看到过期数据。
    '''
    # BM25索引的清空方法，用于删除索引中的所有数据
    def clear(self) -> None:
        with self._lock:   # 确保清空操作的原子性，防止在清空过程中有其他线程读写索引
            self._documents.clear()   # 清空文档存储字典
            self._term_doc_freq.clear()  # 清空词频统计字典
            self._total_length = 0    # 重置总长度
            self._version += 1        # 版本号加1

    # 负责对查询进行预训练和验证
    def search(
        self,
        query: str,  # 查询字符串
        top_k: int = 10,  # 返回前K个结果
        legal_field: Optional[str] = None,  # 法律领域过滤
        knowledge_type: Optional[str] = None,  # 知识类型过滤
    ) -> List[Dict]:  # 返回文档列表
        query_terms = _tokenize(query)  # 将用户输入的查询字符串转换为词列表
        if not query_terms:  # 检查分词结果
            return []  # 空查询返回空列表，没有有效的搜索词，无法进行检索

        with self._lock:   # 获取线程锁，确保读取 _documents 时不会与其他修改操作冲突
            doc_count = len(self._documents)      # 获取索引中的文档数
            if doc_count == 0:       # 是否为空
                return []

            # 核心评分部分，负责计算每个文档的 BM25 分数并应用过滤器
            avgdl = self._total_length / doc_count if doc_count else 0.0   # 计算索引中文档的平均长度
            '''
            逻辑：
                如果 doc_count > 0：avgdl = _total_length / doc_count
                如果 doc_count == 0：avgdl = 0.0（避免除零错误）
            '''
            ranked: List[Dict] = []    # 存储评分结果的列表
            for payload in self._documents.values():
                doc = payload["doc"]   # 读取文档doc
                if legal_field and doc.get("legal_field") != legal_field:  # 法律邻域过滤
                    continue
                if knowledge_type and doc.get("knowledge_type") != knowledge_type:  # 知识邻域的过滤
                    continue

                score = self._score_document(query_terms, payload["terms"], payload["length"], avgdl, doc_count)
                if score <= 0:      # 分数<=0的文档不加入结果
                    continue
                '''
                为什么过滤 score <= 0？
                    BM25 分数理论上可以接近 0
                    分数为 0 或负数的文档与查询不相关
                    减少后续排序的计算量
                '''
                # 收尾部分，负责收集评分结果、排序并返回 top-k 个最相关的文档
                # 收集评分结果
                ranked.append(
                    {
                        **doc,           # 展开原始文档数据
                        "score": score,  # 添加 BM25 分数
                        "retrieval_type": "bm25",   # 标记检索类型
                    }
                )

        ranked.sort(key=lambda item: item["score"], reverse=True)   # 按分数降序排序
        return ranked[:top_k]

    # BM25 索引的统计信息方法，用于返回索引的当前状态和元数据
    def stats(self) -> Dict:
        with self._lock:      #  获取线程锁
            return {
                "documents": len(self._documents),   # 文档数量
                "average_length": (self._total_length / len(self._documents)) if self._documents else 0,  # 平均文档长度
                "version": self._version,   # 索引版本
            }
    # 负责从 BM25 索引中安全删除文档并更新所有相关统计信息
    def _remove_document_locked(self, doc_id: str) -> None:
        payload = self._documents.pop(doc_id, None)  # 弹出文档数据
        '''
        pop(key, default) 方法：
            如果 doc_id 存在：删除并返回对应的值
            如果 doc_id 不存在：返回 None（不抛出异常）
            
        self._documents = {
            "doc_001": {
                "doc": {"id": "doc_001", "content": "..."},
                "terms": {"劳动": 3, "合同法": 2, "规定": 1},
                "length": 150
            },
            "doc_002": {...}
        }
        
        # 执行 pop("doc_001")
        payload = {
            "doc": {...},
            "terms": {"劳动": 3, "合同法": 2, "规定": 1},
            "length": 150
        }
        # self._documents 中不再有 "doc_001"
        '''
        if not payload:  # 验证文档是否存在，防止删除不存在的文档时出错
            return

        self._total_length -= payload["length"]   # 取出该文档的词数（分词后的数量）
        for term in payload["terms"].keys():  # payload["terms"] 是 Counter 对象，存储每个词的出现次数：
            current = self._term_doc_freq.get(term, 0) - 1   # 删除当前文档后，某个词还出现在多少个文档中
            '''
            为什么要减 1？
                因为我们要删除当前这个文档
                当前文档包含这个词
                删除后，包含该词的文档数会减少 1
            '''
            if current > 0:  # 149 > 0，说明还有其他文档包含"劳动"
                self._term_doc_freq["劳动"] = 149  # 更新为 149
            else:  # current == 0，说明这是最后一个包含"劳动"的文档
                self._term_doc_freq.pop("劳动", None)  # 删除这个词条
                """
                词频统计：Counter(tokens) 记录每个词在文档中出现的次数
                    文档频率更新：_term_doc_freq[term] += 1 记录词出现在多少文档中
                    文档存储：保存原始数据、词频、长度
                    总长度更新：_total_length += len(tokens) 用于计算平均文档长度
                设计精髓：
                    分离存储：原始数据、词频、长度分开存储
                    双向维护：插入和删除操作对称
                    高效查询：为 BM25 检索准备所有必要数据
                """

    def _score_document(
        self,
        query_terms: List[str],  # 查询词列表
        doc_terms: Counter,  # 文档的词频统计
        doc_length: int,  # 文档长度（词数）
        avgdl: float,  # 平均文档长度
        doc_count: int,  # 文档总数
    ) -> float:
        score = 0.0
        effective_avgdl = avgdl or 1.0  # 防止除零

        for term in query_terms:  # 遍历每个查询词
            term_freq = doc_terms.get(term, 0)  # 词在文档中的频率
            if term_freq <= 0:
                continue

            # 计算 IDF（逆文档频率）
            doc_freq = self._term_doc_freq.get(term, 0)
            idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

            # 计算 TF（词频）部分
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / effective_avgdl)

            # 累加分数
            score += idf * (numerator / denominator)

        return score


class BM25IndexManager:  # 单例管理器，管理共享的 BM25 索引
    """定义了一个单例模式的 BM25 索引管理器，负责管理全局共享的 BM25 索引"""

    _instance = None   # 类变量：存储唯一实例，所以实例共享这个变量

    # 单例模式，确保整个应用共享同一个BM25索引，避免重复构建和内存浪费
    def __new__(cls):
        # cls._instance 是 None
        if cls._instance is None:  # True
            # 调用父类 object 的 __new__ 创建实例
            cls._instance = super().__new__(cls)
            # 给新实例添加 _initialized 属性，设为 False
            cls._instance._initialized = False
        return cls._instance  # 返回刚创建的实例

    def __init__(self):   # 单例类的初始化方法，负责创建真正的BM25索引对象
        if self._initialized:  # 检查是否已初始化
            return             # 已初始化则直接返回，什么都不做，直接返回
        self.index = BM25Index()   # 创建真正的BM25索引，包含倒排索引、文档存储等数据结构
        self._initialized = True   # 标记为已初始化

    # 定义了 BM25 索引管理器的两个核心数据操作方法：插入/更新文档和按标题删除文档
    def upsert_documents(self, documents: Iterable[Dict]) -> int:
        count = self.index.upsert_documents(documents)  # 传入可迭代的文档字典集合，返回实际处理的文档数量
        if count:   # 如果文档存在，避免空更新产生无意义的日志
            logger.info("BM25 index updated with %s documents", count)
        return count  # 返回处理的文档数量，调用方可以知道有多少文档被更新/插入
    # 删除文档，按文档标题批量删除，删除所以匹配该标题的文档
    def delete_by_document_title(self, document_title: str) -> int:
        removed = self.index.delete_by_document_title(document_title)  # 参数：document_title - 要删除的文档标题
        # 返回值：实际删除的文档数量
        if removed:  # 只有删除成功（>0）才记录日志，日志中记录删除了多少条、哪个标题
            logger.info("BM25 index removed %s documents for title '%s'", removed, document_title)
        return removed  # 返回删除的文档数量，调用才可以知道删除了多少条

    def search(
        self,
        query: str,  # 查询字符串
        top_k: int = 10,  # 返回前K个结果
        legal_field: Optional[str] = None,  # 法律领域过滤（如"劳动法"）
        knowledge_type: Optional[str] = None,  # 知识类型过滤（如"statute"）
    ) -> List[Dict]:  # 返回文档列表
        return self.index.search(  # 委托给底层索引
            query=query,
            top_k=top_k,
            legal_field=legal_field,
            knowledge_type=knowledge_type,
        )

    # 2. 清空方法
    def clear(self) -> None:
        self.index.clear()  # 委托给底层索引

    # 3. 统计方法
    def stats(self) -> Dict:
        return self.index.stats()  # 委托给底层索引
