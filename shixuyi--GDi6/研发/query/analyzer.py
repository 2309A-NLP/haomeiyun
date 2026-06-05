from __future__ import annotations

import re

from ..models.schemas import EntityBundle, QueryAnalysis


class QueryAnalyzer:
    _company_pattern = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9()（）·\-]{4,80}?(?:股份有限公司|有限责任公司|有限公司))")
    _company_prefix_pattern = re.compile(r"^(?:那家|这家|该家|那么|那|这|该|与|和|跟|同|关于|对于|有关|请问|请说明|请介绍)+")
    _indicator_terms = {
        "收入": "收入",
        "占比": "占比",
        "比重": "比重",
        "注册资本": "注册资本",
        "注册地址": "注册地址",
        "法定代表人": "法定代表人",
        "技术标准": "技术标准",
        "科技进步一等奖": "科技进步一等奖",
        "补充流动资金": "补充流动资金",
        "募集资金": "募集资金",
        "供应商": "供应商",
        "上游": "上游",
        "下游": "下游",
    }

    _domain_terms = ["军用领域", "军队视频指挥领域", "电子信息", "国防军队", "政府", "监狱", "油田", "大型企业"]

    def analyze(self, question: str) -> QueryAnalysis:
        question = (question or "").strip()
        normalized = re.sub(r"\s+", "", question)
        years = re.findall(r"20\d{2}|2019H1|2019年1-6月", question)
        indicators = [value for key, value in self._indicator_terms.items() if key in question]
        domains = [term for term in self._domain_terms if term in question]
        company = self._extract_company(question)

        intent = "fact"
        if any(token in question for token in ("多少", "金额", "万元", "亿元")):
            intent = "amount"
        if any(token in question for token in ("占比", "比重", "比较")):
            intent = "comparison"
        if any(token in question for token in ("是什么", "哪些", "哪个")) and intent == "fact":
            intent = "definition"

        keywords = self._expand_keywords(question, indicators, domains)
        sub_queries = self._build_sub_queries(question, indicators, domains, years)
        disambiguation = self._build_disambiguation(question)

        return QueryAnalysis(
            intent=intent,
            normalized_query=normalized,
            disambiguation=disambiguation,
            sub_queries=sub_queries,
            keywords=keywords,
            entities=EntityBundle(
                company=company,
                years=years,
                indicators=indicators,
                domains=domains,
            ),
        )

    def _extract_company(self, question: str) -> str | None:
        match = self._company_pattern.search(question)
        if match:
            return self._clean_company_name(match.group(1))
        if "兴图新科" in question:
            return "武汉兴图新科电子股份有限公司"
        if "力源信息" in question:
            return "武汉力源信息技术股份有限公司"
        return None

    def _clean_company_name(self, company: str) -> str:
        normalized = re.sub(r"\s+", "", company or "")
        normalized = self._company_prefix_pattern.sub("", normalized)
        return normalized.strip("，。；？、")

    def _expand_keywords(self, question: str, indicators: list[str], domains: list[str]) -> list[str]:
        keywords = [question]
        keywords.extend(indicators)
        keywords.extend(domains)

        expansions = {
            "军用领域收入": ["军品销售额", "军用收入", "军用领域收入占主营业务收入比重"],
            "技术标准": ["视频指挥系统技术标准", "技术规范2.0"],
            "注册资本": ["股本", "注册资本"],
            "注册地址": ["注册地址", "住所", "注册地及主要生产经营地"],
            "法定代表人": ["法定代表人"],
            "补充流动资金": ["募集资金", "补流", "补充流动资金"],
            "供应商": ["重要供应商", "视频指挥"],
            "上游": ["产业链", "上游企业"],
            "下游": ["下游行业", "应用领域"],
        }
        for key, values in expansions.items():
            if key in question or key in indicators:
                keywords.extend(values)

        seen: set[str] = set()
        deduplicated: list[str] = []
        for item in keywords:
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            deduplicated.append(item)
        return deduplicated

    def _build_sub_queries(
        self,
        question: str,
        indicators: list[str],
        domains: list[str],
        years: list[str],
    ) -> list[str]:
        sub_queries = []
        if indicators:
            for indicator in indicators:
                pieces = [indicator]
                if domains:
                    pieces.extend(domains[:1])
                if years:
                    pieces.extend(years[:2])
                sub_queries.append(" ".join(pieces))
        if not sub_queries:
            sub_queries.append(question)
        return sub_queries

    def _build_disambiguation(self, question: str) -> list[str]:
        hints = []
        if "多少" in question and "募集资金" not in question and "补流" in question:
            hints.append("优先理解为补充流动资金金额，而不是募集资金总额")
        if "重要供应商" in question:
            hints.append("优先定位军队视频指挥领域供应商相关表述")
        if "技术标准" in question:
            hints.append("优先查找参与制定标准而非拥有专利")
        return hints
