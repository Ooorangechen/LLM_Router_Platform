import re
from src.utils.logger import get_logger
from src.utils.schema import QueryType
from typing import Dict, List, Tuple

logger = get_logger(__name__)

PATTERN_WEIGHT = 0.6
KEYWORD_WEIGHT= 0.4
CONFIDENCE_THRESHOLD = 0.1  
FALLBACK_CONFIDENCE = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9_+#]+")

class QueryClassifier:
    """
    Convert Text query -> QueryType
    """
    def __init__(self) -> None:
        self.patterns: Dict[QueryType, List[str]] = self._init_patterns()
        self.keywords: Dict[QueryType, List[str]] = self._init_keywords()
        self.tfidf = None
        self._is_initialized = False

    def _init_patterns(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.TRANSLATION: [
                r"\b(translat\w+|localiz\w+)\b|翻译|译成|译文",
                r"\b(into|to|from)\s+(english|chinese|spanish|french|german|japanese|korean|russian)\b"
                r"|(翻译成|译为).{0,6}(英|中|日|法|德|西|韩|俄)",
                r"\bin\s+(english|chinese|japanese|french|german|spanish)\b|用(英文|中文|日语|法语)(说|表达)",
                r"\bwhat does .{0,30} mean in\b|的(英文|中文|日文)是什么",
            ],
            QueryType.MATH: [
                r"\b(calculat|comput|solv|deriv|integrat|differentiat)\w*\b|计算|求解|求导|积分",
                r"\b(equation|formula|theorem|proof|matrix|vector|probability|derivative|integral)\b"
                r"|方程|公式|定理|证明|矩阵|向量|概率|导数",
                r"\d+\s*[\+\-\*/\^=]\s*\d+|\b(sum|product|average|mean|median|percentage)\b|求和|平均值|百分比",
                r"\b(algebra|calculus|geometry|statistic|arithmetic)\w*\b|代数|微积分|几何|统计学|算术",
            ],
            QueryType.CODE_ANALYSIS: [
                r"\b(debug|fix|repair|troubleshoot)\w*\b|修复|调试|排查",
                r"\b(bug|error|exception|traceback|stack ?trace|crash|fail(ing|ed)?)\b|报错|异常|崩溃",
                r"\b(optimiz|refactor|performance|complexity)\w*\b|memory leak|优化|重构|复杂度|内存泄漏",
                r"\b(review|explain|analyz\w+)\b.{0,15}\b(code|function|snippet)\b|(看看|检查|分析).{0,8}(代码|函数)",
            ],
            QueryType.CODE_GENERATION: [
                r"\b(write|generate|create|implement|build)\b.{0,15}"
                r"\b(code|function|class|script|program|api|endpoint)\b"
                r"|(写|实现|生成|编写).{0,10}(代码|函数|类|脚本|程序|接口)",
                r"\b(def|class|import|function|const|async|await|return)\b",
                r"\b(python|java|javascript|typescript|c\+\+|golang|rust|sql|bash|react|html|css)\b",
                r"\bhow (do|to|can) i (write|code|implement|build)\b|怎么(写|实现|编写|做一个)",
            ],
            QueryType.SUMMARIZATION: [
                r"\b(summar\w+|tldr|tl;dr|recap|condense)\b|总结|概括|摘要|归纳",
                r"\bkey (points?|takeaways?|findings?)\b|要点|重点|核心内容",
                r"\b(in (short|brief)|briefly|shorten)\b|简要|简述|一句话",
                r"\b(extract|pull out)\b.{0,15}\b(main|key|important)\b|(提炼|提取).{0,8}(要点|信息)",
            ],
            QueryType.CREATIVE_WRITING: [
                r"\b(write|compose|draft|create)\b.{0,15}"
                r"\b(story|poem|novel|song|lyric|script|essay|article|blog)\b"
                r"|(写|创作|来一[篇个段]).{0,15}(故事|小说|诗|诗歌|歌词|剧本|散文)",
                r"\b(creative|fiction|narrative|character|plot|dialogue)\b|创意|虚构|情节|人物|对白",
                r"\b(imagine|pretend|role.?play|as if you were)\b|想象|假设你是|扮演",
                r"\bin the style of\b|模仿.{0,6}(风格|文风)",
            ],
            QueryType.BRAINSTORMING: [
                r"\b(brainstorm|ideate)\w*\b|头脑风暴|发散",
                r"\b(give|list|generate|suggest)\b.{0,20}"
                r"\b(ideas?|options?|alternatives?|ways?|suggestions?)\b"
                r"|(给|列|想|来).{0,15}(想法|点子|主意|方案|建议)",
                r"\b(what else|any other|other possibilities)\b|还有(什么|哪些)|其他(方案|可能)",
                r"\b\d{1,2}\s+(ideas?|ways?|options?|tips?)\b|\d{1,2}\s*个\s*(点子|想法|方法|建议)",
            ],
            QueryType.PLANNING: [
                r"\b(plan|roadmap|schedule|timeline|milestone|agenda)\b|计划|规划|路线图|时间表|里程碑|日程",
                r"\bstep.by.step\b|\bsteps? to\b|\bhow (do|to) i (start|begin|approach)\b|步骤|分步|怎么(开始|着手)",
                r"\b(strategy|strategic|prioriti\w+|allocate)\b|策略|战略|优先级|排期",
                r"\b(organiz|arrang|prepar)\w*\b.{0,20}\b(project|event|trip|sprint)\b|(安排|筹备).{0,8}(项目|活动|行程)",
            ],
            QueryType.ANALYSIS: [
                r"\b(analyz|analys|evaluat|assess|examin)\w*\b|分析|评估|考察",
                r"\b(compare|contrast|difference|versus|vs\.?|pros and cons)\b|对比|比较|区别|优缺点",
                r"\b(trend|pattern|insight|correlation|breakdown)\b|趋势|规律|洞察|相关性",
                r"\bwhat (are|is) the (impact|effect|implication)\w*\b|有什么(影响|意义)",
            ],
            QueryType.REASONING: [
                r"\b(why|rationale|justif\w+)\b|为什么|为何|原因是|理由",
                r"\b(infer|deduce|conclude|imply|therefore)\b|推理|推导|推断|因此",
                r"\bif\b.{0,30}\bthen\b|\b(suppose|assume|hypothetical|what if)\b|如果.{0,20}(那么|会怎样)|假如|假设",
                r"\b(logic|logical|fallacy|argument|premise|conclusion)\b|逻辑|谬误|论点|前提|结论",
            ],
            QueryType.QUESTION_ANSWERING: [
                r"^\s*(what|who|when|where|which|how much|how many)\b|^\s*(什么|谁|何时|哪里|哪个|多少)",
                r"\b(what (is|are|does|do)|who (is|was)|when (did|was)|where (is|can))\b|是什么|是谁|在哪",
                r"\b(tell me about|do you know|can you tell)\b|告诉我|你知道.{0,10}吗",
                r"\b(definition|meaning|means|stands for)\b|定义|含义|意思是",
            ],
        }


    def _init_keywords(self) -> Dict[QueryType, List[str]]:

        return {
            QueryType.TRANSLATION: [
                "translate", "translation", "chinese", 'english',
                "翻译", "中文", "意思", '英文'
            ],
            QueryType.MATH: [
                "calculate", "equation", "derivative", "probability", 
                "计算", "概率", '运算'
            ],
            QueryType.CODE_ANALYSIS: [
                "bug", "debug", "crash", "refactor",
                "报错", "异常", '修复'
            ],
            QueryType.CODE_GENERATION: [
                "function", "script", "endpoint", "rest", 'c++',
                "函数", "脚本", "实现",
            ],
            QueryType.SUMMARIZATION: [
                "summarize", "tldr", 'summary',
                "总结", "摘要", "压缩", "纪要",
            ],
            QueryType.CREATIVE_WRITING: [
                "story", "poem", "novel", "fiction",
                "故事", "小说", "主角",
            ],
            QueryType.BRAINSTORMING: [
                "brainstorm", "idea", "alternative", "option",
                "点子", "发散", "方案",
            ],
            QueryType.PLANNING: [
                "roadmap", "schedule", "timeline", 'plan',
                "计划", "规划", "排期", "着手",
            ],
            QueryType.ANALYSIS: [
                "analyze", "analysis", "compare", "evaluate", "trend",
                "分析", "评估",
            ],
            QueryType.REASONING: [
                "rationale", "infer", "premise", 
                "为什么", "假如", "怎样",
            ],
            QueryType.QUESTION_ANSWERING: [
                "definition", "meaning", "about", 'explain',
                "是什么", "定义", "谁",
            ],
        }


    async def initialize(self) -> None:
        try:
            pass
        except Exception as e:  
            logger.warning(
                f"Semantic model unavailable, falling back to pattern+keyword only: {e}"
            )
        finally:
            self._is_initialized = True

    def _keyword_classification(self, query: str) -> Dict[QueryType, float]:
        """ 
        score = hits / total keywords count

        """
        text = query.lower()
        tokens = set(_TOKEN_RE.findall(text))

        scores: Dict[QueryType, float] = {}
        for query_type, words in self.keywords.items():
            if not words:
                scores[query_type] = 0.0
                continue

            hits = 0
            for word in words:
                if " " in word or not word.isascii():
                    matched = word in text
                else:
                    matched = any(
                        t == word or (t.startswith(word) and len(t) - len(word) <= 3)
                        for t in tokens
                    )
                if matched:
                    hits += 1

            scores[query_type] = hits / len(words)

        return scores

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        regex + keywords -> confidence -> threshold -> (QueryType, confidence)
        """
        if not query or not query.strip():
            return QueryType.GENERAL, 0.0

        text = query.lower()
        keyword_scores = self._keyword_classification(text)

        best_type = QueryType.GENERAL
        best_score = 0.0

        for query_type, patterns in self.patterns.items():
            if patterns:
                matched = sum(1 for p in patterns if re.findall(p, text))
                pattern_score = matched / len(patterns)
            else:
                pattern_score = 0.0

            keyword_score = keyword_scores.get(query_type, 0.0)
            final_score = pattern_score * PATTERN_WEIGHT + keyword_score * KEYWORD_WEIGHT

            if final_score > best_score:
                best_score = final_score
                best_type = query_type

        if best_score > CONFIDENCE_THRESHOLD:
            return best_type, min(best_score, 1.0)

        return QueryType.GENERAL, FALLBACK_CONFIDENCE