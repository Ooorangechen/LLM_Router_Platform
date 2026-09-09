import re
import random
import math
from copy import deepcopy
from time import perf_counter
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import ModelConfig, QueryRequest, QueryType, RoutingDecision
from typing import Dict, List, Tuple, Any, Optional, Callable

logger = get_logger(__name__)

PATTERN_WEIGHT = 0.6
KEYWORD_WEIGHT= 0.4
CONFIDENCE_THRESHOLD = 0.1  
FALLBACK_CONFIDENCE = 0.5

_FALLBACK_TOKENS_PER_WORD = 1.3
_TOKEN_RE = re.compile(r"[a-z0-9_+#]+")
_CHINESE_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

TOKENIZER_FAMILIES: Dict[str, Dict[str, str]] = {
    "gpt": {
        "backend": "tiktoken",
        "name": "o200k_base",
    },
    "claude": {
        "backend": "tiktoken",
        "name": "o200k_base",
    },
    "llama": {
        "backend": "huggingface",
        "name": "meta-llama/Llama-3.1-70B-Instruct",
    },
    "mistral": {
        "backend": "huggingface",
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
}


@dataclass
class RoutingRule:
    """Internal routing rule loaded from the router configuration."""

    name: str
    condition: str
    models: List[str]
    fallback: str
    weight: float = 1.0

    def matches(self, query_context: Dict[str, Any]) -> bool:
        # P2 conditions come from trusted operator configuration. Disabling
        # builtins does not make eval a sandbox for user-supplied expressions.
        try:
            return bool(
                eval(
                    self.condition,
                    {"__builtins__": {}},
                    query_context,
                )
            )
        except Exception as exc:
            logger.warning(
                "Routing rule '%s' evaluation failed: %s: %s",
                self.name,
                type(exc).__name__,
                exc,
            )
            return False


@dataclass
class ModelCandidate:
    """Internal model-selection result used to build RoutingDecision."""

    model_name: str
    confidence: float
    reason: str


class QueryClassifier:
    """
    Convert Text query -> QueryType
    """
    def __init__(self) -> None:
        self.patterns: Dict[QueryType, List[str]] = self._init_patterns()
        self.chinese_patterns: Dict[
            QueryType, List[str]
        ] = self._init_chinese_patterns()
        self.keywords: Dict[QueryType, List[str]] = self._init_keywords()
        self.chinese_keywords: Dict[
            QueryType, List[str]
        ] = self._init_chinese_keywords()
        self.tfidf = None
        self._is_initialized = False

    def _init_patterns(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.TRANSLATION: [
                r"\b(translat\w+|localiz\w+)\b",
                r"\b(into|to|from)\s+(english|chinese|spanish|french|german|japanese|korean|russian)\b",
                r"\bin\s+(english|chinese|japanese|french|german|spanish)\b",
                r"\bwhat does .{0,30} mean in\b",
            ],
            QueryType.MATH: [
                r"\b(calculat|comput|solv|deriv|integrat|differentiat)\w*\b",
                r"\b(equation|formula|theorem|proof|matrix|vector|probability|derivative|integral)\b",
                r"\d+\s*[\+\-\*/\^=]\s*\d+|\b(sum|product|average|mean|median|percentage)\b",
                r"\b(algebra|calculus|geometry|statistic|arithmetic)\w*\b",
            ],
            QueryType.CODE_ANALYSIS: [
                r"\b(debug|fix|repair|troubleshoot)\w*\b",
                r"\b(bug|error|exception|traceback|stack ?trace|crash|fail(ing|ed)?)\b",
                r"\b(optimiz|refactor|performance|complexity)\w*\b|memory leak",
                r"\b(review|explain|analyz\w+)\b.{0,15}\b(code|function|snippet)\b",
            ],
            QueryType.CODE_GENERATION: [
                r"\b(write|generate|create|implement|build)\b.{0,15}"
                r"\b(code|function|class|script|program|api|endpoint)\b",
                r"\b(def|class|import|function|const|async|await|return)\b",
                r"\b(python|java|javascript|typescript|c\+\+|golang|rust|sql|bash|react|html|css)\b",
                r"\bhow (do|to|can) i (write|code|implement|build)\b",
            ],
            QueryType.SUMMARIZATION: [
                r"\b(summar\w+|tldr|tl;dr|recap|condense)\b",
                r"\bkey (points?|takeaways?|findings?)\b",
                r"\b(in (short|brief)|briefly|shorten)\b",
                r"\b(extract|pull out)\b.{0,15}\b(main|key|important)\b",
            ],
            QueryType.CREATIVE_WRITING: [
                r"\b(write|compose|draft|create)\b.{0,15}"
                r"\b(story|poem|novel|song|lyric|script|essay|article|blog)\b",
                r"\b(creative|fiction|narrative|character|plot|dialogue)\b",
                r"\b(imagine|pretend|role.?play|as if you were)\b",
                r"\bin the style of\b",
            ],
            QueryType.BRAINSTORMING: [
                r"\b(brainstorm|ideate)\w*\b",
                r"\b(give|list|generate|suggest)\b.{0,20}"
                r"\b(ideas?|options?|alternatives?|ways?|suggestions?)\b",
                r"\b(what else|any other|other possibilities)\b",
                r"\b\d{1,2}\s+(ideas?|ways?|options?|tips?)\b",
            ],
            QueryType.PLANNING: [
                r"\b(plan|roadmap|schedule|timeline|milestone|agenda)\b",
                r"\bstep.by.step\b|\bsteps? to\b|\bhow (do|to) i (start|begin|approach)\b",
                r"\b(strategy|strategic|prioriti\w+|allocate)\b",
                r"\b(organiz|arrang|prepar)\w*\b.{0,20}\b(project|event|trip|sprint)\b",
            ],
            QueryType.ANALYSIS: [
                r"\b(analyz|analys|evaluat|assess|examin)\w*\b",
                r"\b(compare|contrast|difference|versus|vs\.?|pros and cons)\b",
                r"\b(trend|pattern|insight|correlation|breakdown)\b",
                r"\bwhat (are|is) the (impact|effect|implication)\w*\b",
            ],
            QueryType.REASONING: [
                r"\b(why|rationale|justif\w+)\b",
                r"\b(infer|deduce|conclude|imply|therefore)\b",
                r"\bif\b.{0,30}\bthen\b|\b(suppose|assume|hypothetical|what if)\b",
                r"\b(logic|logical|fallacy|argument|premise|conclusion)\b",
            ],
            QueryType.QUESTION_ANSWERING: [
                r"^\s*(what|who|when|where|which|how much|how many)\b",
                r"\b(what (is|are|does|do)|who (is|was)|when (did|was)|where (is|can))\b",
                r"\b(tell me about|do you know|can you tell)\b",
                r"\b(definition|meaning|means|stands for)\b",
            ],
        }

    def _init_chinese_patterns(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.TRANSLATION: [
                r"(翻译|翻成|译成|转译|转换成).{0,12}"
                r"(中文|英文|英语|日文|日语|韩文|韩语|法文|法语|德文|德语|西班牙语)",
                r"(中译英|英译中|汉译英|英译汉)",
                r"(这句|这段|这篇|以下内容).{0,8}(怎么翻译|如何翻译|是什么意思)",
                r"用(中文|英文|英语|日文|日语|韩文|韩语).{0,6}(表达|改写|说明)",
            ],
            QueryType.MATH: [
                r"(计算|求解|解出|证明|推导).{0,15}(方程|公式|结果|数值|定理)",
                r"(导数|微分|积分|极限|概率|矩阵|向量|几何|代数)",
                r"\d+\s*[\+\-\*/×÷\^=]\s*\d+",
                r"(平均数|中位数|百分比|标准差|方差|行列式)",
            ],
            QueryType.CODE_ANALYSIS: [
                r"(调试|排查|修复|定位).{0,15}(代码|程序|错误|问题|异常|故障)",
                r"(报错|异常|崩溃|闪退|堆栈|追踪|内存泄漏)",
                r"(优化|重构|改进).{0,15}(代码|性能|复杂度|内存|速度)",
                r"(分析|审查|解释).{0,15}(代码|函数|类|脚本|实现)",
            ],
            QueryType.CODE_GENERATION: [
                r"(写|编写|生成|创建|实现|开发).{0,15}(代码|函数|类|脚本|程序|接口|服务)",
                r"(使用|用).{0,10}"
                r"(python|java|javascript|typescript|c\+\+|go|rust|sql)"
                r".{0,15}(写|实现|开发)",
                r"(定义|实现).{0,10}(函数|类|接口|算法|数据结构)",
                r"(帮我|请).{0,6}(写|实现|生成).{0,15}(代码|程序|函数)",
            ],
            QueryType.SUMMARIZATION: [
                r"(总结|概括|归纳|提炼).{0,12}(内容|文章|文本|报告|要点|重点)",
                r"(生成|写).{0,8}(摘要|概要|总结)",
                r"(简要|简短|一句话).{0,8}(说明|概括|总结)",
                r"(提取|列出).{0,10}(要点|重点|结论|关键信息)",
            ],
            QueryType.CREATIVE_WRITING: [
                r"(写|创作|编写).{0,12}(故事|诗歌|小说|文案|剧本|歌词|散文)",
                r"(设计|塑造).{0,10}(角色|人物|情节|世界观|对白)",
                r"(续写|改写|润色).{0,12}(故事|文章|文案|剧情)",
                r"(想象|虚构|扮演).{0,15}",
            ],
            QueryType.BRAINSTORMING: [
                r"(头脑风暴|集思广益)",
                r"(给出|提供|生成|列出).{0,10}(想法|点子|创意|方案|建议|备选)",
                r"(还有什么|其他可能|更多选择|更多思路)",
                r"(想几个|给几个|列举).{0,10}(方案|名字|点子|创意|方向)",
            ],
            QueryType.PLANNING: [
                r"(制定|设计|创建|安排).{0,12}(计划|规划|路线图|日程|时间表)",
                r"(步骤|流程|阶段|里程碑|时间线)",
                r"(如何|怎么).{0,8}(开始|推进|准备|安排|规划)",
                r"(项目|旅行|活动|学习|冲刺).{0,12}(计划|安排|规划)",
            ],
            QueryType.ANALYSIS: [
                r"(分析|评估|研究|考察).{0,15}(数据|趋势|影响|原因|结果|表现)",
                r"(比较|对比|区别).{0,15}",
                r"(趋势|模式|洞察|相关性|分布|构成)",
                r"(影响|效果|后果|含义|启示).{0,12}(是什么|有哪些|如何)",
            ],
            QueryType.REASONING: [
                r"(为什么|为何|原因是什么|理由是什么)",
                r"(推理|推断|演绎|归纳|论证)",
                r"(如果|假设|假如).{0,30}(那么|则|会不会|是否)",
                r"(逻辑|前提|结论|谬误|矛盾|因果)",
            ],
            QueryType.QUESTION_ANSWERING: [
                r"^\s*(什么是|谁是|何时|什么时候|哪里|哪个|多少|几种)",
                r"(请问|告诉我|介绍一下|解释一下).{0,20}",
                r"(定义|含义|意思|全称).{0,10}(是什么|指什么|为)",
                r".{0,20}(是什么|有哪些|在哪里|有多少)\s*[？?]?$",
            ],
        }

    def _init_keywords(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.TRANSLATION: [
                "translate", "translation", "english",
                "chinese", "spanish", "japanese",
            ],
            QueryType.MATH: [
                "calculate", "equation", "derivative",
                "probability", "integral", "determinant",
            ],
            QueryType.CODE_ANALYSIS: [
                # 不放 fix / error：两者都已在 patterns 里，且 error 会误伤普通英文
                "bug", "debug", "crash",
                "refactor", "traceback", "segfault",
            ],
            QueryType.CODE_GENERATION: [
                "function", "script", "endpoint",
                "rest", "api", "python",
            ],
            QueryType.SUMMARIZATION: [
                "summarize", "summary", "tldr",
                "recap", "condense", "shorten",
            ],
            QueryType.CREATIVE_WRITING: [
                "story", "poem", "novel",
                "fiction", "protagonist", "lyric",
            ],
            QueryType.BRAINSTORMING: [
                "brainstorm", "idea", "alternative",
                "option", "suggestion", "possible",
            ],
            QueryType.PLANNING: [
                "roadmap", "schedule", "timeline",
                "milestone", "sprint", "agenda",
            ],
            QueryType.ANALYSIS: [
                "analyze", "analysis", "compare",
                "evaluate", "trend", "correlation",
            ],
            QueryType.REASONING: [
                # 不放 why：它在 patterns 第 1 条里，重复放会和 CODE_ANALYSIS 抢
                # "why does this loop run so slowly" 这类查询
                "rationale", "infer", "premise",
                "deduce", "fallacy", "hypothesis",
            ],
            QueryType.QUESTION_ANSWERING: [
                # 刻意不放 what / who / when / where —— 见上面原则 1
                "definition", "meaning", "about",
                "explain", "fact", "overview",
            ],
        }

    def _init_chinese_keywords(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.TRANSLATION: [
                "翻译", "译成", "转译", "中译英",
                "英译中", "中文", "英文", "日文",
            ],
            QueryType.MATH: [
                "计算", "求解", "方程", "导数",
                "积分", "概率", "矩阵", "证明",
            ],
            QueryType.CODE_ANALYSIS: [
                "调试", "修复", "报错", "异常",
                "性能", "优化", "复杂度", "代码审查",
            ],
            QueryType.CODE_GENERATION: [
                "写代码", "编写", "实现", "函数",
                "类定义", "脚本", "接口", "程序",
            ],
            QueryType.SUMMARIZATION: [
                "总结", "摘要", "概括", "归纳",
                "要点", "简述", "精简", "提炼",
            ],
            QueryType.CREATIVE_WRITING: [
                "故事", "诗歌", "小说", "文案",
                "剧本", "创作", "角色", "情节",
            ],
            QueryType.BRAINSTORMING: [
                "头脑风暴", "点子", "创意", "想法",
                "方案", "建议", "备选", "思路",
            ],
            QueryType.PLANNING: [
                "计划", "规划", "路线图", "时间表",
                "里程碑", "日程", "步骤", "安排",
            ],
            QueryType.ANALYSIS: [
                "分析", "评估", "比较", "对比",
                "趋势", "模式", "洞察", "相关性",
            ],
            QueryType.REASONING: [
                "为什么", "推理", "推断", "逻辑",
                "前提", "结论", "假设", "论证",
            ],
            QueryType.QUESTION_ANSWERING: [
                "什么是", "谁是", "何时", "哪里",
                "多少", "解释", "定义", "介绍",
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
        """Score exact English and Chinese keyword-set intersections."""
        text = query.lower()
        tokens = set(_TOKEN_RE.findall(text))

        for match in _CHINESE_SPAN_RE.finditer(text):
            span = match.group(0)
            max_ngram_size = min(6, len(span))
            for size in range(1, max_ngram_size + 1):
                for start in range(len(span) - size + 1):
                    tokens.add(span[start:start + size])

        scores: Dict[QueryType, float] = {}
        for query_type, words in self.keywords.items():
            english_keywords = {word.lower() for word in words}
            chinese_keywords = {
                word.lower()
                for word in self.chinese_keywords.get(query_type, [])
            }

            english_score = (
                len(tokens & english_keywords) / len(english_keywords)
                if english_keywords
                else 0.0
            )
            chinese_score = (
                len(tokens & chinese_keywords) / len(chinese_keywords)
                if chinese_keywords
                else 0.0
            )
            scores[query_type] = max(english_score, chinese_score)

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
            english_pattern_score = (
                sum(1 for pattern in patterns if re.findall(pattern, text))
                / len(patterns)
                if patterns
                else 0.0
            )

            chinese_patterns = self.chinese_patterns.get(query_type, [])
            chinese_pattern_score = (
                sum(
                    1
                    for pattern in chinese_patterns
                    if re.findall(pattern, text)
                )
                / len(chinese_patterns)
                if chinese_patterns
                else 0.0
            )
            pattern_score = max(
                english_pattern_score,
                chinese_pattern_score,
            )

            keyword_score = keyword_scores.get(query_type, 0.0)
            final_score = pattern_score * PATTERN_WEIGHT + keyword_score * KEYWORD_WEIGHT

            if final_score > best_score:
                best_score = final_score
                best_type = query_type

        if best_score > CONFIDENCE_THRESHOLD:
            return best_type, min(best_score, 1.0)

        return QueryType.GENERAL, FALLBACK_CONFIDENCE

class TokenCounter:
    def __init__(self) -> None:
        self.encoders: Dict[str, Callable[[str], int]] = {}
        self.encoder_names: Dict[str, str] = {}
        self._initialize_encoders()

    def _initialize_encoders(self) -> None:
        default_counter: Optional[Callable[[str], int]] = None

        tiktoken_names = {"default": "cl100k_base"}
        tiktoken_names.update({
            family: spec["name"] for family, spec in TOKENIZER_FAMILIES.items()
            if spec["backend"] == "tiktoken"
        })
        for family, encoding_name in tiktoken_names.items():
            try:
                import tiktoken

                encoder = tiktoken.get_encoding(encoding_name)
                self.encoders[family] = (
                    lambda text, _encoder=encoder: len(
                        _encoder.encode(text, disallowed_special=())
                    )
                )
                self.encoder_names[family] = f"tiktoken:{encoding_name}"
                if family == "default":
                    default_counter = self.encoders[family]
            except Exception as exc:
                logger.warning(
                    "tiktoken initialization failed for family '%s'; using "
                    "word-count approximation: %s: %s",
                    family, type(exc).__name__, exc,
                )

        for family, spec in TOKENIZER_FAMILIES.items():
            if spec["backend"] != "huggingface":
                continue

            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    spec["name"],
                    use_fast=True,
                    trust_remote_code=False,
                )
                self.encoders[family] = (
                    lambda text, _tokenizer=tokenizer: len(
                        _tokenizer.encode(text, add_special_tokens=False)
                    )
                )
                self.encoder_names[family] = (
                    f"huggingface:{spec['name']} (vocab={len(tokenizer)})"
                )
            except Exception as exc:
                if default_counter is not None:
                    self.encoders[family] = default_counter
                    self.encoder_names[family] = "tiktoken:cl100k_base (fallback)"
                logger.warning(
                    "Failed to load Hugging Face tokenizer '%s' for family "
                    "'%s'; using the available fallback: %s: %s",
                    spec["name"],
                    family,
                    type(exc).__name__,
                    exc,
                )

        logger.info("Token encoders initialized: %s", self.encoder_names)

    def _get_encoder_key(self, model: str) -> str:
        model_lower = (model or "").lower()

        if "gpt" in model_lower:
            return "gpt"
        if "claude" in model_lower:
            return "claude"
        if "llama" in model_lower:
            return "llama"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        return "default"

    def count_tokens(self, text: str, model: str = "default") -> int:
        if not text:
            return 0

        try:
            encoder_key = self._get_encoder_key(model)
            token_count = self.encoders[encoder_key](text)
            return max(0, int(token_count))
        except Exception as exc:
            logger.warning(
                "Token counting failed for model='%s'; falling back to "
                "word-count approximation: %s: %s",
                model,
                type(exc).__name__,
                exc,
            )
            return max(0, round(len(text.split()) * _FALLBACK_TOKENS_PER_WORD))


class ModelRouter:
    """Route a validated QueryRequest to one configured model."""

    VALID_ROUTING_STRATEGIES = {
        "intelligent",
        "round_robin",
        "weighted",
        "fallback",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = deepcopy(config)
        self.models: Dict[str, ModelConfig] = {}
        self.routing_rules: List[RoutingRule] = []
        self.classifier = QueryClassifier()
        self.token_counter = TokenCounter()
        self.default_model = str(config.get("default_model") or "mistral-7b")
        self.routing_strategy = str(
            config.get("routing_strategy", "intelligent")
        )
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self._request_count = 0
        self._is_initialized = False

    async def initialize(self) -> None:
        """Load configuration-backed state and initialize dependencies."""
        if self._is_initialized:
            return
        self._load_models()
        self._load_routing_rules()

        if not self.models:
            raise ValueError("router.models must contain at least one model")
        if self.default_model not in self.models:
            raise ValueError(
                f"Default model '{self.default_model}' is not configured"
            )
        if self.routing_strategy not in self.VALID_ROUTING_STRATEGIES:
            raise ValueError(
                f"Unsupported routing strategy: {self.routing_strategy}"
            )

        try:
            await self.classifier.initialize()
        except Exception as exc:
            logger.warning("Classifier initialization failed: %s", exc)
        self._is_initialized = True
        logger.info(
            "ModelRouter initialized with %d models, %d rules, strategy='%s'",
            len(self.models),
            len(self.routing_rules),
            self.routing_strategy,
        )

    def _load_models(self) -> None:
        """Validate router.models and initialize per-model statistics."""
        self.models.clear()
        self.model_stats.clear()

        for model_name, model_data in self.config.get("models", {}).items():
            model_config = ModelConfig(name=model_name, **model_data)
            self.models[model_name] = model_config
            self.model_stats[model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency": 0.0,
                "success_rate": 1.0,
                "avg_latency": 0.0,
            }

    def _load_routing_rules(self) -> None:
        """Build internal RoutingRule objects from YAML rule mappings."""
        self.routing_rules.clear()

        for rule_data in self.config.get("routing_rules", []):
            self.routing_rules.append(
                RoutingRule(
                    name=rule_data["name"],
                    condition=rule_data["condition"],
                    models=list(rule_data["models"]),
                    fallback=rule_data["fallback"],
                )
            )

    async def route_query(self, request: QueryRequest) -> RoutingDecision:
        """Execute the P2 routing pipeline.

        Implementation order:
        1. Build the query context and model-dependent token counts.
        2. Establish the user-access and token-limit eligible model pool.
        3. Dispatch to the configured routing strategy.
        4. Estimate input/output cost for the selected candidate.
        5. Build RoutingDecision and record router metrics.
        6. On any error, build a default-model fallback RoutingDecision.
        """
        start = perf_counter()
        context: Dict[str, Any] = {}
        try:
            if not self._is_initialized:
                raise RuntimeError("Call await router.initialize() before routing")
            context = await self._build_query_context(request)
            context["available_models"] = self._get_available_models(context)
            if self.routing_strategy == "intelligent":
                selection = await self._intelligent_routing(context)
            elif self.routing_strategy == "round_robin":
                selection = self._round_robin_routing(context)
            elif self.routing_strategy == "weighted":
                selection = self._weighted_routing(context)
            elif self.routing_strategy == "fallback":
                selection = self._fallback_routing(context)
            else:
                raise ValueError(f"Unknown strategy: {self.routing_strategy}")

            decision = RoutingDecision(
                selected_model=selection.model_name,
                query_type=QueryType(context["query_type"]),
                token_count=context["token_counts"][selection.model_name],
                estimated_cost=self._estimate_cost(selection, context),
                routing_reason=selection.reason,
                routing_time_ms=int((perf_counter() - start) * 1000),
                confidence=selection.confidence,
                fallback_models=[
                    name for name in context["available_models"]
                    if name != selection.model_name
                ],
                routing_strategy=self.routing_strategy,
                user_tier=request.user_tier,
            )
        except Exception as exc:
            logger.warning("Routing failed; selecting default model: %s", exc)
            decision = RoutingDecision(
                selected_model=self.default_model,
                query_type=QueryType.GENERAL,
                token_count=context.get("token_counts", {}).get(self.default_model, 0),
                estimated_cost=0.0,
                routing_reason=f"Fallback due to error: {exc}",
                routing_time_ms=int((perf_counter() - start) * 1000),
                confidence=0.0,
                routing_strategy="fallback",
                user_tier=request.user_tier,
            )

        # Observability failures must never discard a completed decision.
        try:
            ROUTER_METRICS.routing_decisions.labels(
                model=decision.selected_model, query_type=decision.query_type.value
            ).inc()
            ROUTER_METRICS.routing_duration.observe(perf_counter() - start)
            ROUTER_METRICS.routing_confidence.labels(
                model=decision.selected_model, query_type=decision.query_type.value
            ).observe(decision.confidence)
        except Exception as exc:
            logger.warning("Router metrics update failed: %s", exc)
        return decision

    async def _build_query_context(
        self,
        request: QueryRequest,
    ) -> Dict[str, Any]:
        """Count query + context once per family, with a count per model.

        There is no universal token_count here. Rule evaluation supplies that
        scalar from token_counts for the particular model being considered.
        Counts cover raw text, not provider-specific message framing or tools.
        """
        try:
            query_type, confidence = self.classifier.classify_query(request.query)
        except Exception as exc:
            logger.warning("Classification failed: %s", exc)
            query_type, confidence = QueryType.GENERAL, 0.0
        prompt = "\n\n".join(part for part in (request.context, request.query) if part)
        counts_by_family: Dict[str, int] = {}
        token_counts: Dict[str, int] = {}
        for name in self.models:
            family = self.token_counter._get_encoder_key(name)
            if family not in counts_by_family:
                counts_by_family[family] = self.token_counter.count_tokens(prompt, name)
            token_counts[name] = counts_by_family[family]
        return {
            "query": request.query,
            "query_type": query_type.value,
            "classification_confidence": confidence,
            "user_tier": request.user_tier.value,
            "max_tokens": request.max_tokens,
            "token_counts": token_counts,
            "has_context": bool(request.context),
        }

    def _get_available_models(
        self,
        context: Dict[str, Any],
    ) -> List[str]:
        """Apply access and input-token limits; provider health is external.

        Per P2, this checks input <= model.max_tokens, without reserving output.
        """
        return [
            name for name, model in self.models.items()
            if self._check_user_access(model, context["user_tier"])
            and context["token_counts"][name] <= model.max_tokens
        ]

    async def _intelligent_routing(
        self,
        context: Dict[str, Any],
    ) -> ModelCandidate:
        """Match ordered rules, then fall through to capability routing."""
        available = set(context["available_models"])
        for rule in self.routing_rules:
            # Supply a scalar token_count for each model; do not use a
            # default-model estimate for every rule/candidate.
            matching = [
                name for name in dict.fromkeys(rule.models)
                if name in self.models and rule.matches({
                    **context, "token_count": context["token_counts"][name]
                })
            ]
            candidates = [name for name in matching if name in available]
            if candidates:
                return ModelCandidate(
                    self._select_best_model(candidates, context),
                    context["classification_confidence"],
                    f"Matched rule '{rule.name}': {rule.condition}",
                )
            if matching and rule.fallback in available:
                return ModelCandidate(
                    rule.fallback, 0.5, f"Fallback for matched rule '{rule.name}'"
                )
        return self._capability_based_routing(context)

    def _capability_based_routing(
        self,
        context: Dict[str, Any],
    ) -> ModelCandidate:
        candidates = [
            name for name in context["available_models"]
            if self._has_capability(self.models[name], context["query_type"])
        ]
        if not candidates:
            return self._fallback_routing(context)
        return ModelCandidate(
            self._select_best_model(candidates, context),
            context["classification_confidence"],
            "Selected by capability and composite score",
        )

    def _has_capability(
        self,
        model_config: ModelConfig,
        query_type_value: str,
    ) -> bool:
        mapping = {
            "general": ["general"],
            "code_generation": ["coding"],
            "code_analysis": ["coding", "analysis"],
            "analysis": ["analysis", "reasoning"],
            "summarization": ["writing", "analysis"],
            "creative_writing": ["creative", "writing"],
            "brainstorming": ["creative", "reasoning", "general"],
            "planning": ["reasoning", "general"],
            "question_answering": ["general"],
            "translation": ["translation", "writing"],
            "math": ["math", "reasoning"],
            "reasoning": ["reasoning"],
        }
        return any(cap in model_config.capabilities
                   for cap in mapping.get(query_type_value, ["general"]))

    def _check_user_access(
        self,
        model_config: ModelConfig,
        user_tier_value: str,
    ) -> bool:
        priorities = {"free": 3, "premium": 2, "enterprise": 1}
        return (user_tier_value in priorities
                and model_config.priority >= priorities[user_tier_value])

    def _select_best_model(
        self,
        candidates: List[str],
        context: Dict[str, Any],
    ) -> str:
        """P2 score, using request-specific input/output blended unit price.

        success*40 + max(0, 20-latency_ms/100) + cost (capped at 20)
        + (10-priority)*2 + context headroom (10 or 5). Free models get
        the maximum cost score; cold-start stats are success=1, latency=0.
        """
        if not candidates:
            raise ValueError("No candidates to score")

        def score(name: str) -> float:
            model = self.models[name]
            stats = self.model_stats.get(name, {})
            tokens = context["token_counts"][name]
            output_tokens = min(tokens * 0.5, context["max_tokens"])
            estimated_cost = self._estimate_cost(ModelCandidate(name, 0.0, "scoring"), context)
            total_tokens = tokens + output_tokens
            effective_price = estimated_cost / total_tokens if total_tokens else 0.0
            cost_score = 20.0 if effective_price == 0 else min(
                1.0 / (effective_price * 1e6), 20.0
            )
            context_score = (10 if tokens <= model.max_tokens * 0.8
                             else 5 if tokens <= model.max_tokens else 0)
            return (
                stats.get("success_rate", 1.0) * 40
                + max(0, 20 - stats.get("avg_latency", 0.0) / 100)
                + cost_score + (10 - model.priority) * 2 + context_score
            )

        # max preserves the first candidate when scores tie.
        return max(candidates, key=score)

    def _round_robin_routing(
        self,
        context: Dict[str, Any],
    ) -> ModelCandidate:
        available = context["available_models"]
        if not available:
            return self._fallback_routing(context)
        name = available[self._request_count % len(available)]
        self._request_count += 1
        return ModelCandidate(name, 0.5, "Round-robin selection")

    def _weighted_routing(
        self,
        context: Dict[str, Any],
    ) -> ModelCandidate:
        available = context["available_models"]
        if not available:
            return self._fallback_routing(context)
        weights = [1.0 / self.models[name].priority for name in available]
        total = sum(weights)
        name = random.choices(available, weights=[w / total for w in weights], k=1)[0]
        return ModelCandidate(name, 0.5, "Weighted selection using inverse model priority")

    def _fallback_routing(
        self,
        context: Dict[str, Any],
    ) -> ModelCandidate:
        """P2 default nomination, including when the eligible pool is empty.

        This does not certify input capacity, access, or provider readiness.
        The caller must not treat a fallback decision as execution validation.
        """
        return ModelCandidate(self.default_model, 0.0, "Selected configured default model")

    def _estimate_cost(
        self,
        selection: ModelCandidate,
        context: Dict[str, Any],
    ) -> float:
        """Estimate separate input-token and output-token costs."""
        model = self.models[selection.model_name]
        input_tokens = context["token_counts"][selection.model_name]
        output_tokens = min(input_tokens * 0.5, context["max_tokens"])
        return float(input_tokens * model.cost_input_token
                     + output_tokens * model.cost_output_token)

    def update_model_stats(
        self,
        model_name: str,
        success: bool,
        latency_ms: int,
    ) -> None:
        try:
            if latency_ms < 0 or not math.isfinite(latency_ms):
                raise ValueError("latency_ms must be finite and nonnegative")
            stats = dict(self.model_stats[model_name])
            stats["total_requests"] += 1
            stats["successful_requests"] += int(success)
            stats["total_latency"] += latency_ms
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
            self.model_stats[model_name] = stats
        except Exception as exc:
            logger.warning("Model stats update failed for '%s': %s", model_name, exc)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        model_config = self.models.get(model_name)
        if model_config is None:
            return None

        return {
            "name": model_name,
            "config": model_config.model_dump(),
            "stats": dict(self.model_stats[model_name]),
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Return a snapshot; request_count is the round-robin cursor only."""
        return {
            "routing_strategy": self.routing_strategy,
            "default_model": self.default_model,
            "request_count": self._request_count,
            "models": {
                model_name: dict(stats)
                for model_name, stats in self.model_stats.items()
            },
        }
