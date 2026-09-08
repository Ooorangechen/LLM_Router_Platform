import re
from src.utils.logger import get_logger
from src.utils.schema import QueryType
from typing import Dict, List, Tuple, Any, Optional, Callable

logger = get_logger(__name__)

PATTERN_WEIGHT = 0.6
KEYWORD_WEIGHT= 0.4
CONFIDENCE_THRESHOLD = 0.1  
FALLBACK_CONFIDENCE = 0.5

_FALLBACK_TOKENS_PER_WORD = 1.3
_TOKEN_RE = re.compile(r"[a-z0-9_+#]+")

HF_TOKENIZER_REPOS: Dict[str, str] = {
    "llama": "unsloth/Meta-Llama-3.1-70B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


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

class TokenCounter:
    def __init__(self) -> None:
        self.endoders: Dict[str, Any] = {}
        self.encoder_names: Dict[str, str] = {}
        self._initialize_encoders()

    def _initialize_encoders(self):
        cl100k_counter: Optional[Callable[[str], int]] = None
        try:
            import tiktoken
            o200k_enc = tiktoken.get_encoding("o200k_base")
            cl100k_enc = tiktoken.get_encoding("cl100k_base")
            cl100k_counter = lambda t: len(cl100k_enc.encode(t))
            o200k_counter = lambda t: len(o200k_enc.encode(t))

            self.encoders["gpt"] = o200k_counter
            self.encoders["claude"] = cl100k_counter
            self.encoders["default"] = cl100k_counter
            self.encoder_names.update(
                {"gpt": "o200k_base", "claude": "cl100k_base", "default": "cl100k_base"}
            )

        except Exception as e:
            logger.warning(
                f"tiktoken unavailable, gpt/claude/default will use "
                f"word-count approximation: {type(e).__name__}: {e}")

        for key, repo in HF_TOKENIZER_REPOS.items():
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(repo)
                self.encoders[key] = (
                    lambda t, _tok=tok: len(_tok.encode(t, add_special_tokens=False))
                )
                self.encoder_names[key] = f"{repo} (vocab={tok.vocab_size})"
            except Exception as e:
                if cl100k_counter is not None:
                    self.encoders[key] = cl100k_counter
                    self.encoder_names[key] = "cl100k_base (fallback)"
                logger.warning(
                    f"Failed to load HF tokenizer '{repo}' for key '{key}', "
                    f"falling back to cl100k_base: {type(e).__name__}: {e}"
                )
        logger.info(f"Token encoders initialized: {self.encoder_names}")

    def _get_encoder_key(self, model:str) -> str:
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
        
    def count_toknes(self, text: str, model: str ="default") -> int:
        if not text:
            return 0
        try:
            return self.encoders[self._get_encoder_key(model)](text)
        except Exception as e:
            logger.warning(
                f"Token counting failed for model='{model}', "
                f"falling back to word-count approximation: {type(e).__name__}: {e}"
            )
            return int(len(text.split()) * _FALLBACK_TOKENS_PER_WORD)

