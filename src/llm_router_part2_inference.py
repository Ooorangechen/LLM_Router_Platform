"""P2 context compression, Redis cache and multi-provider inference."""

import re
import asyncio
import hashlib
import json
import os
from time import perf_counter
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.schema import QueryRequest, InferenceResponse
from src.utils.logger import get_logger
from src.utils.metrics import INFERENCE_METRICS

if TYPE_CHECKING:
    from src.llm_router_part1_router import ModelRouter, TokenCounter

logger = get_logger(__name__)


@dataclass
class InferenceContext:
    request_id: str
    user_id: str
    model_name: str
    start_time: float
    token_count_input: int
    compressed_context: bool = False
    cached_response: bool = False


class ContextCompressor:
    """P2 text heuristics with model-dependent estimated token budgets."""

    def __init__(self, config: Dict[str, Any], token_counter: "TokenCounter") -> None:
        self.token_counter = token_counter
        self.enabled = config.get("enabled", False)
        self.compression_ratio = config.get("compression_ratio", 0.3)
        # Token threshold and compressed-context cap; not a full request budget.
        self.max_context_tokens = config.get("max_context_tokens", 100000)
        self.method = config.get("method", "semantic_graph")

    async def initialize(self) -> None:
        """Prepare text heuristics without requiring tokenizer/model loading."""
        # P2 uses text heuristics only; no external resources to initialize.
        return

    async def compress_context(self, context: str, model_name: str) -> str:
        """Estimate context tokens once, derive a budget, and dispatch compression."""
        if not self.enabled or not context:
            return context
        original_tokens = self.token_counter.count_tokens(context, model_name)
        if original_tokens <= self.max_context_tokens:
            return context
        target_tokens = min(int(original_tokens * self.compression_ratio), self.max_context_tokens)
        if target_tokens <= 0:
            return ""
        methods = {
            "semantic_graph": self._semantic_graph_compression,
            "sketch_based": self._sketch_based_compression,
            "attention_based": self._attention_based_compression,
            "sliding_window": self._sliding_window_compression,
        }
        compress = methods.get(self.method, self._sliding_window_compression)
        return await compress(context, target_tokens, model_name)

    async def _semantic_graph_compression(self, context: str, target_tokens: int, model_name: str) -> str:
        """Split, score, select top sentences, restore order, then truncate."""
        sentences = self._split_sentences(context)
        scores = await self._score_sentences(sentences)
        count = min(len(sentences), max(3, int(len(sentences) * self.compression_ratio)))
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        result = " ".join(sentences[i] for i in sorted(ranked[:count]))
        return self._trim_to_token_budget(result, target_tokens, model_name, suffix="...")

    async def _sketch_based_compression(self, context: str, target_tokens: int, model_name: str) -> str:
        """Keep paragraphs matching top ten bigram phrases; otherwise use prefix."""
        phrases = self._extract_key_phrases(context)
        paragraphs = [part.strip() for part in context.split("\n") if part.strip()]
        selected = [
            part for part in paragraphs
            if any(phrase in " ".join(part.lower().split()) for phrase in phrases)
        ]
        return self._trim_to_token_budget("\n\n".join(selected) or context, target_tokens, model_name)

    async def _attention_based_compression(self, context: str, target_tokens: int, model_name: str) -> str:
        """Rank paragraphs by keyword density and length; restore original order."""
        keywords = set(self._extract_keywords(context))
        paragraphs = [part.strip() for part in context.split("\n") if part.strip()]
        scores = []
        for paragraph in paragraphs:
            words = re.findall(r"\w+", paragraph.lower())
            density = sum(word in keywords for word in words) / max(1, len(words))
            length_score = min(len(words) / 30, 1.0)
            scores.append(density + length_score)
        ranked = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        selected = []
        result = ""
        for index in ranked:
            candidate = "\n\n".join(paragraphs[i] for i in sorted(selected + [index]))
            if self.token_counter.count_tokens(candidate, model_name) <= target_tokens:
                selected.append(index)
                result = candidate
            elif not selected:
                return self._trim_to_token_budget(paragraphs[index], target_tokens, model_name)
        return result

    async def _sliding_window_compression(self, context: str, target_tokens: int, model_name: str) -> str:
        """Keep head and tail around a compression marker within the budget."""
        marker = "\n...[COMPRESSED]...\n"
        remaining = target_tokens - self.token_counter.count_tokens(marker, model_name)
        if remaining <= 0:
            return self._trim_to_token_budget(context, target_tokens, model_name)
        head = self._trim_to_token_budget(context, remaining // 2, model_name)
        tail = self._trim_to_token_budget(context, remaining - remaining // 2, model_name, keep_end=True)
        # Joined text can tokenize differently from the individual pieces.
        return self._trim_to_token_budget(head + marker + tail, target_tokens, model_name)

    def _trim_to_token_budget(self, text: str, budget: int, model_name: str,
                              suffix: str = "", keep_end: bool = False) -> str:
        """Find a fitting text slice with bounded search; include suffix in the count."""
        count = self.token_counter.count_tokens
        if budget <= 0:
            return ""
        if count(text, model_name) <= budget:
            return text
        if count(suffix, model_name) > budget:
            suffix = ""
        result = suffix
        low, high = 1, len(text) - 1
        while low <= high:
            middle = (low + high) // 2
            candidate = (text[-middle:] if keep_end else text[:middle]) + suffix
            if count(candidate, model_name) <= budget:
                result, low = candidate, middle + 1
            else:
                high = middle - 1
        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text using English/Chinese sentence boundaries."""
        return [
            part.strip() for part in re.split(r"(?<=[.!?。！？])\s*|\n+", text)
            if part.strip()
        ]

    async def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score word length, important-keyword hits, and first/last position."""
        important_words = {"important", "key", "main", "crucial", "essential", "critical"}
        scores = []
        for index, sentence in enumerate(sentences):
            words = re.findall(r"\w+", sentence.lower())
            length_score = 2 if 10 <= len(words) <= 30 else 1 if len(words) > 30 else 0
            keyword_score = sum(word in important_words for word in words)
            keyword_score += sum(sentence.count(word) for word in ("重要", "关键", "主要", "核心"))
            position_score = int(index == 0) + int(index == len(sentences) - 1)
            scores.append(float(length_score + keyword_score + position_score))
        return scores

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Return the ten most frequent two-word windows."""
        phrases = Counter()
        for paragraph in text.lower().splitlines():
            words = paragraph.split()
            phrases.update(f"{first} {second}" for first, second in zip(words, words[1:]))
        return [phrase for phrase, _ in phrases.most_common(10)]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract content keywords for paragraph-density scoring."""
        stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with"}
        words = re.findall(r"\w+", text.lower())
        counts = Counter(word for word in words if len(word) > 1 and word not in stopwords)
        return [word for word, _ in counts.most_common(10)]


class ResponseCache:
    """P2 optional Redis cache using JSON values and expiring keys."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.enabled = config.get("enabled", False)
        self.ttl = config.get("ttl", 3600)
        self.max_size = config.get("max_size", 10000)
        self.redis_client = None

    async def initialize(self) -> None:
        """If enabled, connect using host/port/db and ping; disable on failure."""
        if not self.enabled:
            return
        try:
            from redis.asyncio import Redis

            self.redis_client = Redis(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                db=self.config.get("db", 0),
                decode_responses=True,
            )
            await self.redis_client.ping()
        except Exception as exc:
            self.enabled = False
            logger.warning("Redis initialization failed; cache disabled: %s", exc)

    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """GET and decode JSON; count hits/misses; disabled/errors return None."""
        if not self.enabled:
            return None
        try:
            value = await self.redis_client.get(cache_key)
            if value is None:
                INFERENCE_METRICS.cache_misses.inc()
                return None
            response = json.loads(value)
            INFERENCE_METRICS.cache_hits.inc()
            return response
        except Exception as exc:
            logger.warning("Redis cache read failed: %s", exc)
            return None

    async def cache_response(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """SETEX JSON with ttl; skip disabled/error responses, contain Redis errors."""
        if not self.enabled or response_data.get("error"):
            return
        try:
            await self.redis_client.setex(cache_key, self.ttl, json.dumps(response_data))
        except Exception as exc:
            logger.warning("Redis cache write failed: %s", exc)

    def generate_cache_key(self, request: QueryRequest, model_name: str) -> str:
        """P2 MD5 of model:query:temperature:max_tokens; exclude user/attachments."""
        value = f"{model_name}:{request.query}:{request.temperature}:{request.max_tokens}"
        return hashlib.md5(value.encode("utf-8")).hexdigest()

class BaseInferenceProvider(ABC):
    """Task 3.6 interface; no SDK clients are created by the framework."""

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any]) -> None:
        self.config = dict(config)
        # Reuse router ModelConfig input/output prices, rather than a second price table.
        self.models = models

    @abstractmethod
    async def initialize(self) -> bool:
        raise NotImplementedError("Task 3.6: provider initialization")

    @abstractmethod
    async def generate_response(self, request: QueryRequest, model_name: str) -> InferenceResponse:
        raise NotImplementedError("Task 3.6: generate_response")

    @abstractmethod
    def stream_response(self, request: QueryRequest, model_name: str) -> AsyncIterator[str]:
        # Concrete providers implement this interface as async generators.
        raise NotImplementedError("Task 3.6: stream_response")

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        raise NotImplementedError("Task 3.6: provider health")

    def _token_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        model = self.models.get(model_name)
        if model is None:
            # P2 unknown-model estimate: reuse a configured model's prices.
            model = next(iter(self.models.values()), None)
            logger.warning("No configured price for '%s'; using provider fallback price", model_name)
        if model is None:
            return 0.0
        return input_tokens * model.cost_input_token + output_tokens * model.cost_output_token

    def _response(self, model_name: str, provider: str, text: str, input_tokens: int,
                  output_tokens: int, cost: float, elapsed: float,
                  finish_reason: str = "stop") -> InferenceResponse:
        return InferenceResponse(
            response_text=text, model_name=model_name, provider=provider,
            token_count_input=input_tokens, token_count_output=output_tokens,
            latency_ms=int(elapsed * 1000),
            tokens_per_second=output_tokens / elapsed if elapsed > 0 else 0.0,
            cost_usd=cost, finish_reason=finish_reason,
        )

class OpenAIProvider(BaseInferenceProvider):
    """OpenAI async client; context as system message; SDK usage for costs."""

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any]) -> None:
        super().__init__(config, models)
        self.client = None

    async def initialize(self) -> bool:
        """Create client/check readiness; unavailable credentials/provider return False."""
        api_key = self.config.get("api_key") or os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            logger.warning("OpenAI API key missing; provider disabled")
            return False
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key, base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout", 60), max_retries=0,
        )
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    async def generate_response(self, request: QueryRequest, model_name: str) -> InferenceResponse:
        """Generate with P2 retries; populate required Schema fields and actual usage."""
        start = perf_counter()
        result = await self.client.chat.completions.create(
            **self._request_params(request, model_name), stream=False,
        )
        choice = result.choices[0]
        return self._response(
            model_name, "openai", choice.message.content or "", result.usage.prompt_tokens,
            result.usage.completion_tokens, self._calculate_cost(result.usage, model_name),
            perf_counter() - start, choice.finish_reason or "stop",
        )

    async def stream_response(self, request: QueryRequest, model_name: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            **self._request_params(request, model_name), stream=True,
        )
        async with stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def _request_params(self, request: QueryRequest, model_name: str) -> Dict[str, Any]:
        messages = []
        if request.context:
            messages.append({"role": "system", "content": request.context})
        messages.append({"role": "user", "content": request.query})
        # Current Chat Completions API supersedes max_tokens with this parameter.
        return dict(model=model_name, messages=messages,
                    max_completion_tokens=request.max_tokens, temperature=request.temperature)

    def _calculate_cost(self, usage: Any, model_name: str) -> float:
        """Use separate input/output USD-per-token prices; self-hosted cost is zero."""
        return self._token_cost(usage.prompt_tokens, usage.completion_tokens, model_name)

    def get_health_status(self) -> Dict[str, Any]:
        return {"status": "healthy" if self.client is not None else "unhealthy"}

class AnthropicProvider(BaseInferenceProvider):
    """Anthropic async client; text blocks and separate input/output usage."""

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any]) -> None:
        super().__init__(config, models)
        self.client = None

    async def initialize(self) -> bool:
        """Create client/check readiness; unavailable credentials/provider return False."""
        api_key = self.config.get("api_key") or os.getenv(self.config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            logger.warning("Anthropic API key missing; provider disabled")
            return False
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(
            api_key=api_key, base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout", 60), max_retries=0,
        )
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    async def generate_response(self, request: QueryRequest, model_name: str) -> InferenceResponse:
        """Generate with P2 retries; populate required Schema fields and actual usage."""
        start = perf_counter()
        result = await self.client.messages.create(**self._request_params(request, model_name))
        text = "".join(block.text for block in result.content if block.type == "text")
        return self._response(
            model_name, "anthropic", text, result.usage.input_tokens, result.usage.output_tokens,
            self._calculate_cost(result.usage, model_name), perf_counter() - start,
            result.stop_reason or "stop",
        )

    async def stream_response(self, request: QueryRequest, model_name: str) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._request_params(request, model_name)) as stream:
            async for text in stream.text_stream:
                yield text

    def _request_params(self, request: QueryRequest, model_name: str) -> Dict[str, Any]:
        prompt = f"Context: {request.context}\n\nQuery: {request.query}" if request.context else request.query
        # Current Anthropic SDK/models no longer accept custom temperature (unlike old P2 examples).
        return dict(model=model_name, max_tokens=request.max_tokens,
                    messages=[{"role": "user", "content": prompt}])

    def _calculate_cost(self, usage: Any, model_name: str) -> float:
        """Use separate input/output USD-per-token prices; self-hosted cost is zero."""
        return self._token_cost(usage.input_tokens, usage.output_tokens, model_name)

    def get_health_status(self) -> Dict[str, Any]:
        return {"status": "healthy" if self.client is not None else "unhealthy"}

class vLLMProvider(BaseInferenceProvider):
    """OpenAI-compatible HTTP endpoint, health check and SSE streaming."""

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any]) -> None:
        super().__init__(config, models)
        self.http_client = None
        self.base_url = self.config.get(
            "base_url", f"http://{self.config.get('host', 'localhost')}:{self.config.get('port', 8001)}/v1"
        ).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"

    async def initialize(self) -> bool:
        """Create client/check readiness; unavailable credentials/provider return False."""
        import httpx

        api_key = self.config.get("api_key") or os.getenv(self.config.get("api_key_env", "VLLM_API_KEY"))
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url + "/", timeout=self.config.get("timeout", 300),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        try:
            health_url = self.base_url.removesuffix("/v1") + "/health"
            response = await self.http_client.get(health_url)
            response.raise_for_status()
        except Exception:
            await self.http_client.aclose()
            self.http_client = None
            raise
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    async def generate_response(self, request: QueryRequest, model_name: str) -> InferenceResponse:
        """Generate with P2 retries; populate required Schema fields and actual usage."""
        start = perf_counter()
        response = await self.http_client.post("completions", json=self._request_params(request, model_name, False))
        response.raise_for_status()
        result = response.json()
        usage = result.get("usage") or {}
        return self._response(
            model_name, "vllm", result["choices"][0]["text"], usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0), self._calculate_cost(usage, model_name),
            perf_counter() - start, result["choices"][0].get("finish_reason") or "stop",
        )

    async def stream_response(self, request: QueryRequest, model_name: str) -> AsyncIterator[str]:
        async with self.http_client.stream(
            "POST", "completions", json=self._request_params(request, model_name, True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if data:
                    result = json.loads(data)
                    if result["choices"]:
                        yield result["choices"][0]["text"]

    def _request_params(self, request: QueryRequest, model_name: str, stream: bool) -> Dict[str, Any]:
        prompt = f"Context: {request.context}\n\nQuery: {request.query}" if request.context else request.query
        return dict(model=model_name, prompt=prompt, max_tokens=request.max_tokens,
                    temperature=request.temperature, stream=stream)

    def _calculate_cost(self, usage: Any, model_name: str) -> float:
        """Use separate input/output USD-per-token prices; self-hosted cost is zero."""
        return 0.0

    def get_health_status(self) -> Dict[str, Any]:
        return {"status": "healthy" if self.http_client is not None else "unhealthy"}

class BatchProcessor:
    """P2 interface; eventual implementation passes through without batch merging."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = config.get("enabled", False)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.max_wait_ms = config.get("max_wait_time_ms", 50)
        self.pending_requests: List[Any] = []
        self.batch_lock = asyncio.Lock()

    async def add_request(
        self, request: QueryRequest, provider: BaseInferenceProvider, model_name: str
    ) -> InferenceResponse:
        # P2 deliberately uses pass-through for both enabled states.
        return await provider.generate_response(request, model_name)


class InferenceEngine:
    """P2 orchestration; config is the inference subsection."""

    def __init__(self, config: Dict[str, Any], router: "ModelRouter") -> None:
        self.config = dict(config)
        self.router = router
        self.context_compressor = ContextCompressor(config.get("compression", {}), router.token_counter)
        self.cache = ResponseCache(config.get("cache", {}))
        self.batch_processor = BatchProcessor(config.get("batching", {}))
        self.providers: Dict[str, BaseInferenceProvider] = {}
        self.inference_stats: Dict[str, Any] = {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_tokens": 0, "total_cost_usd": 0.0, "total_latency_ms": 0,
        }

    async def initialize(self) -> None:
        """Initialize compressor/cache, then providers with config and router.models."""
        await self.context_compressor.initialize()
        await self.cache.initialize()
        for name, provider_class in (("openai", OpenAIProvider), ("anthropic", AnthropicProvider), ("vllm", vLLMProvider)):
            if name not in self.config:
                continue
            models = {key: model for key, model in self.router.models.items() if model.provider == name}
            provider = provider_class(self.config[name], models)
            try:
                if await provider.initialize():
                    self.providers[name] = provider
            except Exception as exc:
                logger.warning("Provider '%s' initialization failed; skipping: %s", name, exc)

    async def process_query(self, request: QueryRequest) -> InferenceResponse:
        """Route -> resolve provider -> cache lookup -> optional compression on copy
        -> batch/provider generation -> cache write -> stats/metrics -> response.
        Cache writes use response.model_dump(mode="json") for Schema timestamps.
        Errors become InferenceResponse(error=...), per P2.
        """
        start = perf_counter()
        # P2 cache keys exclude context: bypass both reads and writes for contextual requests.
        use_cache = self.cache.enabled and not request.context
        model_name = "unknown"
        error_type = None
        try:
            decision = await self.router.route_query(request)
            model_name = decision.selected_model
            model_name, provider = self._resolve_available_model(model_name)
            if use_cache:
                cache_key = self.cache.generate_cache_key(request, model_name)
                cached = await self.cache.get_cached_response(cache_key)
                if cached is not None:
                    return InferenceResponse(**{**cached, "cached": True})

            compressed = False
            compressor = self.context_compressor
            if compressor.enabled and request.context:
                context = await compressor.compress_context(request.context, model_name)
                compressed = context != request.context
                if compressed:
                    request = request.model_copy(update={"context": context})
                    logger.info("Compressed context for model '%s'", model_name)
                    INFERENCE_METRICS.compressions_total.labels(method=compressor.method).inc()

            response = await self.batch_processor.add_request(request, provider, model_name)
            response.compressed_context = compressed
            if use_cache:
                await self.cache.cache_response(cache_key, response.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Inference failed for model '%s': %s", model_name, exc)
            error_type = type(exc).__name__
            response = InferenceResponse(
                response_text="", model_name=model_name, provider="error", error=str(exc),
                token_count_input=0, token_count_output=0, latency_ms=int((perf_counter() - start) * 1000),
                tokens_per_second=0.0, cost_usd=0.0,
            )
        # One telemetry boundary keeps P2's response contract without retrying inference.
        try:
            await self._update_stats(response)
            self.router.update_model_stats(model_name, success=not response.error, latency_ms=response.latency_ms)
            if error_type:
                INFERENCE_METRICS.errors_total.labels(model=model_name, error_type=error_type).inc()
            INFERENCE_METRICS.requests_total.labels(model=model_name, provider=response.provider).inc()
            INFERENCE_METRICS.request_duration.labels(model=model_name, provider=response.provider).observe(perf_counter() - start)
        except Exception as exc:
            logger.warning("Inference statistics update failed: %s", exc)
        return response

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[str]:
        """Route, resolve provider and forward text; errors yield an error string."""
        try:
            decision = await self.router.route_query(request)
            model_name, provider = self._resolve_available_model(decision.selected_model)
            async for text in provider.stream_response(request, model_name):
                yield text
        except Exception as exc:
            logger.warning("Streaming inference failed: %s", exc)
            yield f"Error: {exc}"

    def _get_provider_for_model(self, model_name: str) -> Optional[BaseInferenceProvider]:
        """Read provider name through router.get_model_info and look up providers."""
        info = self.router.get_model_info(model_name)
        return self.providers.get(info["config"]["provider"]) if info else None

    def _resolve_available_model(self, requested: str) -> tuple[str, BaseInferenceProvider]:
        """Prefer requested model, otherwise first available by ascending priority."""
        provider = self._get_provider_for_model(requested)
        if provider is not None:
            return requested, provider
        for name, model in sorted(self.router.models.items(), key=lambda item: item[1].priority):
            provider = self.providers.get(model.provider)
            if provider is not None:
                return name, provider
        raise ValueError("No inference providers are available")

    async def _update_stats(self, response: InferenceResponse) -> None:
        """Update engine totals; router receives latency_ms, metrics observe seconds."""
        stats = self.inference_stats
        stats["total_requests"] += 1
        stats["successful_requests" if not response.error else "failed_requests"] += 1
        stats["total_tokens"] += response.token_count_input + response.token_count_output
        stats["total_cost_usd"] += response.cost_usd
        stats["total_latency_ms"] += response.latency_ms
        for direction, count in (("input", response.token_count_input), ("output", response.token_count_output)):
            INFERENCE_METRICS.tokens_total.labels(model=response.model_name, direction=direction).inc(count)
        INFERENCE_METRICS.cost_usd_total.labels(model=response.model_name).inc(response.cost_usd)

    def get_health_status(self) -> Dict[str, Any]:
        """Aggregate provider health, enabled cache/compression and engine stats."""
        return {
            "providers": {name: provider.get_health_status() for name, provider in self.providers.items()},
            "cache_enabled": self.cache.enabled,
            "compression_enabled": self.context_compressor.enabled,
            "inference_stats": dict(self.inference_stats),
        }

    async def shutdown(self) -> None:
        """Close initialized provider clients and Redis client."""
        for provider in self.providers.values():
            if isinstance(provider, vLLMProvider):
                await provider.http_client.aclose()
            else:
                await provider.client.close()
        self.providers.clear()
        if self.cache.redis_client is not None:
            await self.cache.redis_client.aclose()
            self.cache.redis_client = None
        self.cache.enabled = False
