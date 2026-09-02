"""
Pydantic Schema

Enum:
- UserTier
- QueryType
- AttachmentType

Models:
- Attachment
- QueryRequest
- InferenceResponse
- RoutingDecision
- ModelSelection
- ModelConfig
- SystemMetric
- UserSession

Constraint:
- No import from logger/metrics/router etc. 
- Validator no network connection, no IO
- All values need to be >= 0

"""

import re 
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
from typing import Optional,List, Dict,Any
from datetime import datetime

## ----------Enum----------- ##

class UserTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class QueryType(str, Enum):
    GENERAL = "general"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    CREATIVE_WRITING = "creative_writing"
    BRAINSTORMING = "brainstorming"
    PLANNING = "planning"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    MATH = "math"
    REASONING = "reasoning"

class AttachmentType(str, Enum):
    FILE = "file"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"

## ---------Model------------ ##

class Attachment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: AttachmentType
    size_bytes: int
    mime_type : str
    url: Optional[str] = None
    content: Optional[bytes] = None

    @field_validator("size_bytes")
    @classmethod
    def check_size(cls, v:int) -> int:
        if v <= 0:
            raise ValueError("size_bytes need to > 0")
        if v > 100_000_000:
            raise ValueError("size_bytes exceeds 100MB limit")
        return v


class QueryRequest(BaseModel):
    ## Necessary
    query: str = Field(min_length=1, max_length=50000)
    user_id: str = Field(min_length=1)

    ## Core with default
    user_tier: UserTier = UserTier.FREE
    request_id: str = Field(default_factory= lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    ## Optional
    context: Optional[str] = Field(default=None, max_length=100000)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    priority: int = Field(default=1, ge=1, le=5)

    ## Attachments
    attachments: List[Attachment] = []

    ## metadata
    # Session_id and conversation_id need to be optional 
    # P2 /route default only have:
    # query 、 user_id 、 user_tier 、 
    # context 、 max_tokens 、 temperature 
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    @field_validator("attachments")
    @classmethod
    def check_attachments(cls, v:List[Attachment]) -> List[Attachment]:
        if len(v) > 10:
            raise ValueError("Attachments exceed 10 items")
        return v

    @field_validator("query")
    @classmethod
    def check_query(cls, v:str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only.")
        return stripped


class InferenceResponse(BaseModel):
    response_text: str
    model_name: str
    provider: str

    ## Token
    token_count_input: int = Field(ge=0)
    token_count_output: int = Field(ge=0)
    total_tokens : Optional[int] = Field(default=None, ge=0)

    ## performance
    latency_ms : int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)

    ## cost
    cost_usd : float = Field(ge=0.0)

    ## system
    cached: bool = Field(default=False)
    compressed_context:  bool = Field(default=False)

    ## quality (tempraory)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    safety_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    ## error
    error: Optional[str] = None
    finish_reason: str = Field(default="stop")

    ## timestamp
    timestamp: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def fill_total_tokens(self) -> "InferenceResponse":
        if self.total_tokens is None:
            self.total_tokens = self.token_count_input + self.token_count_output
        return self 

    
class RoutingDecision(BaseModel):
    selected_model: str = Field(min_length=1) # type is str? 
    query_type: QueryType
    routing_reason: str = Field(min_length=1)

    # metrics
    token_count: int = Field(ge=0)
    estimated_cost: float = Field(ge=0.0)
    routing_time_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

    timestamp: datetime = Field(default_factory=datetime.now)
    fallback_models: List[str] = []
    routing_strategy: str = "intelligent"
    user_tier: UserTier = UserTier.FREE


class ModelSelection(BaseModel):
    model_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str



VALID_CAPABILITIES = frozenset({
    "reasoning", "coding", "analysis", "writing",
    "creative", "general", "math", "translation",
})

class ModelConfig(BaseModel):
    name: str
    provider: str
    max_tokens: int = Field(ge=1)
    cost_per_token: float = Field(ge=0.0)
    priority: int = Field(ge=1)
    capabilities: List[str] = Field(min_length=1)

    # Optional
    model_path:  Optional[str] = None
    gpu_memory_gb:  Optional[int] = Field(default=None, ge=1)
    api_key_env: Optional[str] = None

    @field_validator("capabilities")
    @classmethod
    def check_capabilites(cls, v: List[str]) -> List[str]:
        for i in v:
            if i not in VALID_CAPABILITIES:
                raise ValueError(f"Capabilitie {i} not in required range")
        return v


METRIC_NAME_RE = re.compile(r"[A-Za-z0-9_.]+")

class SystemMetric(BaseModel):
    name: str = Field(min_length=1)
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    labels: Dict[str,str]
    unit: Optional[str] = None

    @field_validator("name")
    @classmethod
    def check_name(cls, v:str) -> str:
        if not METRIC_NAME_RE.fullmatch(v):
            raise ValueError("Metric name must consist of alphanumeric + _ + .")
        return v

class UserSession(BaseModel):
    session_id: str = Field(default_factory= lambda: str(uuid.uuid4()))
    user_id: str
    user_tier: UserTier
    start_time: datetime = Field(default_factory=datetime.now)
        