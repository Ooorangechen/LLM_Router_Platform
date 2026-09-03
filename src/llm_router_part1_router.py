from src.utils.logger import get_logger
from src.utils.schema import QueryType
from typing import Dict, List, Tuple

logger = get_logger(__name__)

class QueryClassifier:
    def __init__(self):
        self.patterns = self._init_patterns()
        self.key


    async initialize(self) -> None:
        self._is_initialized = True
    
    def classify_query(query: str) -> Tuple[QueryType, float]: