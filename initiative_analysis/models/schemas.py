from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class ContentItem:
    """Data model for content items stored in the database."""
    id: Optional[int] = None
    link: str = ""
    title: str = ""
    date: Optional[datetime] = None
    summary: Optional[str] = None
    full_content: Optional[str] = None
    themes: List[str] = field(default_factory=list)
    organization: Optional[str] = None
    sentiment: str = "Neutral"
    language: str = "English"
    initiative: Optional[str] = None
    initiative_key: Optional[str] = None
    benefit_categories: Optional[Dict[str, float]] = None
    benefit_examples: Optional[List[Dict[str, Any]]] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class SearchResult:
    """Data model for search results."""
    title: str = ""
    link: str = ""
    snippet: str = ""
    position: int = 0
    date: Optional[str] = None
    source: Optional[str] = None
    preliminary_relevance: float = 0.0
