"""
Models and types for Speculative RAG
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class RagDraftingResponse(BaseModel):
    """Response from the RAG drafting model"""
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.") 