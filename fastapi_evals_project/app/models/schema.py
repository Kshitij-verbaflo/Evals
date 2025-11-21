from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict


class EvalRow(BaseModel):
    transcript: Optional[str]
    lead_data: Optional[str]
    latest_message: Optional[str]
# extend with other columns as needed


class EvalResult(BaseModel):
    original_row: Dict[str, Any]
    llm_output: Optional[Dict[str, Any]]
    feedback: Optional[str]
    success: bool = True