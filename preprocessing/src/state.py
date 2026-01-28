"""
Pydantic state model for multi-agent workflow.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AgentState(BaseModel):
    """多智能体协作的共享状态（LangGraph State）"""
    
    # 输入字段
    task_id: str
    task_type: Literal["exam", "dialogue"]
    raw_content: str
    answer: Optional[str] = None  # 选择题答案（仅exam类型）
    
    # 过程字段
    system_prompt: str = ""
    draft_response: str = ""
    review_opinion: str = ""
    revision_count: int = 0
    
    # 输出字段
    final_response: str = ""
    
    # 元数据
    tokens_used: int = 0
    api_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
