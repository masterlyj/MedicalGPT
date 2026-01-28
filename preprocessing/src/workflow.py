"""
LangGraph workflow definition.
"""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    persona_node, 
    generator_node, 
    reviewer_node, 
    revisor_node, 
    should_revise,
    check_revision_limit
)
from .api_router import APIRouter


def create_workflow(api_router: APIRouter) -> StateGraph:
    """创建LangGraph工作流"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    async def _persona(state: AgentState) -> AgentState:
        return await persona_node(state, api_router)

    async def _generator(state: AgentState) -> AgentState:
        return await generator_node(state, api_router)
    
    async def _reviewer(state: AgentState) -> AgentState:
        return await reviewer_node(state, api_router)
    
    async def _revisor(state: AgentState) -> AgentState:
        return await revisor_node(state, api_router)
    
    workflow.add_node("persona", _persona)
    workflow.add_node("generate", _generator)
    workflow.add_node("review", _reviewer)
    workflow.add_node("revise", _revisor)
    
    # 设置入口
    workflow.set_entry_point("persona")
    
    # 添加边
    workflow.add_edge("persona", "generate")
    workflow.add_edge("generate", "review")
    
    # 条件路由：根据审核结果决定是否修订
    workflow.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "revise",
            "accept": END
        }
    )
    
    # 修订后逻辑优化：如果已达到最大修订次数，直接结束，不再进行无意义的最后一次审核
    workflow.add_conditional_edges(
        "revise",
        check_revision_limit,
        {
            "continue": "review",
            "end": END
        }
    )
    
    return workflow.compile()
