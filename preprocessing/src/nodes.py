"""
LangGraph async nodes for Generator/Reviewer/Revisor.
"""
import yaml
from jinja2 import Template
from .state import AgentState
from .api_router import APIRouter
from .logger import logger


def load_prompt(task_type: str, node_name: str, context: dict) -> tuple[str, str]:
    """从统一的 prompts.yaml 加载并渲染 Prompt
    
    Args:
        task_type: 任务类型 (exam/dialogue)
        node_name: 节点名称 (persona/generator/reviewer/revisor)
        context: 渲染模板的上下文
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    node_config = config.get(task_type, {}).get(node_name, {})
    
    system_prompt = node_config.get("system", "")
    user_template = node_config.get("user_template", "")
    user_prompt = Template(user_template).render(**context)
    
    return system_prompt, user_prompt


async def persona_node(state: AgentState, api_router: APIRouter) -> AgentState:
    """动态生成 System Persona（异步）"""
    try:
        # 从统一配置文件加载 Persona 生成所需的 Prompt
        system_prompt, user_prompt = load_prompt(state.task_type, "persona", state.model_dump())
        
        # 使用 base 级别模型生成 Persona
        persona, tokens = await api_router.call_async(
            tier="base",
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        state.system_prompt = persona.strip()
        state.tokens_used += tokens
        state.api_calls += 1
    except Exception as e:
        state.errors.append(f"Persona generation failed: {str(e)}")
        # 回退到默认
        state.system_prompt = "你是一名专业的医学 AI 助手，擅长通过严密的临床思维给出精炼的建议。"
        
    return state


async def generator_node(state: AgentState, api_router: APIRouter) -> AgentState:
    """生成初稿（异步）"""
    system_prompt, user_prompt = load_prompt(state.task_type, "generator", state.model_dump())
    
    try:
        response, tokens = await api_router.call_async(
            tier="fast",
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        state.draft_response = response
        state.tokens_used += tokens
        state.api_calls += 1
    except Exception as e:
        state.errors.append(f"Generator failed: {str(e)}")
        logger.error(f"[Node] {state.task_id} -> Generator 失败")
    
    return state


async def reviewer_node(state: AgentState, api_router: APIRouter) -> AgentState:
    """质量审核（异步）"""
    system_prompt, user_prompt = load_prompt(state.task_type, "reviewer", state.model_dump())
    
    try:
        response, tokens = await api_router.call_async(
            tier="smart",
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        state.review_opinion = response
        
        # --- 硬编码规则约束 (Fail Fast) ---
        # 如果 Generator 彻底漏掉了标签，而 Reviewer 却给了“通过”，则强制驳回
        has_think = "<think>" in state.draft_response and "</think>" in state.draft_response
        if not has_think and "通过" in state.review_opinion:
            state.review_opinion = "[硬性格式错误]\n[具体问题]：Generator 的回复完全缺失 <think></think> 标签。\n[修改建议]： 使用 <think> 标签包裹，并对推理过程进行润色。"
        
        state.tokens_used += tokens
        state.api_calls += 1
    except Exception as e:
        state.errors.append(f"Reviewer failed: {str(e)}")
        logger.error(f"[Node] {state.task_id} -> Reviewer 失败")
    
    return state


async def revisor_node(state: AgentState, api_router: APIRouter) -> AgentState:
    """修订优化（异步）"""
    system_prompt, user_prompt = load_prompt(state.task_type, "revisor", state.model_dump())
    
    try:
        response, tokens = await api_router.call_async(
            tier="strong",
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        state.final_response = response
        state.tokens_used += tokens
        state.api_calls += 1
        state.revision_count += 1
    except Exception as e:
        state.errors.append(f"Revisor failed: {str(e)}")
        logger.error(f"[Node] {state.task_id} -> Revisor 失败")
    
    return state


def should_revise(state: AgentState) -> str:
    """决定是否需要修订（在审核节点后调用）"""
    if "通过" in state.review_opinion:
        return "accept"
    return "revise"


def check_revision_limit(state: AgentState) -> str:
    """检查修订次数限制（在修订节点后调用）
    
    目的是在达到最大修订次数后直接结束，避免无意义的最后一次审核。
    """
    if state.revision_count >= 2:
        return "end"
    return "continue"
