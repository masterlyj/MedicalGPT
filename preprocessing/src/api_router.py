"""
Async API router with multi-source support (Groq/Gemini/DeepSeek).
"""
import os
import json
import asyncio
from typing import Literal, Optional, Dict
import aiohttp
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from .logger import logger

load_dotenv()


class APIRouter:
    """异步API路由器 - 支持自动故障转移"""
    
    def __init__(self, config_path: str | None = None):
        from pathlib import Path
        
        # 配置文件路径
        if config_path is None:
            self.config_file = Path(__file__).parent.parent / "config" / "models.yaml"
        else:
            self.config_file = Path(config_path)
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.providers = {
            "base": self._prepare_providers(config.get("base", [])),
            "fast": self._prepare_providers(config.get("fast", [])),
            "smart": self._prepare_providers(config.get("smart", [])),
            "strong": self._prepare_providers(config.get("strong", []))
        }

        # 初始化全局 Session，复用连接
        self.session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=60)
        )

    async def close(self):
        """关闭 Session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _prepare_providers(self, provider_configs: list) -> list:
        """准备provider配置（替换环境变量）"""
        prepared = []
        for config in provider_configs:
            provider = config.copy()
            # 替换API key中的环境变量
            if 'api_key' in provider:
                env_var = provider['api_key']
                if env_var.startswith('${') and env_var.endswith('}'):
                    env_name = env_var[2:-1]
                    provider['api_key'] = os.getenv(env_name, '')
            prepared.append(provider)
        return prepared

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(min=2, max=15),
        retry=retry_if_exception_type(RuntimeError)
    )
    async def call_async(
        self,
        tier: Literal["base", "fast", "smart", "strong"],
        prompt: str,
        system_prompt: str = ""
    ) -> tuple[str, int]:
        """异步调用（自动故障转移）"""
        providers = self.providers.get(tier, [])
        if not providers:
            raise RuntimeError(f"No providers configured for tier: {tier}")
        
        last_error = None
        for provider in providers:
            try:
                if not provider.get('api_key'):
                    continue
                
                # 直接发起请求，依靠 API 自行返回错误触发切换
                response, tokens = await self._call_provider(
                    provider, prompt, system_prompt
                )
                
                return response, tokens
            except Exception as e:
                last_error = e
                # 记录为 debug 级别，避免干扰正常进度条显示
                logger.debug(f"[Router] {provider['name']} 异常: {str(e)[:50]}")
                continue
        
        raise RuntimeError(f"所有{tier}梯度的provider都失败。最后一次错误: {last_error}")
    
    async def _call_provider(
        self, 
        provider: dict, 
        prompt: str, 
        system: str
    ) -> tuple[str, int]:
        """调用单个provider（支持OpenAI协议，兼容Groq, Deepseek, 硅基流动、智谱等）"""
        headers = {"Authorization": f"Bearer {provider['api_key']}"}
        
        # 标准OpenAI兼容接口
        url = f"{provider['base_url']}/chat/completions"
        payload = {
            "model": provider["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": provider.get("temperature", 0.7)
        }
        
        async with self.session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                    
                raise RuntimeError(f"API error {resp.status}: {error_text}")
            
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return content, tokens
