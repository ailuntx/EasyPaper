"""
ReAct Agent Base Class.

Provides a ReAct (Reasoning + Acting) loop using OpenAI Function Calling
protocol. Agents that inherit from this class can let the LLM autonomously
decide when and how to call tools during generation and review.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseAgent
from .shared.llm_client import LLMClient
from .shared.tools.base import WriterTool, ToolResult
from .shared.tools.registry import ToolRegistry
from ..config.schema import ModelConfig, ToolsConfig


class ReActAgent(BaseAgent):
    """
    Base class for agents that support ReAct-style tool use.

    - **Description**:
        - Extends BaseAgent with a built-in ToolRegistry and a `react_loop()`
          method that implements the ReAct pattern using OpenAI Function Calling.
        - Provides `setup_tools()` to create and register tool instances from
          config-defined tool names and runtime context.
        - Subclasses should call `setup_tools()` with appropriate context
          before invoking `react_loop()`.

    - **Args**:
        - `config` (ModelConfig): LLM model configuration.
        - `tools_config` (ToolsConfig, optional): Tool availability configuration.
    """

    def __init__(self, config: ModelConfig, tools_config: Optional[ToolsConfig] = None):
        self.client = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.tools_config = tools_config or ToolsConfig()
        self.tool_registry = ToolRegistry()

    def setup_tools(self, tool_names: Optional[List[str]] = None, **context) -> None:
        """
        Create and register tool instances based on config and runtime context.
        - **Description**:
            - Uses the global TOOL_FACTORY to create tool instances.
            - Clears any previously registered tools and registers fresh ones.
            - If tool_names is not provided, uses tools from config.

        - **Args**:
            - `tool_names` (List[str], optional): Tool names to register.
              Defaults to self.tools_config.available_tools.
            - `**context`: Runtime context passed to tool factories
              (e.g., valid_keys, key_points).
        """
        from .shared.tools import TOOL_FACTORY

        names = tool_names or self.tools_config.available_tools
        if not names:
            return

        self.tool_registry.clear()
        for name in names:
            factory = TOOL_FACTORY.get(name)
            if factory is None:
                print(f"[ReActAgent] WARNING: Unknown tool '{name}', skipping")
                continue
            try:
                tool = factory(context)
                self.tool_registry.register_or_replace(tool)
                print(f"[ReActAgent] Registered tool: {name}")
            except Exception as e:
                print(f"[ReActAgent] ERROR: Failed to create tool '{name}': {e}")

    async def react_loop(
        self,
        messages: List[Dict[str, Any]],
        tool_names: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model_name: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a ReAct loop using OpenAI Function Calling.
        - **Description**:
            - Calls the LLM with the `tools` parameter.
            - If the LLM responds with tool_calls, executes them via the
              registry, appends tool results to messages, and loops.
            - If the LLM responds with content (no tool_calls), returns
              the final content.
            - Stops after max_iterations to prevent infinite loops.

        - **Args**:
            - `messages` (List[dict]): Conversation messages (system + user).
            - `tool_names` (List[str], optional): Subset of registered tools
              to make available. If None, all registered tools are used.
            - `max_iterations` (int, optional): Maximum ReAct loop iterations.
              Defaults to config value or 5.
            - `temperature` (float): LLM temperature. Default 0.7.
            - `max_tokens` (int): Max tokens for LLM response. Default 4000.
            - `model_name` (str, optional): Override model name.

        - **Returns**:
            - `Tuple[str, List[dict]]`: Final content string and the full
              message history including tool calls.
        """
        max_iter = max_iterations or self.tools_config.max_react_iterations
        model = model_name or self.model_name

        # Get OpenAI-format tool definitions
        openai_tools = self.tool_registry.get_openai_tools(tool_names)
        available_tool_names = [t["function"]["name"] for t in openai_tools]
        if not openai_tools:
            # No tools available, just do a normal LLM call
            print("[ReActAgent] No tools registered, doing plain LLM call")
            return await self._plain_llm_call(messages, temperature, max_tokens, model)

        print(f"[ReActAgent] Starting ReAct loop (max_iter={max_iter}, "
              f"model={model}, tools={available_tool_names})")

        # Copy messages to avoid mutating the original
        working_messages = list(messages)
        iteration = 0
        total_tool_calls = 0
        loop_start = time.time()

        while iteration < max_iter:
            iteration += 1
            iter_start = time.time()
            print(f"[ReActAgent] ── Iteration {iteration}/{max_iter} ──")

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=working_messages,
                    tools=openai_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                llm_elapsed = time.time() - iter_start
                print(f"[ReActAgent]   LLM responded in {llm_elapsed:.1f}s")
            except Exception as e:
                print(f"[ReActAgent]   LLM call FAILED: {e}")
                return f"% Error in ReAct loop: {str(e)}", working_messages

            choice = response.choices[0]
            assistant_message = choice.message

            # Append assistant message to history
            # Convert to dict for serialization
            msg_dict = {"role": "assistant", "content": assistant_message.content}
            if assistant_message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            working_messages.append(msg_dict)

            # Check if LLM wants to call tools
            if assistant_message.tool_calls:
                n_calls = len(assistant_message.tool_calls)
                total_tool_calls += n_calls
                print(f"[ReActAgent]   LLM requested {n_calls} tool call(s):")
                for i, tc in enumerate(assistant_message.tool_calls, 1):
                    # Truncate long arguments for readability
                    args_str = tc.function.arguments or "{}"
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"[ReActAgent]     ({i}) {tc.function.name}({args_str})")

                # Execute tools with timing
                tool_exec_start = time.time()
                tool_messages = await self.tool_registry.execute_tool_calls(
                    assistant_message.tool_calls
                )
                tool_elapsed = time.time() - tool_exec_start
                print(f"[ReActAgent]   Tool execution completed in {tool_elapsed:.1f}s")

                # Log tool results summary
                for tm in tool_messages:
                    try:
                        import json
                        result_data = json.loads(tm["content"])
                        success = result_data.get("success", "?")
                        msg = result_data.get("message", "")
                        if len(msg) > 150:
                            msg = msg[:150] + "..."
                        status = "✓" if success else "✗"
                        print(f"[ReActAgent]     {status} {msg}")
                    except (json.JSONDecodeError, KeyError):
                        pass

                working_messages.extend(tool_messages)
                # Continue loop to let LLM process tool results
            else:
                # No tool calls = LLM is done reasoning, return content
                final_content = assistant_message.content or ""
                total_elapsed = time.time() - loop_start
                content_preview = final_content[:100].replace('\n', ' ')
                print(f"[ReActAgent] ── ReAct loop completed ──")
                print(f"[ReActAgent]   Iterations: {iteration}, Tool calls: {total_tool_calls}, "
                      f"Total time: {total_elapsed:.1f}s")
                print(f"[ReActAgent]   Output: {len(final_content)} chars, "
                      f"preview: \"{content_preview}...\"")
                return final_content, working_messages

        # Max iterations reached - return whatever content we have
        total_elapsed = time.time() - loop_start
        print(f"[ReActAgent] WARNING: Max iterations ({max_iter}) reached "
              f"after {total_elapsed:.1f}s, {total_tool_calls} tool calls")
        # Make one final call without tools to get a response
        final_content, _ = await self._plain_llm_call(
            working_messages, temperature, max_tokens, model
        )
        return final_content, working_messages

    async def _plain_llm_call(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model_name: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Make a plain LLM call without tools.
        - **Description**:
            - Simple wrapper for chat.completions.create without tools.
            - Used as fallback when no tools are registered or after
              max iterations.

        - **Args**:
            - `messages` (List[dict]): Conversation messages.
            - `temperature` (float): LLM temperature.
            - `max_tokens` (int): Max tokens.
            - `model_name` (str, optional): Override model name.

        - **Returns**:
            - `Tuple[str, List[dict]]`: Content and updated messages.
        """
        model = model_name or self.model_name
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content, messages + [{"role": "assistant", "content": content}]
        except Exception as e:
            print(f"[ReActAgent] Plain LLM call failed: {e}")
            return f"% Error: {str(e)}", messages
