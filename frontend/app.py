import json
import logging

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage


class ChatApp:
    def __init__(self, agent, debug_tool_calls: bool = True):
        self._agent = agent
        self._debug = debug_tool_calls
        self._logger = logging.getLogger(__name__)

    def _history_to_lc_messages(self, history: list[tuple[str, str]] | None):
        messages: list = []
        if not history:
            return messages

        for item in history:
            user_msg = None
            bot_msg = None

            # Gradio history 可能是 [user, bot] 或 [user, bot, extra...]
            if isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    user_msg = item[0]
                if len(item) >= 2:
                    bot_msg = item[1]
            else:
                # 其他型別先略過，避免解構錯誤
                continue

            if user_msg:
                messages.append(HumanMessage(content=str(user_msg)))
            if bot_msg:
                messages.append(AIMessage(content=str(bot_msg)))

        return messages

    def _extract_tool_calls(self, messages: list) -> list[dict]:
        """從 LangChain messages 擷取 tool call 名稱與參數。"""
        tool_calls: list[dict] = []
        for msg in messages:
            calls = getattr(msg, "tool_calls", None) or []
            for call in calls:
                tool_calls.append(
                    {
                        "name": call.get("name"),
                        "args": call.get("args"),
                    }
                )
        return tool_calls

    def _print_tool_debug(self, tool_calls: list[dict]) -> None:
        """在主終端輸出可視化除錯資訊。"""
        if tool_calls:
            called_tools = [tc.get("name") for tc in tool_calls]
            tool_args = [tc.get("args") for tc in tool_calls]
            self._logger.info("==== Tool Use ====")
            self._logger.info("called_tools: %s", called_tools)
            self._logger.info("tool_args: %s", json.dumps(tool_args, ensure_ascii=False))
            self._logger.info("==================")

    async def _chat_fn(self, message: str, history: list[tuple[str, str]]):
        messages = self._history_to_lc_messages(history)
        messages.append(HumanMessage(content=message))
        result = await self._agent.ainvoke({"messages": messages})

        # create_agent 回傳的通常是帶有 messages 的 dict state
        if isinstance(result, dict) and "messages" in result:
            msgs = result["messages"]
            if self._debug:
                tool_calls = self._extract_tool_calls(msgs if isinstance(msgs, list) else [])
                self._print_tool_debug(tool_calls)
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                content = getattr(last, "content", None)
                if content:
                    return content
        return str(result)

    def run(self):
        demo = gr.ChatInterface(
            fn=self._chat_fn,
            title="AI Agent (LangChain + Tools)",
            description="A toy AI agent, authored by Chia-An Yang",
        )
        demo.launch()
