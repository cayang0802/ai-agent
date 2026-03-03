from __future__ import annotations

import json
import logging
import os

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage


class ChatApp:
    def __init__(self, agent, indexer=None, debug_tool_calls: bool = True):
        self._agent = agent
        self._indexer = indexer
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

    def _upload_pdf(self, file):
        """Handle PDF upload: index the file and return a status string."""
        if self._indexer is None:
            return "PDF indexing is not configured."
        if file is None:
            return "No file uploaded."
        try:
            n_chunks = self._indexer.add_file_to_db(file)
            filename = os.path.basename(file)
            return f"Indexed {n_chunks} chunks from {filename}"
        except Exception as exc:
            self._logger.exception("PDF indexing failed")
            return f"Indexing failed: {exc}"

    def run(self):
        with gr.Blocks(title="AI Agent (LangChain + Tools)") as demo:
            gr.Markdown("## AI Agent (LangChain + Tools)\nA toy AI agent, authored by Chia-An Yang")

            with gr.Row():
                pdf_upload = gr.File(
                    file_types=[".pdf"],
                    label="Upload PDF",
                    type="filepath",
                )
                pdf_status = gr.Textbox(
                    label="Indexing Status",
                    interactive=False,
                    placeholder="Upload a PDF to index it…",
                )

            pdf_upload.upload(
                fn=self._upload_pdf,
                inputs=pdf_upload,
                outputs=pdf_status,
            )

            gr.ChatInterface(
                fn=self._chat_fn,
            )

        demo.launch()
