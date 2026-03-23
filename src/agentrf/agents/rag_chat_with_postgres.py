from __future__ import annotations

from typing import Callable, Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph import MessagesState


class RAGChatWithPostgres:
    def __init__(
        self,
        llm: BaseChatModel,
        retriever,
        system_prompt: str,
        db_uri: str,
        top_k: int = 5,
        prompt: ChatPromptTemplate | None = None,
    ) -> None:
        """
        RAG chat agent with Postgres-backed LangGraph memory.

        Parameters
        ----------
        llm:
            Already instantiated chat model.
        retriever:
            Retriever object exposing a `retrieve(query, top_k=...)` method.
        system_prompt:
            System prompt string used to build the chat prompt.
        db_uri:
            Postgres connection URI for LangGraph checkpointing.
        top_k:
            Number of documents to retrieve per query.
        prompt:
            Optional fully constructed prompt template. If omitted, a default
            prompt is built from `system_prompt`.
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.db_uri = db_uri
        self.top_k = top_k

        self.prompt = prompt or self._build_prompt()

        self._checkpointer_cm = PostgresSaver.from_conn_string(self.db_uri)
        self.checkpointer = self._checkpointer_cm.__enter__()
        self.checkpointer.setup()

        self.graph = self._build_graph()

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Kontext:\n{context}\n\n"
                    "Bisheriger Chat:\n{history}\n\n"
                    "Frage: {query}\n\n"
                    "Antwort:",
                ),
            ]
        )

    def _format_history(self, messages: Sequence[BaseMessage]) -> str:
        return "\n".join(
            f"{getattr(msg, 'type', 'MESSAGE').upper()}: "
            f"{getattr(msg, 'content', '')}"
            for msg in messages[:-1]
        )

    def _format_context(self, docs: Sequence[Document]) -> str:
        if not docs:
            return "Keine relevanten Dokumente gefunden."

        return "\n\n".join(
            f"--- {doc.metadata.get('filename', 'Unknown')} "
            f"(chunk {doc.metadata.get('chunk_id', '?')}) ---\n"
            f"{doc.page_content}"
            for doc in docs
        )

    def _chatbot(self, state: MessagesState):
        messages = state["messages"]
        user_query = messages[-1].content

        history = self._format_history(messages)
        docs = self.retriever.retrieve(user_query, top_k=self.top_k)
        context = self._format_context(docs)

        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "query": user_query,
                "context": context,
                "history": history,
            }
        )

        return {"messages": [response]}

    def _build_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("rag_chat", self._chatbot)
        builder.add_edge(START, "rag_chat")
        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, user_input: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        final_text_parts: list[str] = []

        for chunk in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="updates",
        ):
            for node_data in chunk.values():
                if "messages" not in node_data:
                    continue

                for msg in node_data["messages"]:
                    text = getattr(msg, "content", "")
                    if text:
                        final_text_parts.append(text)

        return "".join(final_text_parts)

    def stream_answer(self, user_input: str, thread_id: str) -> Iterator[str]:
        config = {"configurable": {"thread_id": thread_id}}

        for message_chunk, _metadata in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            text = getattr(message_chunk, "content", "")
            if text:
                yield text

    def close(self) -> None:
        if getattr(self, "_checkpointer_cm", None) is not None:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass