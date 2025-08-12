"""
prompts.py
Prompt templates for the RAG chat workflow
"""

from langchain_core.prompts import ChatPromptTemplate


ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are analyzing why retrieved documents did not meet the relevance threshold for a RAG query.

Query: {query}
Maximum Relevance Score: {max_score}
Threshold: {threshold}

The documents were deemed not relevant enough. Provide a brief analysis of why the retrieved documents 
might not be suitable for answering this query. This will help inform the general response.

Keep your analysis concise and factual.
""")


GENERAL_PROMPT = ChatPromptTemplate.from_template("""
The user asked a question but the retrieved documents from the knowledge base were not relevant enough to provide a reliable answer.

User Query: {query}
Document Relevance Issue: {context}

Please respond in one of these ways:
1. If you can answer from your general knowledge, provide a helpful response and mention that this is from general knowledge, not the specific documents
2. If the query is very specific to documents that should be in the knowledge base, suggest the user rephrase their question or provide more specific terms
3. If it's a general conversation query, respond naturally

Be helpful and honest about the limitations.

Response:
""")


SUMMARIZER_PROMPT = ChatPromptTemplate.from_template("""
Based on the following highly relevant documents (relevance score above threshold), provide a concise and accurate answer to the user's question.

User Question: {query}

Relevant Documents:
{context}

Instructions:
- Provide a clear, concise answer based on the information in the documents
- The documents have been verified as relevant to the query
- Be factual and cite specific information from the documents when possible
- Keep the response conversational and helpful

Answer:
""")