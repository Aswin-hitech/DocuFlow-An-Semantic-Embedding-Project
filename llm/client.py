import os
import re
from typing import List, Dict, Any, Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient:
    """
    RAG Generation & Conversational AI Client for DocuFlow.
    Synthesizes intelligent answers from retrieved semantic contexts.
    Supports OpenAI-compatible APIs (OpenAI, NVIDIA NIM, Groq, Ollama, etc.)
    with a built-in offline extractive synthesis engine fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        base_url: Optional[str] = None
    ):
        try:
            import dotenv
            dotenv.load_dotenv()
        except Exception:
            pass

        self.api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        self.model = (model or os.getenv("LLM_MODEL", "gpt-4o-mini")).strip()
        
        try:
            self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        except (ValueError, TypeError):
            self.temperature = temperature

        raw_base_url = base_url or os.getenv("BASE_URL", "")
        self.base_url = raw_base_url.strip().rstrip("/") if raw_base_url else None
        self.client = None

        if OPENAI_AVAILABLE and self.api_key:
            try:
                client_kwargs = {
                    "api_key": self.api_key,
                    "timeout": 35.0
                }
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM client: {e}")
                self.client = None

    def build_context_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        source_filter: Optional[str] = None
    ) -> str:
        """
        Format retrieved chunks into a clean context block for prompt engineering.
        """
        context_lines = []
        for i, chunk_info in enumerate(context_chunks, start=1):
            text = chunk_info.get("chunk") or chunk_info.get("text", "")
            meta = chunk_info.get("metadata", {})
            source = meta.get("source") or meta.get("doc_id") or meta.get("original_filename") or "Document"
            score = chunk_info.get("score")
            score_str = f" (Cosine Similarity: {score:.3f})" if score is not None else ""
            context_lines.append(f"--- [Source {i}: {source}{score_str}] ---\n{text}")

        formatted_context = "\n\n".join(context_lines)
        filter_note = f" (User filtered knowledge to source: '{source_filter}')" if source_filter and source_filter != "all" else ""

        return (
            f"You are DocuFlow AI, an intelligent, objective, and precise research assistant{filter_note}.\n"
            f"Answer the user's question clearly, thoroughly, and factually based on the retrieved context below.\n\n"
            f"Instructions:\n"
            f"1. Directly address the user's question using relevant information from the retrieved passages.\n"
            f"2. Cite reference numbers (e.g. [Source 1], [Source 2]) when stating specific facts, figures, or claims.\n"
            f"3. Use structured formatting with markdown (bullet points, clear paragraphs, code blocks if appropriate).\n"
            f"4. If the retrieved context does not contain enough information to answer the question, clearly state what information is missing.\n\n"
            f"=== RETRIEVED CONTEXT ===\n"
            f"{formatted_context}\n"
            f"=========================\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer for a query given retrieved semantic chunks.
        Supports conversational history for multi-turn dialogue.
        """
        if not retrieved_chunks:
            filter_text = f" for '{source_filter}'" if source_filter and source_filter != "all" else ""
            return {
                "answer": f"I couldn't find any relevant passages in the indexed documents{filter_text} matching your question. Please verify the document or webpage has been ingested properly in the Ingestion tab.",
                "model": self.model if self.client else "offline-fallback",
                "retrieved_count": 0,
                "sources": []
            }

        # If live OpenAI / NVIDIA client is configured
        if self.client:
            try:
                prompt = self.build_context_prompt(query, retrieved_chunks, source_filter)
                sys_msg = system_instruction or (
                    "You are DocuFlow AI, an expert semantic research assistant. "
                    "You provide grounded, highly accurate answers with clear citations based solely on retrieved documents."
                )

                messages = [{"role": "system", "content": sys_msg}]

                # Append previous dialogue turns for multi-turn conversational awareness
                if history and isinstance(history, list):
                    for turn in history[-6:]:
                        role = turn.get("role")
                        content = turn.get("content")
                        if role in ["user", "assistant"] and content:
                            messages.append({"role": role, "content": content})

                # Append current context-injected prompt
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024
                )
                choice = response.choices[0]
                msg = choice.message
                answer = msg.content
                if not answer and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    answer = msg.reasoning_content
                elif not answer and getattr(msg, "model_extra", None) and msg.model_extra.get("reasoning_content"):
                    answer = msg.model_extra.get("reasoning_content")

                if not answer:
                    answer = "No response generated from LLM."

                return {
                    "answer": answer.strip(),
                    "model": self.model,
                    "retrieved_count": len(retrieved_chunks),
                    "sources": retrieved_chunks
                }
            except Exception as e:
                print(f"LLM API completion error: {e}. Falling back to offline synthesis engine.")

        # Offline extractive synthesis fallback
        return self._offline_synthesize(query, retrieved_chunks)

    def _offline_synthesize(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Intelligent offline extractive synthesis when LLM API is unavailable.
        Extracts key sentences matching query terms from top chunks and presents a synthesized summary.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {"what", "is", "a", "an", "the", "how", "do", "does", "in", "of", "and", "or", "to", "for", "with", "about"}
        keywords = query_words - stopwords

        top_chunk = retrieved_chunks[0]
        top_text = top_chunk.get("chunk") or top_chunk.get("text", "")
        top_score = top_chunk.get("score", 0.0)

        # Extract sentences from retrieved chunks
        candidate_sentences = []
        for i, chunk_info in enumerate(retrieved_chunks[:3], start=1):
            text = chunk_info.get("chunk") or chunk_info.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(keywords & s_words)
                candidate_sentences.append({
                    "sentence": s_clean,
                    "overlap": overlap,
                    "source_index": i
                })

        candidate_sentences.sort(key=lambda x: x["overlap"], reverse=True)
        key_points = [cs["sentence"] for cs in candidate_sentences if cs["overlap"] > 0][:3]

        if not key_points and candidate_sentences:
            key_points = [candidate_sentences[0]["sentence"]]

        lines = [
            f"**DocuFlow Answer (Synthesized from {len(retrieved_chunks)} retrieved contexts):**\n"
        ]

        if key_points:
            lines.append("Key points matching your query:")
            for pt in key_points:
                lines.append(f"- {pt}")
            lines.append("")

        lines.append(f"**Top Relevant Passage** *(Cosine Similarity: {top_score:.4f})*:")
        lines.append(f"> {top_text}\n")

        lines.append("*Synthesized using DocuFlow Offline Engine.*")

        return {
            "answer": "\n".join(lines),
            "model": "DocuFlow-Offline-Synthesizer",
            "retrieved_count": len(retrieved_chunks),
            "sources": retrieved_chunks
        }


if __name__ == "__main__":
    client = LLMClient()
    print(f"Client model: {client.model}, base_url: {client.base_url}, active: {bool(client.client)}")
