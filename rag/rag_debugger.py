from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks import LlamaDebugHandler
from typing import Any, Dict, List, Optional


class RagDebugger(LlamaDebugHandler):
    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        self._custom_logs(event_type, payload)
        return super().on_event_end(event_type, payload, event_id, **kwargs)

    def _custom_logs(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        if not payload:
            return

        if event_type == CBEventType.RETRIEVE:
            # Extract and print retrieved nodes and their content
            retrieved_nodes = payload.get(EventPayload.NODES)
            if not retrieved_nodes:
                return

            print("--- Retrieved Documents ---")
            for node in retrieved_nodes:
                print(f"Node ID: {node.id_}")
                print(f"Node Text: {node.text}...")
                print("=" * 80)

        elif event_type == CBEventType.SUB_QUESTION:
            qa_pair = payload.get(EventPayload.SUB_QUESTION)
            if not qa_pair:
                return

            print("=" * 80)
            print("Sub Question: " + qa_pair.sub_q.sub_question.strip())
            print("Answer: " + qa_pair.answer.strip())
            print("=" * 80)

        elif event_type == CBEventType.LLM:
            # Extract and print the final prompt and LLM response
            # Note: For multi-step synthesis, there might be multiple LLM calls.
            # This captures the one associated with the main query.
            calls = payload.get(EventPayload.MESSAGES)
            if not calls:
                return

            formatted_prompt = [chat.content for chat in calls]
            llm_response = payload.get(EventPayload.RESPONSE)

            print("=" * 80)
            print("LLM Call:\n" + "\n".join(formatted_prompt))
            print("LLM Response:\n" + llm_response.message.content)
            print("=" * 80)
