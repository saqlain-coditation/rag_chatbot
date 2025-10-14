import json
import os
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks import CBEventType, EventPayload, LlamaDebugHandler


class RagDebugger(LlamaDebugHandler):
    def __init__(
        self,
        event_starts_to_ignore=None,
        event_ends_to_ignore=None,
        print_trace_on_end=True,
        logger=None,
    ):
        super().__init__(
            event_starts_to_ignore, event_ends_to_ignore, print_trace_on_end, logger
        )
        self.file = None

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        self._custom_logs(event_type, payload)
        return super().on_event_end(event_type, payload, event_id, **kwargs)

    # def on_event_start(
    #     self, event_type, payload=None, event_id="", parent_id="", **kwargs
    # ):
    #     self._custom_logs(event_type, payload)
    #     return super().on_event_start(
    #         event_type, payload, event_id, parent_id, **kwargs
    #     )

    def _custom_logs(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        if not payload:
            return

        self._log("\n" + str(event_type))
        if event_type == CBEventType.RETRIEVE:
            # Extract and print retrieved nodes and their content
            retrieved_nodes = payload.get(EventPayload.NODES)
            if not retrieved_nodes:
                return

            self._log("--- Retrieved Documents ---")
            for node in retrieved_nodes:
                self._log(f"Node ID: {node.id_}")
                self._log(f"Node Text: {node.text}...")
                self._log("=" * 80)

        elif event_type == CBEventType.SUB_QUESTION:
            qa_pair = payload.get(EventPayload.SUB_QUESTION)
            if not qa_pair:
                return

            question = qa_pair.sub_q.sub_question
            answer = qa_pair.answer

            self._log("=" * 80)
            self._log("Sub Question: " + question.strip())

            if answer:
                self._log("Answer: " + answer.strip())
                self._log("=" * 80)

        elif event_type == CBEventType.LLM:
            # Extract and print the final prompt and LLM response
            # Note: For multi-step synthesis, there might be multiple LLM calls.
            # This captures the one associated with the main query.
            calls = payload.get(EventPayload.MESSAGES)
            if not calls:
                return

            formatted_prompt = [chat.content for chat in calls]
            llm_response = payload.get(EventPayload.RESPONSE)

            self._log("=" * 80)
            self._log("LLM Call:\n" + "\n".join(formatted_prompt))

            if llm_response:
                self._log("LLM Response:\n" + llm_response.message.content)
                self._log("=" * 80)

        else:
            exclude = [CBEventType.EMBEDDING, CBEventType.SYNTHESIZE]
            if event_type not in exclude:
                self._log(json.dumps(payload, indent=4, default=lambda x: str(x)))

    def _log(self, data: str, location: Optional[str] = None):
        path = ["data", "logs", location, "log.log"]
        log_path = os.path.join(*[x for x in path if x is not None])
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file = self.file = self.file or open(log_path, "w")
        file.write(str(data) + "\n")
        self.file.flush()

    def __del__(self):
        if self.file:
            self.file.close()
