from typing import List

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llms.llm import LLM

from .. import types as t


class ParallelRagSearchEngine(t.BaseRagSearchEngine):
    def __init__(self, llm: LLM, engines: List[t.BaseRagSearchEngine]):
        super().__init__()
        self.llm = llm
        self.engines = engines

    def search(self, query: str) -> RESPONSE_TYPE:
        responses = [e.search(query) for e in self.engines]
        return self.select(query, responses)

    async def asearch(self, query: str) -> RESPONSE_TYPE:
        responses = [await e.asearch(query) for e in self.engines]
        return await self.aselect(query, responses)

    def select(self, query: str, responses: List[RESPONSE_TYPE]) -> RESPONSE_TYPE:
        response = self.llm.complete(self._select_prompt(query, responses))
        return self._pick_answer(responses, response)

    async def aselect(
        self, query: str, responses: List[RESPONSE_TYPE]
    ) -> RESPONSE_TYPE:
        response = await self.llm.acomplete(self._select_prompt(query, responses))
        return self._pick_answer(responses, response)

    def _select_prompt(self, query: str, responses: List[RESPONSE_TYPE]) -> str:
        prompt = f"""
            A user has provided a query and {len(responses)} different strategies have been used
            to try to answer the query. Your job is to decide which strategy best
            answered the query. The query was: {query}

            {"\n".join([f"Response {i}: {response}" for i, response in enumerate(responses)])}

            Please provide the number of the best response.
            Just provide the number, with no other text or preamble.
            """
        return prompt

    def _pick_answer(
        self, responses: List[RESPONSE_TYPE], choice: CompletionResponse
    ) -> RESPONSE_TYPE:
        try:
            best_response = int(str(choice))
            answer = responses[best_response]
        except:
            answer = responses[0]

        return answer
