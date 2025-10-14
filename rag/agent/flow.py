from typing import TypeVar

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from .events import *

T = TypeVar("T")


class AgenticWorkflow(Workflow):

    def __init__(
        self,
        llm: LLM,
        engine: BaseQueryEngine,
        timeout=45,
        disable_validation=False,
        verbose=False,
        resource_manager=None,
        num_concurrent_runs=None,
    ):
        super().__init__(
            timeout, disable_validation, verbose, resource_manager, num_concurrent_runs
        )
        self.llm = llm
        self.engine = engine
        self._memory = ChatMemoryBuffer.from_defaults(llm=llm)

    async def _get(self, ctx: Context, key: str, default: Optional[T] = None) -> T:
        query_data = await ctx.store.get("query_data")
        return query_data.get(key, default)

    async def _set(self, ctx: Context, key: str, data):
        query_data = await ctx.store.get("query_data")
        query_data[key] = data

    @step
    async def initialize(self, ctx: Context, ev: StartEvent) -> JudgeQueryEvent:
        llm = await ctx.store.get("llm", default=None)
        if llm is None:
            await ctx.store.set("llm", self.llm)
            await ctx.store.set("engine", self.engine)

            # we use a chat engine so it remembers previous interactions
            await ctx.store.set(
                "llm_judge", SimpleChatEngine.from_defaults(llm=self.llm)
            )
            await ctx.store.set(
                "llm_improve", SimpleChatEngine.from_defaults(llm=self.llm)
            )
            await ctx.store.set(
                "llm_answer",
                SimpleChatEngine.from_defaults(llm=self.llm, memory=self._memory),
            )

        query_data = {
            "original_query": ev.query,
            "query": ev.query,
            "responses": {},
        }
        await ctx.store.set("query_data", query_data)
        return JudgeQueryEvent(query=ev.query)

    @step
    async def judge_query(
        self,
        ctx: Context,
        ev: JudgeQueryEvent,
    ) -> ImproveQueryEvent | AttachContextEvent:
        current_query = await self._get(ctx, "query")
        judge_query = await self._get(ctx, "judge_query")
        judge_count = await self._get(ctx, "judge_count", 0)
        if judge_query != current_query and judge_count < 1:
            await self._set(ctx, "judge_query", current_query)
            await self._set(ctx, "judge_count", judge_count + 1)
            llm = await ctx.store.get("llm_judge")
            raw_response = await llm.achat(
                f"""
                Given a user query, determine if this is likely to yield good results from a RAG system as-is. If it's good, return 'good', if it's bad, return 'bad'.
                Good queries use a lot of relevant keywords and are detailed. Bad queries are vague or ambiguous.

                Here is the query: {ev.query}

                Don't provide any additional input, metadata, markdown or preamble.
                """
            )
            response = str(raw_response).strip()
        else:
            response = "repeated"

        print(f"\nJudge Query: {response}")
        if response == "bad":
            return ImproveQueryEvent(query=ev.query)
        else:
            return AttachContextEvent(query=ev.query)

    @step
    async def improve_query(
        self,
        ctx: Context,
        ev: ImproveQueryEvent,
    ) -> JudgeQueryEvent:
        llm = await ctx.store.get("llm_improve")
        raw_response = await llm.achat(
            f"""
            This is a query to a RAG system: {ev.query}

            The query is bad because it is too vague. Please provide a more detailed query that includes specific keywords and removes any ambiguity. Don't add any new meaning which the query does not already contain. Only expand the original meaning of the query. If the words are not meaningful, don't try to interpret or correct the words.
            
            Return only the final query and nothing else.
            Don't provide any additional input, metadata, markdown or preamble.
            """
        )
        response = str(raw_response).strip()
        print(f"\nImprove Query: {response}")
        return JudgeQueryEvent(query=response)

    @step
    async def generate_context(
        self, ctx: Context, ev: AttachContextEvent
    ) -> SearchEvent:
        history = await self._memory.aget_all()
        if not history:
            return SearchEvent(query=ev.query)

        llm = await ctx.store.get("llm_answer")
        response = await llm.achat(
            f"""
            Given a user's query and the conversation (between Human and Assistant), 
            create context based on relevant parts of conversation history.
            This context will be used to answer the user's query.

            Add all the data, contextual information and clues to the context.
            Don't create more questions and adapt everything into just contextual information.
            Don't try to answer the query either.
            ----------------------------------
            {messages_to_history_str(history)}
            ----------------------------------

            User's Question: {ev.query}

            Return only the created context and nothing else.
            """
        )

        query = f"{response}\n\n{ev.query}"
        print(f"\nAdd Context: {query}")
        return SearchEvent(query=query)

    @step
    async def search(self, ctx: Context, ev: SearchEvent) -> JudgeResponseEvent:
        engine = await ctx.store.get("engine")
        response = str(await engine.aquery(ev.query)).strip()
        await self._memory.aput(ChatMessage(role=MessageRole.USER, content=ev.query))
        await self._memory.aput(
            ChatMessage(role=MessageRole.ASSISTANT, content=response)
        )
        responses = await self._get(ctx, "responses")
        responses[ev.query] = response
        print(f"\nSearch: {response}")
        return JudgeResponseEvent(query=ev.query, response=response)

    @step
    async def judge(
        self,
        ctx: Context,
        ev: JudgeResponseEvent,
    ) -> ReQueryEvent | AnswerEvent:
        original_query = await self._get(ctx, "original_query")
        responses = await self._get(ctx, "responses")
        llm = await ctx.store.get("llm_judge")
        response = await llm.achat(
            f"""
            A user had provided a query and a response was generated. 
            The response was generated from verified sources.
            Assume the response to be factual & true, without any shred of fabrication.
            Your job is to judge the response relative to the query.

            User query: {original_query}.

            The process was:
            {"\n".join([a for a in responses.values()])}

            The final response was: {ev.response}.

            Take the user's query at face value.
            Does the response clearly answer the user's question ?
            Are there any parts of the *user's query* which are unanswered by the response ?
            If all parts of the user's query can be said to be answered by the response.
            Then it is correct, return 'correct'.

            If the response is judged as not correct only then:
            Does the response provokes more potential questions ?
            Can the question can be expanded using the answer to get more details ?
            If the response has any unanswered segments which should also be answered.
            return 'retry'.

            If the response provides a lot of context/information but still fails to answer the query,
            then it is also possible that the query cannot be answered and retrying is useless.
            If the response cannot answer the user's query,
            even after further revisions, 
            potential expansions of the response,
            return 'failure'.

            Review the response in order, first check if it is correct, and so on.
            Move on to the next step only if the previous step is false.
            Also explain the reason behind your choice, step by step in order.
            
            In the first line your choice, and the reason from the second line:
            For example: 

            ```
            correct
            correct: <reason why it is correct>
            ```

            ```
            retry
            not correct: <reason why it is not correct>
            retry: <reason why retry will work>
            ```

            ```
            failure
            not correct: <reason why it is not correct>
            not retry: <reason why retry will not work>
            failure: <reason why it is considered a failure>
            ```

            Don't provide any additional input, metadata, markdown or preamble.
            """
        )

        response = str(response).strip()
        result = response.splitlines()[0].strip()
        reason = response.split("\n", 1)[-1].strip()
        print(f"\nJudge Response: {response}")
        if result == "retry":
            return ReQueryEvent(query=ev.query, response=ev.response)
        return AnswerEvent(query=ev.query, response=ev.response)

    @step
    async def re_query(self, ctx: Context, ev: ReQueryEvent) -> JudgeQueryEvent:
        original_query = await self._get(ctx, "original_query")
        llm = await ctx.store.get("llm_improve")
        new_query = await llm.achat(
            f"""
            A user had provided a query and an answer was generated. 
            The answer was judged as incorrect. Your job is to create a new query or set of queries, which can help answer the original query.

            The original query was: {original_query}.
            The current query was: {ev.query}.
            The current answer was: {ev.response}.

            Please use the previous queries and see which parts of it are still unanswered. Generate new query to completely answer the original query.

            Return only the new query and nothing else.
            Don't provide any additional input, metadata, markdown or preamble.
            """
        )

        response = await llm.achat(
            f"""
            A user had provided a query and an answer was generated. 
            The answer was judged as incorrect, and the query was revised.
            Your job is to combine all the previous data with the new query.

            The original query was: {original_query}.
            The previous query was: {ev.query}.
            The previous answer was: {ev.response}.
            
            The new query is: {new_query}

            Convert all the previous data to contextual information/clues, and rewrite the new query incorporating all this information in it. Don't change the meaning of the new query, and dont add any new questions either. Keep the meaning of the new query, but add more context.

            Return only the final revised new query and nothing else.
            Don't provide any additional input, metadata, markdown or preamble.
            """
        )
        response = str(response).strip()
        await self._set(ctx, "query", response)
        print(f"\nNew Query: {response}")
        return JudgeQueryEvent(query=response)

    @step
    async def answer(self, ctx: Context, ev: AnswerEvent) -> StopEvent:
        original_query = await self._get(ctx, "original_query")
        llm = await ctx.store.get("llm_answer")
        response = await llm.achat(
            f"""
            Context information is below.
            ---------------------
            {ev.query}
            {ev.response}

            Given the context information and not prior knowledge, answer the query:
            Query: {original_query}
            """
        )

        return StopEvent(response)
