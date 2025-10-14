import asyncio

import nest_asyncio
import typer

nest_asyncio.apply()

import lib.agent.run as agentic_rag

app = typer.Typer(help="A RAG-based conversational agent CLI.")


async def async_chat_loop():
    """
    Handles the asynchronous conversation loop with the RAG agent.
    """
    # 1. Setup the Agent and Context
    agent = agentic_rag.build_agent(
        agentic_rag.build_index(
            input_dir="input",
            index_dir=".index",
        )
    )
    ctx = agentic_rag.Context(agent)

    typer.echo("Agent initialized. Type your question or 'x' to exit.")
    while True:
        try:
            choice = typer.prompt("\nQuestion")  # Use Typer's prompt for better UX
        except typer.Abort:
            # Handles Ctrl+C (KeyboardInterrupt) gracefully
            typer.echo("\nExiting agent.")
            return

        if choice.lower() == "x":
            typer.echo("Exiting agent.")
            return

        typer.echo("Thinking...", nl=False)

        # 2. Run the asynchronous agent
        try:
            result = await agent.run(query=choice, ctx=ctx)
            # Use typer.echo for consistent output
            typer.echo("\rAnswer: " + str(result) + "\n")  # \r overwrites "Thinking..."
        except Exception as e:
            typer.echo(f"\rAn error occurred: {e}. Please try again.")


# --- Synchronous Entry Point for Typer ---


# The @app.callback decorator is the standard Typer entry point.
# It uses 'typer.run' internally to handle the async call.
@app.command()
def run():
    """
    Main entry point for the RAG Agent CLI.
    """
    asyncio.run(async_chat_loop())


main = app
