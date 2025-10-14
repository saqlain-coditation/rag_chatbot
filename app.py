import nest_asyncio

nest_asyncio.apply()

# from ui.chainlit.ui import main
from rag.agent.run import main

main()
