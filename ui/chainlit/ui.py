import asyncio
import uuid

import chainlit as cl

from . import utils

current_index = None
engine = None
context = None


async def actions():
    uid = str(uuid.uuid4())
    name = f"index_action"
    actions = [
        cl.Action(label="🆕 Create Blank Index", name=name, payload={"type": "new"}),
        cl.Action(label="📥 Import Index", name=name, payload={"type": "import"}),
        cl.Action(label="📄 Upload Docs", name=name, payload={"type": "upload"}),
        cl.Action(label="💾 Export Index", name=name, payload={"type": "export"}),
    ]
    await cl.Message(
        content="⚙️ Select next action:", tags=[uid], actions=actions
    ).send()


@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to your RAG App 👋").send()
    await actions()


@cl.action_callback("index_action")
async def handle_action(action):
    global current_index, engine, context
    act = action.payload["type"]
    if act == "new":
        current_index = utils.create_index()
        await cl.Message(content="✅ Blank index created!").send()
    elif act == "import":
        files = await cl.AskFileMessage(
            content="Upload your index JSON files",
            accept=["application/json"],
            max_size_mb=50,
            max_files=5,
        ).send()

        if not files:
            return
        current_index = await asyncio.to_thread(utils.import_index, files)
        await cl.Message(content="📦 Index imported successfully!").send()
    elif act == "upload":
        if current_index is None:
            await cl.Message(
                content="⚠️ No active index. Create or import one first!"
            ).send()
            return
        files = await cl.AskFileMessage(
            content="Upload text or document files",
            accept=["text/plain"],
            max_size_mb=10,
            max_files=5,
        ).send()

        if not files:
            return
        utils.load_document(current_index, files)
        await cl.Message(content="📚 Documents added to index!").send()
    elif act == "export":
        if current_index is None:
            await cl.Message(content="⚠️ No index to export!").send()
            return
        utils.export_index(current_index)
        await cl.Message(content=f"💾 Index exported to `./.export`").send()
    await actions()


@cl.on_message
async def main(msg: cl.Message):
    global current_index, engine, context
    if current_index is None:
        await cl.Message(content="⚠️ Please create or import an index first!").send()
        return

    if not engine:
        engine = utils.create_engine(current_index)
        context = utils.Context(engine)
    response = await engine.run(query=msg.content, ctx=context)
    await cl.Message(content=str(response)).send()
