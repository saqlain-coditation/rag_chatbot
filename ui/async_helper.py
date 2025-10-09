import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
# A dictionary to store ongoing tasks
tasks = {}

# Ensure there is a running event loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Define a helper function to create and schedule async tasks
def schedule_task(coro):
    """Schedules an async task and stores it with a unique key."""
    key = len(tasks)
    tasks[key] = loop.create_task(coro)
    return key


# Run the event loop to process scheduled tasks
def process_tasks():
    """Process pending tasks on the event loop."""
    pending = [task for task in tasks.values() if not task.done()]
    if pending:
        loop.run_until_complete(asyncio.gather(*pending))


def run_async(async_callback: Callable[[], Awaitable[T]]) -> T:
    key = schedule_task(async_callback())
    process_tasks()
    return tasks[key].result()
