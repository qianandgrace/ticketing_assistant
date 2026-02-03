import asyncio
import sys
import uuid
import logging

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from llm import get_llm
from tools import get_tools


## 普通日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("ticket_assistant")

class StreamHandlerNoNewline(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
            self.stream.flush()
        except Exception:
            self.handleError(record)


# ===== AI Token 专用 logger =====
stream_logger = logging.getLogger("ai_stream")
stream_logger.setLevel(logging.INFO)
stream_logger.propagate = False  # ❗ 防止被 root logger 再打一次

stream_handler = StreamHandlerNoNewline(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(message)s"))
stream_handler.terminator = ""

stream_logger.handlers.clear()
stream_logger.addHandler(stream_handler)


# Add this line for Windows compatibility
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


DB_URI= "postgresql://gq210:123456@localhost:5432/postgres?options=-csearch_path%3Dticket_assistant_memory"
LLM = get_llm("openai")

async def build_agent(checkpointer, store) :
    tools = await get_tools()

    system_message = SystemMessage(
        content=(
            "你是一个AI订票助手。"
        )
    )

    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt=system_message,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "book_railway": True,
                    "get_tickets": {"allowed_decisions": ["approve", "reject"]},
                },
                description_prefix="Tool execution pending approval",
            )
        ],
        checkpointer=InMemorySaver(),  # ✅ 短期记忆
        store=store,                # ✅ 长期记忆
    )

    return agent

async def load_long_term_memory(store, user_id: str) -> str:
    namespace = ("memories", user_id)
    memories = await store.asearch(namespace, query="")

    if not memories:
        logger.info("📦 长期记忆：无")
        return "无长期记忆信息"

    info = " ".join([m.value["data"] for m in memories])
    logger.info("📦 长期记忆检索结果: %s", info)
    return info

async def run_with_stream_hitl(
    agent,
    user_input: str,
    config: dict,
):
    logger.info("========== 🤖 Agent Start ==========")
    logger.info("User Input: %s", user_input)

    interrupted = False

    # -------- 第一轮：模型 →（可能）HITL --------
    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, _ = chunk

            # 模型 token
            if msg.content:
                stream_logger.info(msg.content)

            # 模型决定调用工具
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                logger.debug("🛠️ 模型决定调用工具")
                for tc in msg.tool_calls:
                    logger.debug(
                        "Tool Call | name=%s | id=%s | args=%s",
                        tc["name"],
                        tc["id"],
                        tc["args"],
                    )

        elif mode == "updates":
            # HITL 中断
            if "__interrupt__" in chunk:
                interrupted = True
                logger.warning("🚨 HITL 中断触发")
                logger.warning("Interrupt Payload: %s", chunk["__interrupt__"])
                # break  # ❗ 非常关键：不能在这里 break，否则无法继续后续流程
    stream_logger.info("\n")

    # -------- 第二轮：人类决策 → resume --------
    if interrupted:
        decision = {"decisions": [{"type": "approve"}]}  # 模拟人类决策
        logger.warning("🧑‍⚖️ 人类决策: %s", decision)

        async for mode, chunk in agent.astream(
            Command(resume=decision),
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg, _ = chunk

                # 模型最终输出
                if msg.content:
                    stream_logger.info(msg.content)

                # tool 执行结果
                if msg.type == "tool":
                    logger.debug(
                        f"🔧 Tool 执行完成 | tool_call_id={msg.tool_call_id} | result={msg.content}",
                    )
    stream_logger.info("\n")
    logger.info("========== ✅ Agent End ==========")

async def run_agent():
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        agent = await build_agent(checkpointer, store)

        config = {
            "configurable": {
            "thread_id": "5",
            "user_id": "1",
            }
        }
        logger.info("初始化 Agent 完成 | thread_id=%s | user_id=%s",
                    config["configurable"]["thread_id"],
                    config["configurable"]["user_id"])

        info = await load_long_term_memory(store, user_id="1")

        user_input = f"预定明天的北京到上海的火车票，我的附加信息有：{info}"
        logger.info("构造用户输入完成")

        await run_with_stream_hitl(
            agent=agent,
            user_input=user_input,
            config=config,
        )

"""
# 自定义存储逻辑 对用户输入进行处理，检查是否需要存储长期记忆
        # namespace = ("memories", config["configurable"]["user_id"])
        # memory1 = "我的名字叫gq"
        # await store.aput(namespace, str(uuid.uuid4()), {"data": memory1})
        # memory2 = "我的订票偏好是只定价格最低的车次"
        # await store.aput(namespace, str(uuid.uuid4()), {"data": memory2})
        # print("已存储长期记忆！")
"""


if __name__ == "__main__":
    asyncio.run(run_agent())    