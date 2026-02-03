import logging
import asyncio
import sys
from typing import TypedDict
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from llm import get_llm
from utils import parse_messages, save_graph_visualization, pre_model_hook, add_human_in_the_loop
from tools import get_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("ticket_assistant")
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class UserConfig(TypedDict):
        user_id: str

llm = get_llm("openai")
DB_URI= "postgresql://gq210:123456@localhost:5432/postgres?options=-csearch_path%3Dticket_assistant_memory"

async def non_streamble_ivoke(agent, config: RunnableConfig, messages: str, debug: bool = False):
    """
    Docstring for non_streamble_ivoke
    """
    agent_response = await agent.ainvoke({"messages": [HumanMessage(content=messages)]}, config)
    # # 将返回的messages进行格式化输出
    if debug:
        parse_messages(agent_response['messages'])
    agent_response_content = agent_response["messages"][-1].content
    logger.info(f"initial_agent_response:{agent_response_content}")
    # (1)模拟人类反馈：测试3种反馈方式
    agent_response = await agent.ainvoke(
        Command(resume=[{"type": "accept"}]),
        # Command(resume=[{"type": "edit", "args": {"args": {'train_number': 'G805'}}}]),
        # Command(resume=[{"type": "reject", "args": "我不想查询了"}]),
        config
    )
    # 将返回的messages进行格式化输出
    if debug:
        parse_messages(agent_response['messages'])
    agent_response_content = agent_response["messages"][-1].content
    logger.info(f"final_agent_response:{agent_response_content}")


async def stream_ivoke(agent, config: RunnableConfig, message: str, debug: bool = False):
    """
    Docstring for stream_ivoke
    
    :param agent: Description
    :param config: Description
    :type config: RunnableConfig
    :param message: Description
    :type message: str
    :param debug: Description
    :type debug: bool
    """ 
    async for message_chunk, metadata in agent.astream(
            input={"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages"
    ):
        # 测试原始输出
        if debug:
            logger.info(f"Message Chunk: {message_chunk}")
            logger.info(f"Metadata: {metadata}")    
    
        # 跳过工具输出
        if metadata["langgraph_node"]=="tools":
            continue
    
        # 输出最终结果
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
    
    # 模拟人类反馈：测试3种反馈方式
    async for message_chunk, metadata in agent.astream(
        Command(resume=[{"type": "accept"}]),
        # Command(resume=[{"type": "edit", "args": {"args": {'location': '120.619585,31.299379'}}}]),
        # Command(resume=[{"type": "response", "args": "我不想查询了"}]),
        config,
        stream_mode="messages"
    ):
        # 测试原始输出
        if debug:
            logger.info(f"Message Chunk: {message_chunk}")
            logger.info(f"Metadata: {metadata}")
    
        # 跳过工具输出
        if metadata["langgraph_node"]=="tools":
            continue
        # 输出最终结果
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    
async def load_long_term_memory(store, user_id: str) -> str:
    namespace = ("memories", user_id)
    memories = await store.asearch(namespace, query="")

    if not memories:
        logger.info("📦 长期记忆：无")
        return "无长期记忆信息"

    info = " ".join([m.value["data"] for m in memories])
    logger.info("📦 长期记忆检索结果: %s", info)
    return info

# 定义并运行agent
async def run_agent(save_node=False, store_memory: bool = False):
    # 从MCP Server中获取可提供使用的全部工具
    # MCP Client 能够动态感知工具的变化
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        all_tools = await get_tools()
        # 12306工具使用这种方式存在bug，暂时不清楚原因
        # tools = [await add_human_in_the_loop(all_tools[6])]
        add_human_tools = [await add_human_in_the_loop(index) for index in all_tools[8:]] # type: ignore
        tools = all_tools[:8] + add_human_tools

        # 定义系统消息
        system_message = SystemMessage(content=(
            "你是一个AI助手。"
        ))
        # 创建ReAct风格的agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
            checkpointer=InMemorySaver(), 
            # 这样写会报错
            # checkpointer=checkpointer,
            store=store,
            pre_model_hook=pre_model_hook
        )

        # 将定义的agent的graph进行可视化输出保存至本地
        if save_node:
            save_graph_visualization(agent)
    
        # 定义用户配置和线程ID
        user_config = UserConfig(user_id = "1")
        config: RunnableConfig = {
            "configurable": {
            "thread_id": "5",
            **user_config
            }
        }
        if store_memory:
            # 自定义存储逻辑 对用户输入进行处理，检查是否需要存储长期记忆
            namespace = ("memories", config["configurable"]["user_id"])
            memory1 = "我的名字叫gq"
            await store.aput(namespace, str(uuid.uuid4()), {"data": memory1})
            memory2 = "我的订票偏好是只定价格最低的车次"
            await store.aput(namespace, str(uuid.uuid4()), {"data": memory2})
        info = await load_long_term_memory(store, user_config["user_id"])
        logger.info(f"长期记忆信息加载完成: {info}")

        # # 1、非流式处理查询
        """
        案例：预定1631次列车， 调用工具查询下上海的天气 调用工具查询北京到上海的高铁班次
        """
        message = "预定明天北京到上海的高铁班次"
        message += info
        await non_streamble_ivoke(agent, config, message, debug=True)
        return


        # 2、流式处理查询
        # await stream_ivoke(agent, config, "查询上海的天气", debug=False)
        # return


if __name__ == "__main__":
    asyncio.run(run_agent())


