import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from llm import llm
from tools import book_hotel
from utils import parse_messages, save_graph_visualization

# 定义并运行agent
async def run_agent():
    tools = [book_hotel]
    # 生产环境中需要使用数据库落盘（持久化）
    checkpointer = InMemorySaver()

    # 定义系统消息
    system_message = SystemMessage(content=(
        "你是一个AI订票助手。"
    ))

    # 创建ReAct风格的agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
        middleware=[
        HumanInTheLoopMiddleware( 
            interrupt_on={
                "book_hotel": True,  # All decisions (approve, edit, reject) allowed
            },
            description_prefix="Tool execution pending approval",
        ),
    ], 
        checkpointer=checkpointer # 开启记忆能力
    )

    # 将定义的agent的graph进行可视化输出保存至本地
    save_graph_visualization(agent)

    # 定义short-term需使用的 thread_id
    config = {"configurable": {"thread_id": "1"}}

    # 1、非流式处理查询
    agent_response = await agent.ainvoke({"messages": [HumanMessage(content="预定一个汉庭酒店")]}, config=config) # type: ignore
    # 将返回的messages进行格式化输出
    parse_messages(agent_response['messages'])
    agent_response_content = agent_response["messages"][-1].content
    # The interrupt contains the full HITL request with action_requests and review_configs
    print(agent_response['__interrupt__'])

    # (1)模拟人类反馈：测试3种反馈方式
  # Resume with approval decision
    new_respose = agent.invoke(
           Command( 
            resume={"decisions": [{"type": "reject"}]}  # "approve" or "reject"
            # resume={
            # "decisions": [
            #     {
            #         "type": "edit",
            #         # Edited action with tool name and args
            #         "edited_action": {
            #             # Will usually be the same as the original action.
            #             "name": "book_hotel",
            #             # Arguments to pass to the tool.
            #             "args": {"hotel_name": "七天酒店"},
            #         }
            #     }
            #     ]
            # }  
        ), 
        config=config # Same thread ID to resume the paused conversation # type: ignore
    )
    # 将返回的messages进行格式化输出
    parse_messages(new_respose['messages'])
    agent_response_content = new_respose["messages"][-1].content
    print(f"agent_response:{agent_response_content}")
if __name__ == "__main__":
    asyncio.run(run_agent())    