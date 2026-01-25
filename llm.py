import os
from langchain.chat_models import init_chat_model

# 使用langgraph推荐方式定义大模型
llm = init_chat_model(
    model="gpt-5-chat-latest",
    temperature=0,
    base_url="https://api.laozhang.ai/v1",
    api_key=os.getenv("LAOZHANG_API_KEY")
)