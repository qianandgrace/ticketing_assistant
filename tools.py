from langchain_core.tools import tool


@tool("book_hotel",description="需要人工审查/批准的预定酒店的工具")
def book_hotel(hotel_name: str):
    # 实际业务场景：处理酒店预定的业务逻辑
    return f"成功预定了在{hotel_name}的住宿。"