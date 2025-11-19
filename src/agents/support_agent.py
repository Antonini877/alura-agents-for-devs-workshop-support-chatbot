from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools_agent import build_tools
from memory.sql_memory import wrap_with_sql_history


def build_tools_agent(llm) -> AgentExecutor:
    """Cria um agente de ferramentas com prompt preparado para memória (chat_history)."""
    tools = build_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente de suporte. Use ferramentas quando necessário e mantenha contexto da conversa.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def build_agent_with_memory(llm):
    """Retorna um runnable do agente com memória SQLite por sessão."""
    executor = build_tools_agent(llm)
    return wrap_with_sql_history(executor)