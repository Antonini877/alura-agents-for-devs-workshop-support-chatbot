from typing import List, Optional

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def safety_guard(text: str) -> None:
    """Bloqueia padrões potencialmente perigosos em entradas de ferramentas."""
    bloqueados = ["rm -rf", "drop table", "shutdown", "format c:\\"]
    low = str(text).lower()
    if any(b in low for b in bloqueados):
        raise ValueError("Conteúdo potencialmente perigoso bloqueado.")


class CalculaMediaArgs(BaseModel):
    numeros: List[float] = Field(..., description="Lista de números para média")


def calcula_media(numeros: List[float]) -> str:
    safety_guard(numeros)
    if not numeros:
        return "Lista vazia; informe ao menos um número."
    media = sum(numeros) / len(numeros)
    return f"Média: {media:.4f}"


class CriaTicketArgs(BaseModel):
    titulo: str = Field(..., description="Título do ticket")
    prioridade: Optional[str] = Field(
        default="media", description="Prioridade: baixa|media|alta"
    )


def cria_ticket(titulo: str, prioridade: str = "media") -> str:
    safety_guard(titulo)
    pr = str(prioridade).lower()
    if pr not in {"baixa", "media", "alta"}:
        pr = "media"
    if len(titulo) > 140:
        titulo = titulo[:140]
    return f"Ticket criado: TCK-{abs(hash(titulo)) % 10000} ({pr}) — '{titulo}'"


def build_tools() -> List:
    """Constroi a lista de ferramentas disponíveis."""
    calcula_media_tool = StructuredTool.from_function(
        name="calcula_media",
        description="Calcula a média aritmética com checagem de segurança.",
        func=calcula_media,
        args_schema=CalculaMediaArgs,
    )

    cria_ticket_tool = StructuredTool.from_function(
        name="cria_ticket",
        description="Cria um ticket fictício com prioridade válida e título sanitizado.",
        func=cria_ticket,
        args_schema=CriaTicketArgs,
    )

    search_tool = DuckDuckGoSearchRun(name="web_search")

    return [calcula_media_tool, cria_ticket_tool, search_tool]


def get_tool_names() -> List[str]:
    """Retorna nomes das ferramentas disponíveis."""
    return [t.name for t in build_tools()]


def build_tools_agent(llm) -> AgentExecutor:
    """Cria o agente de tool_calling pronto para uso com as ferramentas definidas."""
    tools = build_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente de suporte. Use ferramentas quando necessário e siga fielmente as descrições das ferramentas.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)