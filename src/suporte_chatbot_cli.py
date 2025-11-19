import os
import sys
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_factory import build_llm
from tools_agent import get_tool_names
from agents.support_agent import build_agent_with_memory


# =========================
# Modelos e Parser
# =========================
class SuporteResposta(BaseModel):
    tipo: str = Field(description="Tipo de solicitação: 'faq' ou 'troubleshooting'")
    resposta: str
    passos: List[str]
    prioridade: Optional[str] = Field(default=None, description="baixa|media|alta")
    tags: List[str]


def make_parser() -> Tuple[StructuredOutputParser, str]:
    schemas = [
        ResponseSchema(name="tipo", description="faq ou troubleshooting"),
        ResponseSchema(name="resposta", description="Resposta clara e objetiva"),
        ResponseSchema(name="passos", description="Lista de passos recomendados"),
        ResponseSchema(name="prioridade", description="baixa|media|alta"),
        ResponseSchema(name="tags", description="Lista de palavras-chave"),
    ]

    parser = StructuredOutputParser.from_response_schemas(schemas)
    return parser, parser.get_format_instructions()


# =========================
# Prompts e seleção
# =========================
def build_prompts() -> Tuple[PromptTemplate, PromptTemplate]:
    short_tpl = PromptTemplate.from_template(
        """
        Você é um assistente de suporte conciso. Responda diretamente e estruture a saída.

        Pergunta: {pergunta}

        {format_instructions}
        """
    )

    long_tpl = PromptTemplate.from_template(
        """
        Você é um assistente de suporte detalhado e empático. Identifique sintomas, causas prováveis e passos de resolução.

        Pergunta: {pergunta}
        Contexto: {contexto}

        {format_instructions}
        """
    )

    return short_tpl, long_tpl


def escolher_prompt(vars, short_tpl: PromptTemplate, long_tpl: PromptTemplate) -> str:
    pergunta = vars["pergunta"].lower()
    # Se pergunta longa ou contém palavras-chave de erro -> prompt detalhado
    if len(pergunta) > 180 or any(
        k in pergunta for k in ["erro", "falha", "não funciona", "exception", "traceback"]
    ):
        return long_tpl.format(
            pergunta=vars["pergunta"],
            contexto=vars["contexto"],
            format_instructions=vars["format_instructions"],
        )
    # Caso contrário -> prompt curto
    return short_tpl.format(
        pergunta=vars["pergunta"],
        contexto=vars.get("contexto", ""),
        format_instructions=vars["format_instructions"],
    )


def make_selector(short_tpl: PromptTemplate, long_tpl: PromptTemplate) -> RunnableLambda:
    return RunnableLambda(lambda v: escolher_prompt(v, short_tpl, long_tpl))


# =========================
# LLM
# =========================
# LLM é fornecido por llm_factory.build_llm()


# Ferramentas e agente foram extraídos para 'tools_agent.py'


# =========================
# Função principal de atendimento
# =========================
def atender_usuario(
    llm: ChatGoogleGenerativeAI,
    pergunta: str,
    contexto: str = "",
    parser: StructuredOutputParser = None,
    format_instructions: str = "",
    selector: RunnableLambda = None,
) -> SuporteResposta:
    if parser is None or not format_instructions:
        parser, format_instructions = make_parser()

    if selector is None:
        short_tpl, long_tpl = build_prompts()
        selector = make_selector(short_tpl, long_tpl)

    vars = {
        "pergunta": pergunta,
        "contexto": contexto,
        "format_instructions": format_instructions,
    }

    # Escolhe o prompt dinamicamente e já retorna string pronta
    final_prompt = selector.invoke(vars)

    # Chama o LLM
    resposta_raw = llm.invoke(final_prompt).content

    # Parser estruturado do LangChain
    parsed = parser.parse(resposta_raw)

    # Garantir que 'passos' e 'tags' sejam listas
    if isinstance(parsed.get("passos"), str):
        parsed["passos"] = [
            p.strip()
            for p in parsed["passos"].replace("1.", "\n1.").split("\n")
            if p.strip()
        ]

    if isinstance(parsed.get("tags"), str):
        parsed["tags"] = [t.strip() for t in parsed["tags"].split(",") if t.strip()]

    return SuporteResposta(**parsed)


# =========================
# CLI Interativo
# =========================
def print_banner():
    print("=" * 72)
    print("Assistente de Suporte — modo interativo (suporte | ferramentas)")
    print("Comandos: /sair, /contexto <txt>, /modo suporte, /modo ferramentas, /tools")
    print("=" * 72)


def should_use_tools(text: str) -> bool:
    t = text.lower()
    keywords = [
        "média",
        "media",
        "cria ticket",
        "ticket",
        "buscar",
        "pesquisa",
        "web",
        "search",
    ]
    return any(k in t for k in keywords)


def main():
    try:
        llm = build_llm()
    except Exception as e:
        print(f"Erro ao inicializar LLM: {e}")
        print("Verifique seu .env (GOOGLE_API_KEY, MODEL_NAME) e dependências.")
        sys.exit(1)

    parser, format_instructions = make_parser()
    short_tpl, long_tpl = build_prompts()
    selector = make_selector(short_tpl, long_tpl)
    agent_with_memory = build_agent_with_memory(llm)

    contexto_padrao = ""
    modo = "suporte"  # suporte | ferramentas
    print_banner()
    print("Assistente preparado. Informe sua pergunta.")

    while True:
        try:
            pergunta = input("\nPergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando. Até mais!")
            break

        if not pergunta:
            continue

        if pergunta.lower() in {"/sair", "sair", "exit", "quit"}:
            print("Encerrando. Até mais!")
            break

        if pergunta.lower().startswith("/contexto"):
            _, *rest = pergunta.split(" ", 1)
            contexto_padrao = rest[0] if rest else ""
            print(f"Contexto atualizado: '{contexto_padrao}'")
            continue

        if pergunta.lower() in {"/modo ferramentas", "/modo ferr", "/ferramentas"}:
            modo = "ferramentas"
            print("Modo alterado: ferramentas (tool_calling). Use /tools para listar.")
            continue

        if pergunta.lower() in {"/modo suporte", "/modo sup", "/suporte"}:
            modo = "suporte"
            print("Modo alterado: suporte estruturado.")
            continue

        if pergunta.lower() == "/tools":
            print("Ferramentas disponíveis:")
            try:
                print(", ".join(get_tool_names()))
            except Exception:
                print("calcula_media, cria_ticket, web_search")
            continue

        try:
            if modo == "suporte":
                # Roteamento automático para ferramentas quando a pergunta sugere uso
                if should_use_tools(pergunta):
                    res = agent_with_memory.invoke(
                        {"input": pergunta},
                        config={"configurable": {"session_id": "CLI"}},
                    )
                    print("\n— Ferramentas (com memória) —")
                    print(res.get("output") or res)
                    continue
                resposta = atender_usuario(
                    llm=llm,
                    pergunta=pergunta,
                    contexto=contexto_padrao,
                    parser=parser,
                    format_instructions=format_instructions,
                    selector=selector,
                )

                print("\n— Resultado —")
                print(f"Tipo: {resposta.tipo}")
                print(f"Prioridade: {resposta.prioridade}")
                print(f"Tags: {', '.join(resposta.tags)}")
                print("Resposta:")
                print(resposta.resposta)
                print("Passos:")
                for idx, passo in enumerate(resposta.passos, start=1):
                    print(f"  {idx}. {passo}")
            else:
                res = agent_with_memory.invoke(
                    {"input": pergunta},
                    config={"configurable": {"session_id": "CLI"}},
                )
                print("\n— Ferramentas (com memória) —")
                print(res.get("output") or res)

        except Exception as e:
            print(f"Erro: {e}")


if __name__ == "__main__":
    main()