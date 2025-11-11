import os
import sys
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI


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
def build_llm() -> ChatGoogleGenerativeAI:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrado. Defina no .env ou ambiente."
        )

    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


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
    print("Assistente de Suporte — modo interativo")
    print("Comandos: /sair para encerrar, /contexto para definir contexto")
    print("=" * 72)


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

    contexto_padrao = ""
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

        try:
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

        except Exception as e:
            print(f"Erro ao atender: {e}")


if __name__ == "__main__":
    main()