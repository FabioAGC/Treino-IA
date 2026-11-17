"""Demo do RAG jurídico para Hugging Face Spaces (CPU gratuita).

Pipeline igual ao melhor sistema dos notebooks: BM25 + BGE-M3 (RRF) → reranker → LLM com citações.
- O índice (chunks.parquet + emb_bge_m3.npy) vem da saída do notebook 02: copie para app/data/.
- A geração usa a API de inferência do Hugging Face (defina o secret HF_TOKEN no Space).
  Sem token, o app funciona em "modo busca": mostra os artigos encontrados, sem resposta gerada.
"""
import os
import re
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA = Path(__file__).parent / "data"
EMB_ID = os.environ.get("EMB_MODEL", "BAAI/bge-m3")
RERANKER_ID = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_ID = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
USAR_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"
CANDIDATOS = int(os.environ.get("CANDIDATES", "15"))  # na CPU, menos candidatos = reranker mais rápido
TOP_CHUNKS = 5
RRF_K = 60
FRASE_RECUSA = "Não encontrei essa informação nos documentos consultados."

SYSTEM = f"""Você é um assistente que explica a legislação brasileira para cidadãos comuns.
Regras:
1. Responda SOMENTE com base nos trechos de lei fornecidos. Não use conhecimento externo.
2. Cite os artigos usados no formato [LEI, Art. N], por exemplo [CDC, Art. 49].
3. Se os trechos não responderem à pergunta, responda exatamente: "{FRASE_RECUSA}"
4. Seja claro e direto, em no máximo 5 frases. Isto não é aconselhamento jurídico."""

# ---------------------------------------------------------------- índices
chunks = pd.read_parquet(DATA / "chunks.parquet")
emb = np.load(DATA / "emb_bge_m3.npy").astype(np.float32)
assert len(chunks) == len(emb), "chunks.parquet e emb_bge_m3.npy precisam vir da mesma execução do notebook 02"

STOP = set(
    "a o e é de da do das dos em no na nos nas um uma uns umas para por pelo pela com sem que se os as ao aos "
    "ou mas como mais seu sua seus suas ele ela eles elas isso isto este esta esse essa meu minha eu você".split()
)


def sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


try:  # mesmo pré-processamento do notebook 02, quando o NLTK está disponível
    import nltk

    nltk.download("stopwords", quiet=True)
    nltk.download("rslp", quiet=True)
    from nltk.corpus import stopwords

    STOP = set(stopwords.words("portuguese"))
    stem = nltk.stem.RSLPStemmer().stem
except Exception:
    stem = lambda w: w  # noqa: E731


def tokenizar(t: str) -> list[str]:
    return [sem_acento(stem(w)) for w in re.findall(r"\w+", t.lower()) if w not in STOP and len(w) > 1]


bm25 = BM25Okapi([tokenizar(t) for t in chunks.texto_indexado])
embedder = SentenceTransformer(EMB_ID, device="cpu")
reranker = CrossEncoder(RERANKER_ID, device="cpu", max_length=512) if USAR_RERANKER else None


def buscar(pergunta: str) -> list[int]:
    s_bm25 = bm25.get_scores(tokenizar(pergunta))
    qv = embedder.encode([pergunta], normalize_embeddings=True)[0].astype(np.float32)
    s_dense = emb @ qv
    fusao = {}
    for scores in (s_bm25, s_dense):
        for pos, i in enumerate(np.argsort(-scores)[:CANDIDATOS]):
            fusao[int(i)] = fusao.get(int(i), 0.0) + 1.0 / (RRF_K + pos + 1)
    candidatos = sorted(fusao, key=lambda i: -fusao[i])[:CANDIDATOS]
    if reranker is not None:
        notas = reranker.predict([(pergunta, chunks.texto_indexado.iloc[i]) for i in candidatos])
        candidatos = [candidatos[j] for j in np.argsort(-np.asarray(notas))]
    return candidatos[:TOP_CHUNKS]


def contexto(ids: list[int]) -> str:
    return "\n\n---\n\n".join(
        f"[{c.lei}, Art. {c.artigo.split('#')[0]}] ({c.lei_nome})\n{c.texto}" for c in chunks.iloc[ids].itertuples()
    )


def gerar(pergunta: str, ctx: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "_Modo busca: defina o secret `HF_TOKEN` no Space para gerar respostas._"
    from huggingface_hub import InferenceClient

    try:
        out = InferenceClient(model=LLM_ID, token=token).chat_completion(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Trechos de lei:\n\n{ctx}\n\n=====\nPergunta: {pergunta}"}],
            max_tokens=400, temperature=0.0,
        )
        return out.choices[0].message.content
    except Exception as e:  # cota esgotada, modelo indisponível etc.: o app continua mostrando a busca
        return f"_Não consegui gerar a resposta agora ({type(e).__name__}). Veja os artigos encontrados abaixo._"


def responder(pergunta: str):
    if not pergunta or not pergunta.strip():
        return "Digite uma pergunta.", ""
    ids = buscar(pergunta)
    ctx = contexto(ids)
    fontes = "\n\n".join(
        f"**{c.lei}, Art. {c.artigo.split('#')[0]}** · {c.lei_nome}\n\n> " + c.texto[:700].replace("\n", "\n> ")
        for c in chunks.iloc[ids].itertuples()
    )
    return gerar(pergunta, ctx), fontes


EXEMPLOS = [
    "Comprei um celular pela internet e me arrependi. Posso devolver?",
    "Quero sair do apartamento alugado antes do fim do contrato. Pago multa?",
    "Quantas horas extras posso fazer por dia?",
    "Uma empresa pode usar meus dados sem o meu consentimento?",
    "Qual a pena para o crime de furto?",  # fora do escopo: o sistema deve recusar
]

with gr.Blocks(title="RAG Jurídico BR") as demo:
    gr.Markdown(
        "# ⚖️ RAG Jurídico BR\n"
        "Perguntas sobre **CDC, LGPD, Marco Civil da Internet, Lei do Inquilinato e CLT**, "
        "respondidas com citação dos artigos. Busca híbrida (BM25 + BGE-M3) com reranker.\n\n"
        "_Projeto educacional: não é aconselhamento jurídico._"
    )
    pergunta = gr.Textbox(label="Sua pergunta", placeholder="Ex.: Posso devolver um produto comprado online?")
    botao = gr.Button("Perguntar", variant="primary")
    resposta = gr.Markdown(label="Resposta")
    with gr.Accordion("Artigos usados como contexto", open=False):
        fontes = gr.Markdown()
    gr.Examples(EXEMPLOS, inputs=pergunta)
    botao.click(responder, inputs=pergunta, outputs=[resposta, fontes])
    pergunta.submit(responder, inputs=pergunta, outputs=[resposta, fontes])

if __name__ == "__main__":
    demo.launch()
