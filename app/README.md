---
title: RAG Jurídico BR
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Demo: RAG Jurídico BR

Interface da demo do projeto. Para publicar no Hugging Face Spaces:

1. Crie um Space (SDK **Gradio**, hardware **CPU basic** gratuito).
2. Copie para o Space: `app.py`, `requirements.txt`, este `README.md` e uma pasta `data/` com
   `chunks.parquet` e `emb_bge_m3.npy`, que vêm da **saída do notebook 02**.
3. Em *Settings → Variables and secrets*, crie o secret `HF_TOKEN` (um token de leitura da sua conta),
   que é usado para gerar as respostas pela API de inferência do Hugging Face.
   Sem ele, o app funciona em "modo busca" (mostra só os artigos encontrados).

Variáveis opcionais: `LLM_MODEL` (padrão `Qwen/Qwen2.5-7B-Instruct`), `USE_RERANKER` (`1`/`0`) e
`CANDIDATES` (padrão 15; menos candidatos deixam o reranker mais rápido na CPU).
