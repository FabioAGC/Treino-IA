# ⚖️ RAG Jurídico BR: busca e resposta sobre leis brasileiras, com avaliação rigorosa

Sistema de **Retrieval-Augmented Generation (RAG)** que responde perguntas de cidadãos sobre
**CDC, LGPD, Marco Civil da Internet, Lei do Inquilinato e CLT**, citando os artigos usados.

O foco do projeto não é só "fazer um RAG funcionar", e sim **medir** cada decisão:
qual busca encontra o artigo certo, quanto o RAG melhora em relação ao LLM sozinho,
se o sistema sabe dizer "não sei" e o quanto dá para confiar no LLM usado como juiz.

> 🔗 **Demo:** _link do Hugging Face Space_ · **Notebooks no Kaggle:** _links_

## Resultados

> Preencha depois de rodar os notebooks: os números saem em `metricas_retrieval.csv` e `metricas_geracao.csv`.

**Retrieval**: N perguntas com artigo esperado (sintéticas filtradas + golden escritas à mão)

| Método | Recall@1 | Recall@5 | MRR@10 | Latência |
|---|---|---|---|---|
| BM25 | | | | |
| multilingual-e5-base | | | | |
| BGE-M3 | | | | |
| Híbrido (RRF) | | | | |
| **Híbrido + reranker** | | | | |

**Geração** (mesmo LLM, Qwen3-4B-Instruct)

| Sistema | Respostas corretas | Cita o artigo certo | Recusa fora do escopo | Falsa recusa |
|---|---|---|---|---|
| Sem RAG | | | | |
| RAG + BM25 | | | | |
| RAG + melhor retriever | | | | |

**Confiabilidade do juiz:** concordância com anotação humana em N respostas: κ ponderado = _x_.

![Recall@k](docs/recall_at_k.png)

## Arquitetura

```mermaid
flowchart LR
    A[Planalto: 5 leis em HTML] --> B[Limpeza: remove texto revogado e anotações]
    B --> C[Chunks por artigo]
    C --> D[BM25]
    C --> E[BGE-M3 embeddings]
    Q[Pergunta] --> D & E
    D & E --> F[Reciprocal Rank Fusion]
    F --> G[Reranker bge-reranker-v2-m3]
    G --> H[Top-5 trechos]
    H --> I[Qwen3-4B: resposta com citação LEI, Art. N]
```

## Como a avaliação foi construída

| Conjunto | Como foi feito | Para que serve |
|---|---|---|
| **Sintéticas** | Qwen3-4B gera uma pergunta por artigo sorteado (estratificado por lei); Qwen3-8B dá notas e descarta as ruins | volume para comparar métodos |
| **Golden** | 33 perguntas escritas à mão, em linguagem do dia a dia | medida honesta; evita o viés de "LLM avaliando perguntas de LLM" |
| **Fora do escopo** | 20 perguntas que as leis do corpus não respondem (penal, trânsito, família…) | mede se o sistema recusa em vez de alucinar |

**Métricas**
- **Retrieval:** Recall@k, MRR@10 e nDCG@10 no nível de artigo, com **IC 95% por bootstrap**
  e **teste pareado** entre os melhores métodos.
- **Geração, determinísticas:** citação do artigo correto, citação de artigo que não estava no contexto,
  recusa correta e falsa recusa. Tudo por regex, porque o formato de citação e a frase de recusa são fixos no prompt.
- **Geração, LLM-as-a-judge:** correção (vs. resposta de referência) e fidelidade ao contexto (1–5).
- **Validação do juiz:** 40 respostas anotadas à mão, com o κ de Cohen ponderado e a correlação de Spearman entre a pessoa e o juiz.

## Decisões de projeto

- **Texto riscado removido.** Nas versões compiladas do Planalto, o texto revogado continua na página, só que riscado.
  Sem filtrar, o RAG responderia com a lei antiga.
- **Chunk = artigo.** É a unidade natural da lei e permite avaliar o retrieval pelo artigo correto.
  Artigos longos (comuns na CLT) viram sub-chunks com o mesmo `artigo_id`.
- **Busca híbrida.** O BM25 acerta termos exatos ("caução", "justa causa"); os embeddings acertam paráfrases
  ("me arrependi da compra" → direito de arrependimento). O RRF combina as duas listas sem calibrar escalas.
- **Formato de citação fixo `[LEI, Art. N]`.** Permite medir alucinação de citação sem depender de outro LLM.

## Limitações

- O juiz (Qwen3-8B) é da mesma família do gerador e pode favorecê-lo. Por isso existe a validação humana.
- As perguntas sintéticas vêm de um único artigo; perguntas reais às vezes dependem de vários.
- O corpus cobre 5 leis e não inclui jurisprudência nem regulamentações.
- **Projeto educacional, não é aconselhamento jurídico.**

## Como reproduzir (Kaggle, ~2h de GPU no total)

| Notebook | O que faz | GPU |
|---|---|---|
| [`01_corpus_e_dataset`](notebooks/01_corpus_e_dataset.ipynb) | baixa e limpa as leis, monta os chunks e o dataset de avaliação | ~40 min |
| [`02_avaliacao_retrieval`](notebooks/02_avaliacao_retrieval.ipynb) | compara 5 métodos de busca e salva o índice para o app | ~20 min |
| [`03_avaliacao_geracao`](notebooks/03_avaliacao_geracao.ipynb) | Sem RAG vs. RAG, juiz LLM e validação humana | ~1h30 |

1. No Kaggle: *Create → Notebook → File → Import Notebook* e escolha o `.ipynb`.
2. *Settings*: **Accelerator = GPU T4 x2** e **Internet = On** (exige verificar o telefone na conta).
3. Rode com **Save Version → Save & Run All**: a execução continua mesmo com a aba fechada.
4. No notebook seguinte, use *Add Input → Your Work* para adicionar a saída do anterior.
5. Demo: veja [`app/README.md`](app/README.md).

Cada notebook também roda fora do Kaggle num modo de teste (`SMOKE_TEST=1`), que troca os modelos por versões
minúsculas para validar o código na CPU.

## Estrutura

```
notebooks/   01 → 02 → 03 (cada um usa a saída do anterior)
app/         demo em Gradio para Hugging Face Spaces
docs/        gráficos usados neste README
```

## Stack

Python · Hugging Face Transformers · sentence-transformers · rank-bm25 · bitsandbytes (4 bits) ·
pandas · scikit-learn · Gradio · Kaggle (GPU T4)
