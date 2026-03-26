import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import logging
    import shutil
    import json
    from typing import List, Optional
    from dotenv import load_dotenv

    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    return (
        Chroma,
        CrossEncoder,
        Document,
        List,
        PyPDFLoader,
        RecursiveCharacterTextSplitter,
        SentenceTransformer,
        load_dotenv,
        logging,
        mo,
        os,
        shutil,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Config
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(load_dotenv, logging):
    # --- Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Load Env Vars ---
    load_dotenv()
    return (logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classes
    """)
    return


@app.cell
def _(logger, mo, os, shutil, torch):
    # ✅ Embeddings: Multilingual E5 (verified model ID)
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    # Alternative (lighter): "intfloat/multilingual-e5-small"

    # ✅ Reranker: BGE v2 m3 (verified model ID - NOT "multilingual-base")
    RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
    # Alternative (lighter): "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # ✅ LLM: Local inference with Phi-3-mini (lightweight, multilingual)
    # This model will be downloaded and run locally - no API token needed!
    LLM_REPO_ID = "bigscience/bloomz-1b7"  # ~3.5GB, no RoPE, fast loading
    LLM_MAX_NEW_TOKENS = 512
    LLM_TEMPERATURE = 0.1

    # Path to your PDF file (use absolute path for reliability)
    PDF_PATH = "./source/NLP/pdf/TCC  versão 4.6.pdf"

    # Array of questions to ask
    QUESTIONS = [
        'No que tange ao endereçamento, qual a principal diferença entre o IPv4 e o IPv6?',
        'As duas versões de protocolo poderão coexistir?',
        'Ao usar a opção de Jumbo Payload, qual o tamanho máximo suportado?',
        'Qual campo define os endereços das rotas que o pacote deve percorrer?',
        'Em que condição é necessário fragmentar um pacote na transmissão?',
        'O Authentication Header impede que a informação que está sendo transmitida seja descoberta em caso de análise da rede?',
        'Quantas vezes os cabeçalhos opcionais de extensão podem aparecer num pacote IPv6?'
    ]

    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100

    TOP_K_INITIAL = 15
    TOP_K_FINAL = 4

    PERSIST_DIRECTORY = "./source/NLP/chroma_langchain_db"
    COLLECTION_NAME = "pdf_rag_collection"

    CLEAR_DB = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SYSTEM_PROMPT = """Você é um assistente útil que responde perguntas em português do Brasil (pt-BR) com base APENAS no contexto fornecido. Se a resposta não estiver no contexto, diga que não sabe. Não invente informações.

    Contexto:
    {context}

    Pergunta: {question}

    Resposta (em português):"""

    # Check VRAM for local LLM
    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Available VRAM: {vram_gb:.2f} GB")
        if vram_gb < 6:
            logger.warning("⚠️ Low VRAM detected. Consider using a smaller model or HF API.")

    HF_TOKEN = os.getenv("HF_TOKEN")

    from huggingface_hub import whoami

    user = whoami(token=HF_TOKEN)

    print(user)

    # =============================================================================
    # Validation
    # =============================================================================

    if not os.path.exists(PDF_PATH):
        raise ValueError(f"PDF file not found: {PDF_PATH}")

    if CLEAR_DB and os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        logger.info("Cleared existing Chroma DB directory.")

    mo.md(f"### ✅ Configuração Carregada\n- **PDF**: {PDF_PATH}\n- **Perguntas**: {len(QUESTIONS)}\n- **Dispositivo**: {DEVICE}\n- **LLM**: {LLM_REPO_ID} (local)")
    return (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        DEVICE,
        EMBEDDING_MODEL_NAME,
        LLM_MAX_NEW_TOKENS,
        LLM_REPO_ID,
        LLM_TEMPERATURE,
        PDF_PATH,
        PERSIST_DIRECTORY,
        QUESTIONS,
        RERANKER_MODEL_NAME,
        SYSTEM_PROMPT,
        TOP_K_FINAL,
        TOP_K_INITIAL,
    )


@app.cell
def _(List, SentenceTransformer, logger):
    class CustomEmbeddings:
        """Multilingual SentenceTransformer embeddings."""
        def __init__(self, model_name: str, device: str):
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            logger.info(f"✅ Loaded embedding model: {model_name}")

        def _add_prefix(self, texts: List[str], is_query: bool = False) -> List[str]:
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            texts_with_prefix = self._add_prefix(texts, is_query=False)
            embeddings = self.model.encode(texts_with_prefix, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()

        def embed_query(self, text: str) -> List[float]:
            text_with_prefix = self._add_prefix([text], is_query=True)[0]
            embedding = self.model.encode(text_with_prefix, normalize_embeddings=True)
            return embedding.tolist()

    return (CustomEmbeddings,)


@app.cell
def _(CrossEncoder, Document, List, logger):
    class CustomReranker:
        """Multilingual CrossEncoder Reranker."""
        def __init__(self, model_name: str, top_k: int, device: str):
            self.model = CrossEncoder(model_name, device=device)
            self.top_k = top_k
            logger.info(f"✅ Loaded reranker model: {model_name}")

        def rerank(self, query: str, documents: List[Document]) -> List[Document]:
            if not documents:
                return []
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.top_k]]

    return (CustomReranker,)


@app.cell
def _(logger, torch):
    class CustomLLM:
        """Simplified LLM for models without complex RoPE configs."""

        def __init__(self, repo_id: str, device: str, max_new_tokens: int = 512, temperature: float = 0.1):
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading simplified LLM: {repo_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            logger.info(f"✅ Simplified LLM loaded: {repo_id}")

        def invoke(self, prompt: str) -> str:
            # Simple prompt format for BLOOMZ/GPT-Neo
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            result = self.pipeline(formatted)
            response = result[0]['generated_text'].replace(formatted, "").strip()
            return response if response else "⚠️ Resposta vazia."

    return (CustomLLM,)


@app.cell
def _(
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    Chroma,
    CustomEmbeddings,
    CustomLLM,
    CustomReranker,
    DEVICE,
    Document,
    EMBEDDING_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_REPO_ID,
    LLM_TEMPERATURE,
    List,
    PERSIST_DIRECTORY,
    PyPDFLoader,
    RERANKER_MODEL_NAME,
    RecursiveCharacterTextSplitter,
    SYSTEM_PROMPT,
    TOP_K_FINAL,
    TOP_K_INITIAL,
    logger,
):
    class BatchRAGPipeline:
        """RAG pipeline with local LLM inference."""

        def __init__(self):
            self.vectorstore = None
            self._initialize_models()

        def _initialize_models(self):
            logger.info(f"Loading multilingual models on {DEVICE}...")
            self.embeddings = CustomEmbeddings(EMBEDDING_MODEL_NAME, DEVICE)
            self.reranker = CustomReranker(RERANKER_MODEL_NAME, TOP_K_FINAL, DEVICE)
            self.llm = CustomLLM(
                repo_id=LLM_REPO_ID, 
                device=DEVICE,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE
            )
            logger.info("✅ All models initialized (local inference).")

        def ingest_pdf(self, pdf_path: str) -> int:
            logger.info(f"Ingesting pt-BR document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            texts = text_splitter.split_documents(documents)

            for i, doc in enumerate(texts):
                doc.metadata['id'] = i
                doc.metadata['language'] = 'pt-BR'

            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=PERSIST_DIRECTORY,
                collection_name=COLLECTION_NAME
            )

            logger.info(f"✅ Ingested {len(texts)} pt-BR chunks into Chroma DB.")
            return len(texts)

        def _retrieve_and_rerank(self, query: str) -> List[Document]:
            initial_docs = self.vectorstore.similarity_search(query, k=TOP_K_INITIAL)
            return self.reranker.rerank(query, initial_docs)

        def _build_prompt(self, query: str, context: str) -> str:
            return SYSTEM_PROMPT.format(context=context, question=query)

        def query(self, question: str) -> dict:
            if not self.vectorstore:
                return {"error": "Vector store not initialized"}
            try:
                relevant_docs = self._retrieve_and_rerank(question)

                if not relevant_docs:
                    return {
                        "question": question,
                        "answer": "❌ Nenhum documento relevante encontrado.",
                        "sources": [],
                        "status": "no_results"
                    }

                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                prompt = self._build_prompt(question, context)
                response = self.llm.invoke(prompt)

                return {
                    "question": question,
                    "answer": response,
                    "sources": [
                        {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
                        for doc in relevant_docs
                    ],
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"Query error for '{question}': {e}")
                return {
                    "question": question,
                    "answer": f"❌ Erro: {str(e)}",
                    "sources": [],
                    "status": "error",
                    "error": str(e)
                }

        def batch_query(self, questions: List[str]) -> List[dict]:
            results = []
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing pt-BR question {i}/{len(questions)}: {question[:50]}...")
                result = self.query(question)
                results.append(result)
            return results

    return (BatchRAGPipeline,)


@app.cell
def _(BatchRAGPipeline, PDF_PATH):
    # Initialize and run multilingual pipeline
    batch_pipeline = BatchRAGPipeline()
    ingest_count = batch_pipeline.ingest_pdf(PDF_PATH)  
    return batch_pipeline, ingest_count


@app.cell
def _(QUESTIONS, batch_pipeline, ingest_count, mo):
    batch_results = batch_pipeline.batch_query(QUESTIONS)

    mo.md(f"### ✅ Execução Concluída\n- **Documentos Ingeridos**: {ingest_count} chunks\n- **Perguntas Processadas**: {len(batch_results)}")  
    return (batch_results,)


@app.cell
def _(batch_results, ingest_count):
    # --- Display Results ---
    if batch_results:
        # Summary Stats
        success_count = sum(1 for r in batch_results if r["status"] == "success")
        error_count = sum(1 for r in batch_results if r["status"] == "error")
        no_results_count = sum(1 for r in batch_results if r["status"] == "no_results")

        print(f"""## 📊 Batch Results Summary
    - **Total Questions**: {len(batch_results)}
    - **Successful**: {success_count} ✅
    - **No Results**: {no_results_count} ⚠️
    - **Errors**: {error_count} ❌
    - **Documents Ingested**: {ingest_count} chunks
        """)

        # Individual Results
        for i, result in enumerate(batch_results, 1):
            status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "error" else "⚠️"

            print(f"""### {status_icon} Question {i}
    **Q**: {result["question"]}

    **A**: {result["answer"]}
            """)

            if result.get("sources"):
                sources_text = "\n".join([f"- {s['content']}" for s in result["sources"]])
                print(f"""**Sources**:
    {sources_text}
                """)

            if result.get("error"):
                print(f"""**Error Details**:
    ```\n{result["error"]}\n```
                """)
    return


if __name__ == "__main__":
    app.run()
