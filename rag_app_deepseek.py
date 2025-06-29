# === MOTOR RAG CON BÚSQUEDA HÍBRIDA Y ROUTER DE INTENCIÓN ===
# VERSIÓN 4.1 - CORRECCIÓN DE UNBOUNDLOCALERROR

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# --- Importaciones modernizadas de LangChain ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# --- Carga inicial de configuración ---
load_dotenv()

# --- FUNCIÓN DE DETECCIÓN DE INTENCIÓN ---
def detectar_intencion_de_cita(query: str) -> tuple[bool, str]:
    """Detecta si la pregunta del usuario es una solicitud de cita textual."""
    patrones = [
        r"c[íi]tame textualmente el art[íi]culo (\d+)",
        r"qu[ée] dice el art[íi]culo (\d+)",
        r"mu[ée]strame el art[íi]culo (\d+)",
        r"ver textualmente el art[íi]culo (\d+)",
        r"art[íi]culo (\d+) textualmente"
    ]
    for patron in patrones:
        match = re.search(patron, query, re.IGNORECASE)
        if match:
            return True, match.group(1)
    return False, None

# --- CLASE PRINCIPAL DEL MOTOR DE RAG ---
class RAG_Engine:
    def __init__(self, embedding_model="intfloat/multilingual-e5-large-instruct", llm_model="deepseek-chat"):
        print("Inicializando motor de RAG...")
        self.base_dir = Path(__file__).resolve().parent
        self.output_dir = self.base_dir / "procesado"
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("No se encontró la DEEPSEEK_API_KEY en el archivo .env.")
        self.db_faiss = None
        self.all_docs = None
        self.llm = None
        self.is_loaded = False
        print("Motor listo. Los componentes pesados se cargarán en la primera consulta.")

    def _load_components(self):
        """Carga todos los componentes pesados una sola vez."""
        if self.is_loaded: return
        print("Cargando componentes pesados por primera vez...")
        print(f"Cargando modelo de embeddings: {self.embedding_model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        print("Cargando base de datos vectorial FAISS...")
        self.db_faiss = FAISS.load_local(str(self.output_dir), embeddings, allow_dangerous_deserialization=True)
        print("Extrayendo documentos desde FAISS...")
        self.all_docs = list(self.db_faiss.docstore._dict.values())
        print("Configurando el LLM (DeepSeek)...")
        self.llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            model=self.llm_model_name,
            temperature=0.1, max_tokens=1500, streaming=True
        )
        self.is_loaded = True
        print("¡Todos los componentes base han sido cargados!")

    def _get_retriever(self, filtro_dominio: str):
        """Construye y devuelve el retriever híbrido aplicando el filtro de dominio dinámico."""
        print(f"Construyendo retriever para el dominio: '{filtro_dominio}'")
        docs_filtrados = self.all_docs
        if filtro_dominio == "Facultad de Ciencias":
            docs_filtrados = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'legislacion_unam']
        elif filtro_dominio == "R3D (Derechos Digitales)":
            docs_filtrados = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'r3d']
        if not docs_filtrados: return None
        bm25_retriever = BM25Retriever.from_documents(documents=docs_filtrados); bm25_retriever.k = 4
        search_filter = {}
        if filtro_dominio == "Facultad de Ciencias": search_filter = {"dominio": "legislacion_unam"}
        elif filtro_dominio == "R3D (Derechos Digitales)": search_filter = {"dominio": "r3d"}
        faiss_retriever = self.db_faiss.as_retriever(search_type="similarity", search_kwargs={'k': 4, 'filter': search_filter if search_filter else None})
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    def format_docs_for_prompt(self, docs: list[Document]) -> tuple[str, set]:
        """Formatea los documentos para el prompt y extrae las fuentes."""
        contexto, fuentes = [], set()
        for doc in docs:
            # --- LA CORRECCIÓN ESTÁ AQUÍ ---
            # 1. Primero asignamos 'meta'
            meta = doc.metadata
            # 2. Luego usamos 'meta' para obtener los demás valores
            fuente = meta.get('source', 'N/A')
            articulo = meta.get('article', None)
            
            header = f"Fuente: {fuente}" + (f", Artículo: {articulo}" if articulo else "")
            contexto.append(f"---\n{header}\nTexto: {doc.page_content}\n---")
            fuentes.add(fuente)
        return "\n".join(contexto), fuentes

    def get_verbatim_text(self, article_number: str, filtro_dominio: str):
        """Busca un artículo específico por su número y devuelve su texto formateado."""
        self._load_components()
        docs_a_buscar = self.all_docs
        if filtro_dominio == "Facultad de Ciencias":
            docs_a_buscar = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'legislacion_unam']
        
        for doc in docs_a_buscar:
            if doc.metadata.get('article') == article_number:
                fuente = doc.metadata.get('source', 'N/A')
                respuesta = f"**Cita textual del Artículo {article_number} encontrado en la fuente: *{fuente}***\n\n---\n\n"
                respuesta += doc.page_content
                yield respuesta
                yield {"fuentes": {fuente}}
                return
        yield f"No pude encontrar un Artículo con el número '{article_number}' en el ámbito de búsqueda seleccionado."
        yield {"fuentes": set()}

    def answer_question_stream(self, query: str, filtro_dominio: str):
        """Función principal que actúa como un ROUTER."""
        self._load_components()
        es_cita, numero_articulo = detectar_intencion_de_cita(query)

        if es_cita:
            print(f"Intención detectada: Cita textual del artículo {numero_articulo}.")
            yield from self.get_verbatim_text(numero_articulo, filtro_dominio)
            return
        
        print("Intención detectada: Pregunta de razonamiento (RAG).")
        try:
            retriever = self._get_retriever(filtro_dominio)
            if not retriever:
                yield "No hay documentos en el dominio seleccionado para realizar la búsqueda."
                return
            retrieved_docs = retriever.invoke(query)
            if not retrieved_docs:
                yield f"No se encontró información relevante en el dominio '{filtro_dominio}'."
                yield {"fuentes": set()}
                return
            contexto_str, fuentes = self.format_docs_for_prompt(retrieved_docs)
            prompt_template = f"""
            Eres un asistente experto. Usa solo el CONTEXTO para responder la PREGUNTA. Si la respuesta no está, dilo explícitamente. No cites fuentes aquí.
            --- CONTEXTO ---
            {contexto_str}
            --- FIN DEL CONTEXTO ---
            PREGUNTA DEL USUARIO: "{query}"
            Respuesta:
            """
            print("Generando respuesta con el LLM...")
            for chunk in self.llm.stream(prompt_template):
                yield chunk.content
            yield {"fuentes": fuentes}
        except Exception as e:
            print(f"Ocurrió un error en el motor de RAG: {e}")
            yield f"Ocurrió un error al procesar su solicitud. Detalle técnico: {e}"
            yield {"fuentes": set()}

# --- Instancia única para ser usada por la UI ---
rag_engine = RAG_Engine()