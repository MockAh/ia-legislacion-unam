# === MOTOR RAG CON BÚSQUEDA HÍBRIDA Y AUTOCONSTRUCCIÓN DE BASE DE DATOS ===
# VERSIÓN 5.0 (PRODUCCIÓN)

import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# --- Importaciones de Terceros ---
import fitz  # PyMuPDF
import docx2txt
import pytesseract
from tqdm import tqdm
from pdf2image import convert_from_path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# --- Carga inicial de configuración ---
load_dotenv()

# --- Definición de la Clase Principal del Motor de RAG ---
class RAG_Engine:
    def __init__(self, embedding_model="intfloat/multilingual-e5-large-instruct", llm_model="deepseek-chat"):
        print("✅ Inicializando motor de RAG...")
        # Configuración de rutas y modelos
        self.base_dir = Path(__file__).resolve().parent
        self.input_docs_dir = self.base_dir / "Juegos"
        self.output_dir = self.base_dir / "procesado"
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model

        # Verificación de la clave de API
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("No se encontró la DEEPSEEK_API_KEY en el archivo .env.")

        # Componentes que se cargarán de forma perezosa (lazy loading)
        self.db_faiss = None
        self.all_docs = None
        self.llm = None
        self.is_loaded = False
        print("✅ Motor listo. Los componentes se cargarán y la DB se construirá si es necesario en la primera consulta.")
    
    # --- NUEVA SECCIÓN: LÓGICA DE PROCESAMIENTO DE DOCUMENTOS (TRAÍDA DE script_embeddings_01.py) ---
    def _parse_filename_for_metadata(self, filename: str, dominio: str) -> dict:
        name_without_ext = Path(filename).stem; parts = name_without_ext.split('_'); metadata = {}
        if dominio == 'legislacion_unam':
            if len(parts) == 5: metadata["entidad_unam"], metadata["area_interna"], metadata["titulo"], metadata["tipo_documento"], metadata["año"] = [p.replace('-', ' ') for p in parts]
            else: metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
        else: metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
        return metadata

    def _robust_text_extraction(self, file_path: Path) -> str:
        # Simplificado para el bootstrap, asumiendo que los archivos especiales ya están manejados
        path_str = str(file_path); raw_text = ""
        try:
            if path_str.endswith('.pdf'):
                with fitz.open(path_str) as doc: raw_text = "".join([page.get_text("text", sort=True) for page in doc])
            elif path_str.endswith('.docx'): raw_text = docx2txt.process(path_str)
            elif path_str.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f: raw_text = f.read()
        except Exception as e: print(f"  (ERROR extrayendo {file_path.name}: {e})"); return ""
        text = raw_text.replace('-\n', ''); text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text); text = re.sub(r' +', ' ', text)
        return text.strip()

    def _chunk_by_article(self, text: str) -> list[str]:
        patron = r"(\bART[IÍ]CULO \d+[°º]?[.\-]?\s?)"; parts = re.split(patron, text, flags=re.IGNORECASE)
        if len(parts) <= 1: return [text] if text.strip() else []
        initial_chunks = [parts[0].strip()] if parts[0].strip() else []
        for i in range(1, len(parts), 2): initial_chunks.append((parts[i] + parts[i+1]).strip())
        if not initial_chunks: return []
        merged_chunks = [initial_chunks[0]]
        for i in range(1, len(initial_chunks)):
            current_chunk, previous_chunk = initial_chunks[i], merged_chunks[-1]
            current_match, previous_match = re.search(r'ART[IÍ]CULO (\d+)', current_chunk, re.IGNORECASE), re.search(r'ART[IÍ]CULO (\d+)', previous_chunk, re.IGNORECASE)
            if current_match and previous_match and current_match.group(1) == previous_match.group(1): merged_chunks[-1] = previous_chunk + "\n\n" + current_chunk
            else: merged_chunks.append(current_chunk)
        return merged_chunks

    def _chunk_normatividad_general(self, text: str) -> list[str]:
        patron_numerico = r"(\b\d+(\.\d+)*\.\s)"; parts = re.split(patron_numerico, text)
        if len(parts) <= 1: return [chunk for chunk in text.split('\n\n') if chunk.strip()]
        initial_chunks = [parts[0].strip()] if parts[0].strip() else []
        for i in range(1, len(parts), 2):
            delimiter, content = parts[i], parts[i+1] if i + 1 < len(parts) else ""
            chunk_parts = [p for p in [delimiter, content] if p]; 
            if chunk_parts: initial_chunks.append("".join(chunk_parts).strip())
        if not initial_chunks: return []
        merged_chunks = [initial_chunks[0]]
        patron_extraccion = re.compile(r"^(\d+(\.\d+)*\.)")
        for i in range(1, len(initial_chunks)):
            current_chunk, previous_chunk = initial_chunks[i], merged_chunks[-1]
            current_match, previous_match = patron_extraccion.match(current_chunk), patron_extraccion.match(previous_chunk)
            if current_match and previous_match and current_match.group(0).startswith(previous_match.group(0)): merged_chunks[-1] = previous_chunk + "\n\n" + current_chunk
            else: merged_chunks.append(current_chunk)
        return merged_chunks

    # --- NUEVA FUNCIÓN DE AUTOCONSTRUCCIÓN ---
    def _bootstrap_database(self, embeddings):
        """
        Verifica si la base de datos existe. Si no, la construye desde cero.
        """
        if self.output_dir.exists() and (self.output_dir / "index.faiss").exists():
            print("Base de datos encontrada localmente. Cargando...")
            return

        print("¡ADVERTENCIA! Base de datos no encontrada. Iniciando proceso de construcción por primera vez...")
        print("Este proceso puede tardar varios minutos y solo ocurrirá una vez (o si la app se reinicia).")
        
        self.output_dir.mkdir(exist_ok=True)
        all_files = [p for p in self.input_docs_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.pdf', '.docx', '.txt']]
        
        docs_to_embed = []
        for file_path in tqdm(all_files, desc="Procesando documentos para la base de datos"):
            filename, dominio = file_path.name, file_path.relative_to(self.input_docs_dir).parts[0]
            processed_text = self._robust_text_extraction(file_path)
            if not processed_text: continue
            
            chunks = []
            if dominio == 'legislacion_unam':
                if re.search(r'\bART[IÍ]CULO ', processed_text, re.IGNORECASE): chunks = self._chunk_by_article(processed_text)
                else: chunks = self._chunk_normatividad_general(processed_text)
            else: chunks = [chunk for chunk in processed_text.split('\n\n') if chunk.strip()]
            
            base_metadata = self._parse_filename_for_metadata(filename, dominio)
            base_metadata.update({"dominio": dominio, "source": filename})
            
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                if dominio == 'legislacion_unam':
                    article_match = re.search(r'ART[IÍ]CULO (\d+)', chunk_text, re.IGNORECASE)
                    if article_match: chunk_metadata["article"] = article_match.group(1)
                chunk_metadata["chunk_index"] = i
                docs_to_embed.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        if not docs_to_embed: raise RuntimeError("No se pudieron generar chunks para la base de datos.")
        
        print(f"\nGenerando embeddings para {len(docs_to_embed)} chunks...")
        db = FAISS.from_documents(docs_to_embed, embeddings)
        db.save_local(str(self.output_dir))
        
        print(f"✅ Base de datos FAISS construida y guardada exitosamente en: '{self.output_dir}'")
        
    def _load_components(self):
        """Carga todos los componentes pesados, asegurándose de que la DB exista."""
        if self.is_loaded: return
        
        print("Iniciando carga de componentes...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # --- LLAMADA A LA FUNCIÓN DE AUTOCONSTRUCCIÓN ---
        self._bootstrap_database(embeddings)
        
        print("Cargando base de datos vectorial FAISS...")
        self.db_faiss = FAISS.load_local(str(self.output_dir), embeddings, allow_dangerous_deserialization=True)
        print("Extrayendo documentos desde FAISS para el retriever de keywords...")
        self.all_docs = list(self.db_faiss.docstore._dict.values())
        print("Configurando el LLM (DeepSeek)...")
        self.llm = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1", model=self.llm_model_name, temperature=0.1, max_tokens=1500, streaming=True)
        
        self.is_loaded = True
        print("✅ ¡Todos los componentes están cargados y listos!")
    
    # --- El resto de las funciones (get_retriever, format_docs_for_prompt, get_verbatim_text, answer_question_stream) se mantienen igual que en la v4.1 ---
    def _get_retriever(self, filtro_dominio: str):
        print(f"Construyendo retriever para el dominio: '{filtro_dominio}'")
        docs_filtrados = self.all_docs
        if filtro_dominio == "Facultad de Ciencias": docs_filtrados = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'legislacion_unam']
        elif filtro_dominio == "R3D (Derechos Digitales)": docs_filtrados = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'r3d']
        if not docs_filtrados: return None
        bm25_retriever = BM25Retriever.from_documents(documents=docs_filtrados); bm25_retriever.k = 4
        search_filter = {}; 
        if filtro_dominio == "Facultad de Ciencias": search_filter = {"dominio": "legislacion_unam"}
        elif filtro_dominio == "R3D (Derechos Digitales)": search_filter = {"dominio": "r3d"}
        faiss_retriever = self.db_faiss.as_retriever(search_type="similarity", search_kwargs={'k': 4, 'filter': search_filter if search_filter else None})
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    def format_docs_for_prompt(self, docs: list[Document]) -> tuple[str, set]:
        contexto, fuentes = [], set()
        for doc in docs:
            meta = doc.metadata; fuente = meta.get('source', 'N/A'); articulo = meta.get('article', None)
            header = f"Fuente: {fuente}" + (f", Artículo: {articulo}" if articulo else "")
            contexto.append(f"---\n{header}\nTexto: {doc.page_content}\n---")
            fuentes.add(fuente)
        return "\n".join(contexto), fuentes

    def get_verbatim_text(self, article_number: str, filtro_dominio: str):
        self._load_components()
        docs_a_buscar = self.all_docs
        if filtro_dominio == "Facultad de Ciencias": docs_a_buscar = [doc for doc in self.all_docs if doc.metadata.get('dominio') == 'legislacion_unam']
        for doc in docs_a_buscar:
            if doc.metadata.get('article') == article_number:
                fuente = doc.metadata.get('source', 'N/A')
                respuesta = f"**Cita textual del Artículo {article_number} de la fuente: *{fuente}***\n\n---\n\n" + doc.page_content
                yield respuesta; yield {"fuentes": {fuente}}; return
        yield f"No pude encontrar un Artículo '{article_number}' en el ámbito de búsqueda."; yield {"fuentes": set()}

    def answer_question_stream(self, query: str, filtro_dominio: str):
        self._load_components()
        es_cita, numero_articulo = detectar_intencion_de_cita(query)
        if es_cita:
            print(f"Intención detectada: Cita textual del artículo {numero_articulo}.")
            yield from self.get_verbatim_text(numero_articulo, filtro_dominio)
            return
        print("Intención detectada: Pregunta de razonamiento (RAG).")
        try:
            retriever = self._get_retriever(filtro_dominio)
            if not retriever: yield "No hay documentos en el dominio seleccionado."; return
            retrieved_docs = retriever.invoke(query)
            if not retrieved_docs:
                yield f"No se encontró información relevante en el dominio '{filtro_dominio}'."; yield {"fuentes": set()}; return
            contexto_str, fuentes = self.format_docs_for_prompt(retrieved_docs)
            prompt_template = f"Eres un asistente experto. Usa solo el CONTEXTO para responder la PREGUNTA. Si la respuesta no está, dilo. No cites fuentes aquí.\n--- CONTEXTO ---\n{contexto_str}\n--- FIN DEL CONTEXTO ---\nPREGUNTA DEL USUARIO: \"{query}\"\nRespuesta:"
            print("Generando respuesta con el LLM...")
            for chunk in self.llm.stream(prompt_template): yield chunk.content
            yield {"fuentes": fuentes}
        except Exception as e:
            print(f"Ocurrió un error en el motor de RAG: {e}"); yield f"Ocurrió un error al procesar su solicitud. Detalle técnico: {e}"; yield {"fuentes": set()}

# --- Instancia única para ser usada por la UI ---
rag_engine = RAG_Engine()