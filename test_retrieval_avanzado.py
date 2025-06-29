# test_retrieval_avanzado.py (v2.0 - Corregido)
import json
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

# --- CONFIGURACIÓN ---
OUTPUT_DIR = Path(__file__).resolve().parent / "procesado"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
TOP_K_RESULTS = 4 # Cuántos chunks queremos recuperar

# --- CARGAR LA BASE DE DATOS Y EL MODELO ---
print("Cargando modelo y base de datos FAISS...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db_faiss = FAISS.load_local(str(OUTPUT_DIR), embeddings, allow_dangerous_deserialization=True)
print("¡Carga completa!")

# --- PREPARAR LOS COMPONENTES PARA LA BÚSQUEDA HÍBRIDA ---

# --- LÓGICA CORREGIDA ---
# 1. Extraemos TODOS los documentos (texto + metadatos) directamente desde la docstore de FAISS.
# Esto es más robusto y nos da toda la información necesaria.
print("Extrayendo todos los documentos desde la base de datos FAISS...")
# Accedemos al diccionario interno de la docstore para obtener todos los Documentos
all_docs = list(db_faiss.docstore._dict.values())

# Filtramos para usar solo los del dominio que nos interesa en nuestros retrievers
all_docs_unam = [
    doc for doc in all_docs if doc.metadata.get('dominio') == 'legislacion_unam'
]
print(f"Se usarán {len(all_docs_unam)} chunks del dominio 'legislacion_unam' para la búsqueda.")


# 2. Creamos el retriever de Búsqueda por Palabras Clave (BM25)
print("Creando retriever de keywords (BM25)...")
bm25_retriever = BM25Retriever.from_documents(documents=all_docs_unam)
bm25_retriever.k = TOP_K_RESULTS

# 3. Creamos el retriever Vectorial (FAISS) con FILTRO DE METADATOS
print("Creando retriever vectorial (FAISS) con filtro de metadatos...")
faiss_retriever = db_faiss.as_retriever(
    search_type="similarity",
    search_kwargs={'k': TOP_K_RESULTS, 'filter': {'dominio': 'legislacion_unam'}}
)

# 4. Creamos el Ensemble Retriever (Búsqueda Híbrida)
print("Creando retriever híbrido (Ensemble)...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5] # 50% keywords, 50% semántica
)


# --- PREGUNTAS DE PRUEBA ---
preguntas_de_prueba = [
    "¿Cuáles son las opciones de titulación en la Facultad de Ciencias?",
    "¿Qué se necesita para obtener una mención honorífica?",
    "Artículo 95 del Estatuto General",
    "Causas graves de responsabilidad según el Artículo 95 del Estatuto General"
]

# --- EJECUTAR LAS PRUEBAS ---
for pregunta in preguntas_de_prueba:
    print("\n" + "="*80)
    print(f"PREGUNTA: {pregunta}")
    print("="*80)
    
    # Usamos el retriever híbrido para obtener los documentos
    retrieved_docs = ensemble_retriever.invoke(pregunta)
    
    print(f"Se encontraron {len(retrieved_docs)} chunks relevantes:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Fuente: {doc.metadata.get('source')}")
        if 'article' in doc.metadata and doc.metadata['article']:
            print(f"Artículo: {doc.metadata['article']}")
        print("Texto del Chunk:")
        print(doc.page_content[:350] + "...")