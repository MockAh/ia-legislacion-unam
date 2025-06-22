# === APLICACIÓN RAG CON DEEPSEEK (USANDO CLIENTE COMPATIBLE OPENAI) ===
# VERSIÓN 2.0

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI  # <--- CAMBIO IMPORTANTE: Importamos OpenAI en lugar de DeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURACIÓN Y CARGA INICIAL ---

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROCESADO_DIR = BASE_DIR / "procesado"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("No se encontró la DEEPSEEK_API_KEY. Asegúrate de crear un archivo .env con tu clave.")

print("✅ Configuración cargada.")

# --- 2. INICIALIZACIÓN DE COMPONENTES ---

def inicializar_sistema():
    """
    Carga todos los componentes necesarios para el RAG.
    """
    print("Iniciando componentes del sistema RAG...")

    if not PROCESADO_DIR.exists() or not (PROCESADO_DIR / "index.faiss").exists():
        print("\n--- ERROR ---")
        print(f"El directorio '{PROCESADO_DIR}' o el índice FAISS no existen.")
        print("Por favor, ejecuta primero 'script_embeddings_01.py' para generar la base de datos.")
        return None, None

    print(f"Cargando modelo de embeddings local: {EMBEDDING_MODEL}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Error al cargar el modelo de embeddings: {e}")
        return None, None

    print("Cargando base de datos vectorial FAISS...")
    try:
        db = FAISS.load_local(
            str(PROCESADO_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error al cargar la base de datos FAISS: {e}")
        return None, None

    # --- CAMBIO IMPORTANTE: INICIALIZACIÓN DEL CLIENTE ---
    # Ahora usamos el cliente de OpenAI, pero le decimos que apunte a la URL de DeepSeek.
    # Esta es la forma oficial y más estable de hacerlo.
    print("Inicializando cliente compatible con OpenAI para DeepSeek...")
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    print("✅ Sistema RAG listo para operar.")
    return db, client

# --- 3. LÓGICA PRINCIPAL DEL RAG (Esta sección no necesita cambios) ---

def obtener_respuesta_rag(db, client, query):
    """
    Realiza el ciclo completo de RAG: búsqueda, construcción del prompt y generación.
    """
    start_time = time.time()

    try:
        retrieved_docs = db.similarity_search(query, k=4)
    except Exception as e:
        print(f"Error durante la búsqueda de similitud: {e}")
        return

    contexto = ""
    fuentes = set()

    for doc in retrieved_docs:
        contexto += f"--- Fragmento de: {doc.metadata.get('source', 'N/A')} ---\n"
        contexto += f"{doc.page_content}\n"
        contexto += f"[Metadatos: Título='{doc.metadata.get('titulo', 'N/A')}', Año='{doc.metadata.get('año', 'N/A')}']\n\n"
        fuentes.add(doc.metadata.get('source', 'N/A'))

    if not retrieved_docs:
        print("\nNo se encontraron documentos relevantes en la base de datos.")
        return

    prompt_template = f"""
    Eres un asistente de IA especializado en la legislación y normatividad de la UNAM, actuando con la máxima precisión y objetividad.
    INSTRUCCIONES:
    1. Tu única fuente de verdad es el CONTEXTO que te proporciono. NO uses conocimiento externo.
    2. Responde a la PREGUNTA DEL USUARIO basándote exclusivamente en el CONTEXTO.
    3. Si la respuesta no está en el CONTEXTO, responde: "La información solicitada no se encuentra en los documentos disponibles."
    4. Al final de tu respuesta, DEBES citar las fuentes utilizadas bajo un título "Fuentes Consultadas:".
    5. Responde siempre en español.
    --- CONTEXTO ---
    {contexto}
    --- FIN DEL CONTEXTO ---
    PREGUNTA DEL USUARIO: "{query}"
    Respuesta:
    """

    print("\n\n--- Respuesta de DeepSeek ---")
    try:
        # ¡Esta parte del código funciona igual gracias a la compatibilidad de la API!
        stream_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Sigue las instrucciones del prompt del usuario."},
                {"role": "user", "content": prompt_template}
            ],
            stream=True,
            max_tokens=1024,
            temperature=0.1
        )

        full_response = ""
        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content

        end_time = time.time()
        print(f"\n\n(Tiempo de respuesta: {end_time - start_time:.2f} segundos)")

    except Exception as e:
        print(f"\nError al llamar a la API de DeepSeek: {e}")


# --- 4. PUNTO DE ENTRADA INTERACTIVO (Sin cambios) ---

if __name__ == "__main__":
    db_instance, client_instance = inicializar_sistema()

    if db_instance and client_instance:
        print("\n--- Asistente de Legislación UNAM (usando DeepSeek) ---")
        print("Escribe tu pregunta y presiona Enter. Escribe 'salir' o presiona Ctrl+C para terminar.")

        while True:
            try:
                user_query = input("\nPregunta: ")
                if user_query.lower() in ["salir", "exit", "quit"]:
                    break
                if user_query:
                    obtener_respuesta_rag(db_instance, client_instance, user_query)
            except KeyboardInterrupt:
                print("\n\nCerrando aplicación. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"Ocurrió un error inesperado: {e}")
                break