# ui_app.py
# VERSIÓN 1.0 - Lista para Despliegue

import streamlit as st
import time

# Reutilizamos las funciones y la configuración de tu script RAG.
# ¡Asegúrate de que este archivo esté en la misma carpeta que rag_app_deepseek.py!
from rag_app_deepseek import inicializar_sistema, DEEPSEEK_MODEL_NAME

# --- 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DEL SISTEMA ---

st.set_page_config(
    page_title="Asistente Legislación UNAM",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Asistente de Legislación UNAM (Facultad de Ciencias)")
st.caption(f"Potenciado por DeepSeek ({DEEPSEEK_MODEL_NAME}) y una base de datos local.")

# Usamos el cache de Streamlit para inicializar el sistema solo una vez.
# Esto es CRUCIAL para el rendimiento en la nube.
@st.cache_resource
def cargar_recursos():
    """Carga la base de datos y el cliente de API. Muestra un spinner durante la carga."""
    # El spinner solo se mostrará la primera vez que un usuario cargue la app.
    with st.spinner("Iniciando sistema por primera vez... Cargando base de datos y modelos. Esto puede tardar unos segundos."):
        db, client = inicializar_sistema()
    return db, client

db_instance, client_instance = cargar_recursos()

# Manejo de error si el sistema no pudo inicializar (p. ej., falta el índice FAISS)
if not db_instance or not client_instance:
    st.error("El sistema no pudo inicializarse. Verifica que el índice FAISS esté en el repositorio y que los 'Secrets' de la API estén configurados en Streamlit Cloud.", icon="🚨")
    st.stop() # Detiene la ejecución de la app si hay un error crítico.


# --- 2. LÓGICA DE GENERACIÓN DE RESPUESTA PARA LA UI ---

def generar_respuesta_stream(query, db, client):
    """
    Esta función adapta tu lógica de RAG para que funcione con st.write_stream.
    En lugar de imprimir, 'yield' entrega cada fragmento de la respuesta.
    """
    # 1. Búsqueda de similitud (igual que en tu script)
    retrieved_docs = db.similarity_search(query, k=5) 
    
    if not retrieved_docs:
        yield "La información solicitada no se encuentra en los documentos disponibles."
        return

    # 2. Construcción del contexto y fuentes (igual que en tu script)
    contexto = ""
    fuentes = set()
    for doc in retrieved_docs:
        contexto += f"--- Fragmento de: {doc.metadata.get('source', 'N/A')} ---\n"
        contexto += f"{doc.page_content}\n\n"
        fuentes.add(doc.metadata.get('source', 'N/A'))
    
    fuentes_str = "\n".join(f"- {f}" for f in fuentes)

    # 3. Construcción del prompt (igual que en tu script)
    prompt_template = f"""
    Eres un asistente de IA especializado en la legislación y normatividad de la UNAM. Actúa con máxima precisión.
    INSTRUCCIONES:
    1. Tu única fuente de verdad es el CONTEXTO proporcionado. NO uses conocimiento externo.
    2. Responde a la PREGUNTA DEL USUARIO basándote exclusivamente en el CONTEXTO.
    3. Si la respuesta no está en el CONTEXTO, responde: "La información solicitada no se encuentra en los documentos disponibles."
    4. NO cites las fuentes en el cuerpo de la respuesta. La UI las mostrará por separado.
    --- CONTEXTO ---
    {contexto}
    --- FIN DEL CONTEXTO ---
    PREGUNTA DEL USUARIO: "{query}"
    Respuesta:
    """

    # 4. Llamada a la API y 'yield' de los fragmentos
    try:
        stream_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Sigue las instrucciones del prompt del usuario."},
                {"role": "user", "content": prompt_template}
            ],
            stream=True,
            max_tokens=1500,
            temperature=0.1
        )
        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                
    except Exception as e:
        yield f"Ocurrió un error al contactar la API de DeepSeek: {e}"

    # Al final, guardamos las fuentes en el estado de la sesión para mostrarlas fuera del stream.
    st.session_state.fuentes = fuentes_str

# --- 3. INTERFAZ DE USUARIO ---

# Inicializar el historial de chat en el estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fuentes" not in st.session_state:
    st.session_state.fuentes = ""

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceptar la entrada del usuario
if prompt := st.chat_input("¿Qué deseas saber sobre la legislación de la Facultad de Ciencias?"):
    # Añadir el mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        # Usamos st.write_stream para mostrar la respuesta en tiempo real
        response_placeholder = st.empty()
        full_response = response_placeholder.write_stream(generar_respuesta_stream(prompt, db_instance, client_instance))
        
        # Una vez que la respuesta está completa, mostramos las fuentes si existen.
        if st.session_state.fuentes:
            with st.expander("Fuentes Consultadas"):
                st.markdown(st.session_state.fuentes)
            st.session_state.fuentes = "" # Limpiar para la siguiente pregunta

    # Añadir la respuesta completa del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": full_response})