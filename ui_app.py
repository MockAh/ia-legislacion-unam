# ui_app.py
# VERSIÓN 2.0 - CON FILTROS DE DOMINIO INTERACTIVOS

import streamlit as st
import time

# Asumimos que el archivo rag_app_deepseek.py está en la misma carpeta
# y contiene las funciones `inicializar_sistema` y la variable `DEEPSEEK_MODEL_NAME`.
try:
    from rag_app_deepseek import inicializar_sistema, DEEPSEEK_MODEL_NAME
except ImportError:
    st.error("Error: No se pudo encontrar el archivo 'rag_app_deepseek.py'. Asegúrate de que esté en el mismo directorio.")
    # Funciones y variables de respaldo para que la app no se rompa si falta el archivo
    def inicializar_sistema(): return None, None
    DEEPSEEK_MODEL_NAME = "deepseek-chat"
    st.stop()


# --- 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DEL SISTEMA ---

st.set_page_config(
    page_title="Asistente de Legislación Mexicana",
    page_icon="🤖",
    layout="wide"
)

# Título y subtítulo actualizados según tu solicitud
st.title("🤖 Asistente de Legislación Mexicana")
st.markdown("""
Legislación de la UNAM para la Facultad de Ciencias / Red en Defensa de los Derechos Digitales (R3D)
\n*Potenciado por DeepSeek (`deepseek-chat`) y una base de datos local.*
""")

@st.cache_resource
def cargar_recursos():
    """Carga la base de datos y el cliente de API. Muestra un spinner durante la carga."""
    with st.spinner("Iniciando sistema por primera vez... Cargando base de datos y modelos. Esto puede tardar unos segundos."):
        db, client = inicializar_sistema()
    return db, client

db_instance, client_instance = cargar_recursos()

if not db_instance or not client_instance:
    st.error("El sistema no pudo inicializarse. Verifica que el índice FAISS ('procesado/') exista y que la API de DeepSeek esté configurada.", icon="🚨")
    st.stop()


# --- 2. LÓGICA DE GENERACIÓN DE RESPUESTA MODIFICADA ---

def generar_respuesta_stream(query, db, client, filtro_dominio):
    """
    Genera una respuesta usando RAG, aplicando un filtro de dominio si se especifica.
    """
    # 1. Construcción del filtro para la búsqueda
    search_filter = {}
    if filtro_dominio == "Facultad de Ciencias":
        # Filtra por chunks cuya metadata tenga 'entidad_unam' igual a 'fciencias'
        search_filter = {"entidad_unam": "fciencias"}
    elif filtro_dominio == "R3D (Derechos Digitales)":
        # Filtra por chunks cuyo dominio sea 'r3d'
        search_filter = {"dominio": "r3d"}
        
    # 2. Búsqueda de similitud con el filtro aplicado
    try:
        retrieved_docs = db.similarity_search(query, k=5, filter=search_filter)
    except Exception as e:
        yield f"Ocurrió un error al buscar en la base de datos: {e}"
        return
    
    if not retrieved_docs:
        yield f"La información solicitada no se encuentra en los documentos del dominio '{filtro_dominio}'."
        return

    # 3. Construcción del contexto y fuentes
    contexto = ""
    fuentes = set()
    for doc in retrieved_docs:
        contexto += f"--- Fragmento de: {doc.metadata.get('source', 'N/A')} ---\n"
        contexto += f"{doc.page_content}\n\n"
        fuentes.add(doc.metadata.get('source', 'N/A'))
    
    fuentes_str = "\n".join(f"- {f}" for f in fuentes)

    # 4. Construcción del prompt
    prompt_template = f"""
    Eres un asistente de IA especializado. Actúa con máxima precisión.
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

    # 5. Llamada a la API de DeepSeek
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

    st.session_state.fuentes = fuentes_str

# --- 3. INTERFAZ DE USUARIO CON SELECTOR DE DOMINIO ---

# Inicializar el historial de chat y las fuentes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fuentes" not in st.session_state:
    st.session_state.fuentes = ""

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Elementos de la UI: Selector de Dominio y Chat Input ---

# Crear dos columnas para organizar la UI
col1, col2 = st.columns([1, 3])

with col1:
    # Widget de radio para seleccionar el dominio de búsqueda
    dominio_seleccionado = st.radio(
        "**Ámbito de Búsqueda:**",
        ("Búsqueda General", "Facultad de Ciencias", "R3D (Derechos Digitales)"),
        index=0, # Opción por defecto
        help="Selecciona un ámbito para restringir la búsqueda a un conjunto específico de documentos."
    )

with col2:
    # Aceptar la entrada del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Añadir el mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.info(f"Buscando en: **{dominio_seleccionado}**")
            # Usamos st.write_stream para mostrar la respuesta en tiempo real
            response_placeholder = st.empty()
            
            # Pasamos la selección del usuario a la función de generación
            full_response = response_placeholder.write_stream(
                generar_respuesta_stream(prompt, db_instance, client_instance, dominio_seleccionado)
            )
            
            # Mostrar las fuentes si existen
            if st.session_state.fuentes:
                with st.expander("Fuentes Consultadas"):
                    st.markdown(st.session_state.fuentes)
                st.session_state.fuentes = ""

        # Añadir la respuesta completa del asistente al historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})

