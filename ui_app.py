# ui_app.py
# VERSIÓN 5.0 - CON CARGA DE RECURSOS EN CACHÉ PARA DESPLIEGUE

import streamlit as st

# Importamos la instancia única del motor RAG desde nuestro backend
try:
    from rag_app_deepseek import rag_engine
except ImportError as e:
    st.error(f"Error: No se pudo encontrar el archivo 'rag_app_deepseek.py' o su contenido es incorrecto. Detalle: {e}")
    st.stop()


# --- 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DE RECURSOS ---

st.set_page_config(
    page_title="Asistente de Legislación",
    page_icon="🤖",
    layout="wide"
)

# --- NUEVA FUNCIÓN DE CARGA CON CACHÉ ---
@st.cache_resource
def cargar_motor_rag():
    """
    Esta función carga la instancia del motor RAG y ejecuta la carga de componentes pesados.
    El decorador @st.cache_resource asegura que esto se ejecute UNA SOLA VEZ
    y el resultado se guarde en caché para toda la sesión de la app.
    """
    with st.spinner("Iniciando sistema por primera vez... Este proceso puede tardar varios minutos."):
        # Forzamos la carga de todos los modelos y bases de datos aquí
        rag_engine._load_components()
    return rag_engine

# --- LLAMADA INICIAL AL MOTOR ---
# Esto ejecutará la carga la primera vez que se visite la página
# y en las siguientes visitas, devolverá instantáneamente el objeto en caché.
rag_engine_cargado = cargar_motor_rag()

st.title("🤖 Asistente de Legislación")
st.markdown("""
*UNAM (Facultad de Ciencias) / Red en Defensa de los Derechos Digitales (R3D)*
\n*Potenciado por Búsqueda Híbrida (`e5-large`) y LLMs.*
""")


# --- 2. INTERFAZ DE USUARIO (CHAT) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2 = st.columns([1, 3])

with col1:
    dominio_seleccionado = st.radio(
        "**Ámbito de Búsqueda:**",
        ("Búsqueda General", "Facultad de Ciencias", "R3D (Derechos Digitales)"),
        index=0,
        help="Restringe la búsqueda a un conjunto específico de documentos."
    )

with col2:
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.info(f"Buscando en: **{dominio_seleccionado}**")
            
            full_response = ""
            fuentes_encontradas = set()
            response_placeholder = st.empty()
            
            # La llamada al backend ahora usa el motor cargado en caché
            for chunk in rag_engine_cargado.answer_question_stream(prompt, dominio_seleccionado):
                if isinstance(chunk, dict) and 'fuentes' in chunk:
                    fuentes_encontradas = chunk['fuentes']
                    break 
                else:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            if fuentes_encontradas:
                with st.expander("Fuentes Consultadas"):
                    st.markdown("\n".join(f"- {f}" for f in fuentes_encontradas))
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})