# ui_app.py
# VERSIÓN 4.0 - FINAL, SINCRONIZADA CON RAG_ENGINE AVANZADO

import streamlit as st
import time

# Importamos la instancia única del motor RAG desde nuestro backend
try:
    from rag_app_deepseek import rag_engine
except ImportError as e:
    st.error(f"Error: No se pudo encontrar el archivo 'rag_app_deepseek.py' o su contenido es incorrecto. Detalle: {e}")
    st.stop()


# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Asistente de Legislación",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Asistente de Legislación")
st.markdown("""
*UNAM (Facultad de Ciencias) / Red en Defensa de los Derechos Digitales (R3D)*
\n*Potenciado por Búsqueda Híbrida y LLMs.*
""")

# --- 2. INTERFAZ DE USUARIO (CHAT) ---

# Inicializar el historial de chat en el estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes del historial en cada recarga de la página
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Crear columnas para organizar los controles de la UI
col1, col2 = st.columns([1, 3])

with col1:
    # Widget para seleccionar el ámbito de búsqueda
    dominio_seleccionado = st.radio(
        "**Ámbito de Búsqueda:**",
        ("Búsqueda General", "Facultad de Ciencias", "R3D (Derechos Digitales)"),
        index=0,
        help="Restringe la búsqueda a un conjunto específico de documentos."
    )

with col2:
    # Aceptar la entrada del usuario con el input de chat de Streamlit
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Añadir y mostrar el mensaje del usuario en la UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.info(f"Buscando en: **{dominio_seleccionado}**")
            
            # --- LÓGICA DE STREAMING CORREGIDA ---
            # Usamos un bucle for manual para manejar los diferentes tipos de datos del stream
            
            full_response = ""
            fuentes_encontradas = set()
            response_placeholder = st.empty() # Placeholder para actualizar la respuesta en tiempo real
            
            # La llamada al backend ahora es una única función limpia
            for chunk in rag_engine.answer_question_stream(prompt, dominio_seleccionado):
                # Verificamos si el chunk es el diccionario final de fuentes
                if isinstance(chunk, dict) and 'fuentes' in chunk:
                    fuentes_encontradas = chunk['fuentes']
                    # Salimos del bucle una vez que recibimos las fuentes
                    break 
                else:
                    # Si no es el diccionario, es un trozo de texto de la respuesta
                    full_response += chunk
                    # Actualizamos el placeholder con la respuesta acumulada y un cursor parpadeante
                    response_placeholder.markdown(full_response + "▌")
            
            # Escribimos la respuesta final sin el cursor
            response_placeholder.markdown(full_response)
            
            # Mostrar las fuentes encontradas si existen
            if fuentes_encontradas:
                with st.expander("Fuentes Consultadas"):
                    st.markdown("\n".join(f"- {f}" for f in fuentes_encontradas))
        
        # Guardar la respuesta completa en el historial para la sesión
        st.session_state.messages.append({"role": "assistant", "content": full_response})