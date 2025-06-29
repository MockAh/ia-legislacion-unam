# ui_app.py
# VERSIÓN 3.0 - SIMPLIFICADA USANDO RAG_ENGINE

import streamlit as st

# Importamos la instancia única del motor RAG desde nuestro backend
try:
    from rag_app_deepseek import rag_engine
except ImportError as e:
    st.error(f"Error: No se pudo encontrar el archivo 'rag_app_deepseek.py' o su contenido es incorrecto. Detalle: {e}")
    st.stop()


# --- CONFIGURACIÓN DE LA PÁGINA Y CARGA DE RECURSOS ---
st.set_page_config(page_title="Asistente de Legislación", page_icon="🤖", layout="wide")

st.title("🤖 Asistente de Legislación")
st.markdown("""
*UNAM (Facultad de Ciencias) / Red en Defensa de los Derechos Digitales (R3D)*
\n*Potenciado por Búsqueda Híbrida y LLMs.*
""")

# La carga de recursos ahora es manejada internamente por la clase RAG_Engine
# y se activa en la primera consulta. No necesitamos @st.cache_resource aquí.

# --- INTERFAZ DE USUARIO ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Crear columnas para los controles
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
        # Mostrar el mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.info(f"Buscando en: **{dominio_seleccionado}**")
            full_response = ""
            fuentes_encontradas = set()
            
            # Usamos un placeholder para el streaming de la respuesta
            response_placeholder = st.empty()
            
            # La llamada al backend ahora es una única función limpia
            for chunk in rag_engine.answer_question_stream(prompt, dominio_seleccionado):
                if isinstance(chunk, dict) and 'fuentes' in chunk:
                    # Este es el diccionario de fuentes que enviamos al final
                    fuentes_encontradas = chunk['fuentes']
                else:
                    # Esto es un trozo de texto de la respuesta
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌") # Muestra el cursor parpadeante
            
            response_placeholder.markdown(full_response) # Respuesta final
            
            # Mostrar las fuentes encontradas
            if fuentes_encontradas:
                with st.expander("Fuentes Consultadas"):
                    st.markdown("\n".join(f"- {f}" for f in fuentes_encontradas))
        
        # Guardar la respuesta completa en el historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})