# === SCRIPT DE PROCESAMIENTO Y GENERACIÓN DE EMBEDDINGS ===
# VERSIÓN 10.0: FUSIÓN INTELIGENTE DE CHUNKS EN AMBAS LÓGICAS Y METADATO 'ENUMERATION'

import os
import re
import json
import time
from pathlib import Path

# --- Dependencias de Terceros ---
import fitz  # PyMuPDF
import docx2txt
import pytesseract
from tqdm import tqdm
from pdf2image import convert_from_path
from langchain_huggingface import HuggingFaceEmbeddings # Importación actualizada
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document # Importación actualizada

# 1. --- CONFIGURACIÓN INICIAL ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DOCS_DIR = BASE_DIR / "Juegos"
OUTPUT_DIR = BASE_DIR / "procesado"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# --- CONFIGURACIÓN PARA DOCUMENTOS ESPECIALES ---
DOCUMENTOS_CON_MARGENES = {
    "fciencias_Titulacion_Licenciatura-en-Matematicas_Reglamento-Interno_2025.pdf": {'right': 0.20},
    "fciencias_Titulacion_Licenciatura-en-Matematicas-Aplicadas_Reglamento-Interno_2025.pdf": {'right': 0.20},
    "fciencias_Titulacion_Menciones-Honorificas_Normatividad_2024.pdf": {'right': 0.20},
}
ARCHIVOS_FORZAR_OCR = {}
ARCHIVOS_A_IGNORAR = {}


# 2. --- FUNCIONES DE PROCESAMIENTO DE DATOS ---

def parse_filename_for_metadata(filename: str, dominio: str) -> dict:
    """Extrae metadatos de un nombre de archivo de forma adaptativa."""
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    metadata = {}
    if dominio == 'legislacion_unam':
        if len(parts) == 5:
            metadata["entidad_unam"], metadata["area_interna"], metadata["titulo"], metadata["tipo_documento"], metadata["año"] = [p.replace('-', ' ') for p in parts]
        else:
            metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
            print(f"  (ADVERTENCIA: El archivo '{filename}' en 'legislacion_unam' no sigue el formato de 5 partes.)")
    else:
        metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
    return metadata

def robust_text_extraction_and_cleaning(file_path: Path, clip_margins: dict = None) -> str:
    """Función mejorada que extrae y limpia texto, aceptando reglas de márgenes."""
    margins = {'top': 0.1, 'bottom': 0.1, 'left': 0.0, 'right': 0.0}
    if clip_margins: margins.update(clip_margins)
    path_str = str(file_path)
    raw_text = ""
    try:
        if file_path.name in ARCHIVOS_FORZAR_OCR: raise Exception("Forzando OCR según configuración.")
        if path_str.endswith('.pdf'):
            with fitz.open(path_str) as doc:
                full_text_list = []
                for page in doc:
                    page_rect = page.rect
                    clip_area = fitz.Rect(page_rect.x0 * (1 + margins['left']), page_rect.y0 * (1 + margins['top']), page_rect.x1 * (1 - margins['right']), page_rect.y1 * (1 - margins['bottom']))
                    full_text_list.append(page.get_text("text", clip=clip_area, sort=True))
                raw_text = "".join(full_text_list)
        elif path_str.endswith('.docx'): raw_text = docx2txt.process(path_str)
        elif path_str.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f: raw_text = f.read()
    except Exception as e:
        if path_str.endswith('.pdf'):
            try:
                print(f"  (INFO: Falló Fitz o se forzó OCR para '{file_path.name}'. Usando OCR...)")
                images = convert_from_path(path_str, dpi=200)
                raw_text = "".join(pytesseract.image_to_string(img, lang='spa') for img in images)
            except Exception as ocr_e: print(f"  (ERROR: El proceso de OCR también falló para {file_path.name}: {ocr_e})"); return ""
        else: print(f"  (ERROR al procesar {file_path.name}: {str(e)})"); return ""
    text = raw_text.replace('-\n', '')
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def chunk_by_article(text: str) -> list[str]:
    """Chunker para documentos con estructura de 'Artículo', con lógica de fusión."""
    patron = r"(\bART[IÍ]CULO \d+[°º]?[.\-]?\s?)"
    parts = re.split(patron, text, flags=re.IGNORECASE)
    if len(parts) <= 1: return [text] if text.strip() else []
    
    initial_chunks = [parts[0].strip()] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        full_article = (parts[i] + parts[i+1]).strip()
        initial_chunks.append(full_article)
    
    if not initial_chunks: return []
    merged_chunks = [initial_chunks[0]]
    for i in range(1, len(initial_chunks)):
        current_chunk = initial_chunks[i]
        previous_chunk = merged_chunks[-1]
        current_match = re.search(r'ART[IÍ]CULO (\d+)', current_chunk, re.IGNORECASE)
        previous_match = re.search(r'ART[IÍ]CULO (\d+)', previous_chunk, re.IGNORECASE)
        if current_match and previous_match and current_match.group(1) == previous_match.group(1):
            merged_chunks[-1] = previous_chunk + "\n\n" + current_chunk
        else:
            merged_chunks.append(current_chunk)
    return merged_chunks

def chunk_normatividad_general(text: str) -> list[str]:
    """Chunker para documentos normativos, con lógica de fusión jerárquica."""
    patron_numerico = r"(\b\d+(\.\d+)*\.\s)"
    parts = re.split(patron_numerico, text)
    if len(parts) <= 1: return [chunk for chunk in text.split('\n\n') if chunk.strip()]
    
    initial_chunks = [parts[0].strip()] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        delimiter = parts[i]
        content = parts[i+1] if i + 1 < len(parts) else ""
        chunk_parts = [p for p in [delimiter, content] if p]
        if chunk_parts: initial_chunks.append("".join(chunk_parts).strip())
        
    if not initial_chunks: return []
    merged_chunks = [initial_chunks[0]]
    patron_extraccion = re.compile(r"^(\d+(\.\d+)*\.)")
    for i in range(1, len(initial_chunks)):
        current_chunk = initial_chunks[i]
        previous_chunk = merged_chunks[-1]
        current_match = patron_extraccion.match(current_chunk)
        previous_match = patron_extraccion.match(previous_chunk)
        if current_match and previous_match and current_match.group(0).startswith(previous_match.group(0)):
            merged_chunks[-1] = previous_chunk + "\n\n" + current_chunk
        else:
            merged_chunks.append(current_chunk)
    return merged_chunks

# 3. --- FUNCIÓN PRINCIPAL DEL PIPELINE ---
def process_documents():
    """Orquesta todo el proceso de ingesta de datos."""
    print("--- Iniciando el Pipeline de Procesamiento de Documentos ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    db, all_metadata, processed_files = None, [], set()
    db_path_faiss, metadata_path = OUTPUT_DIR / "index.faiss", OUTPUT_DIR / "metadata.json"
    if db_path_faiss.exists() and metadata_path.exists():
        print(f"Base de datos existente encontrada. Cargando...")
        try:
            db = FAISS.load_local(str(OUTPUT_DIR), embeddings, allow_dangerous_deserialization=True)
            with open(metadata_path, "r", encoding="utf-8") as f:
                all_metadata, processed_files = json.load(f), {meta.get('source') for meta in all_metadata}
            print(f"Carga completa. {len(processed_files)} archivos previamente procesados.")
        except Exception as e: print(f"ERROR al cargar la base de datos: {e}. Se creará una nueva."); db, all_metadata = None, []
    else: print("No se encontró base de datos. Se creará una nueva.")

    all_files_in_dir = [p for p in INPUT_DOCS_DIR.rglob('*') if p.is_file() and p.suffix.lower() in ['.pdf', '.docx', '.txt']]
    new_files_to_process = [f for f in all_files_in_dir if f.name not in processed_files and f.name not in ARCHIVOS_A_IGNORAR and not f.name.startswith('~$')]
    if not new_files_to_process: print("\nNo hay documentos nuevos."); return

    print(f"\nSe encontraron {len(new_files_to_process)} documentos nuevos. Procesando...")
    new_docs_to_embed = []
    for file_path in tqdm(new_files_to_process, desc="Procesando documentos"):
        filename = file_path.name
        dominio = file_path.relative_to(INPUT_DOCS_DIR).parts[0]
        custom_margins = DOCUMENTOS_CON_MARGENES.get(filename)
        processed_text = robust_text_extraction_and_cleaning(file_path, clip_margins=custom_margins)
        if not processed_text: continue

        chunks = []
        if dominio == 'legislacion_unam':
            if re.search(r'\bART[IÍ]CULO ', processed_text, re.IGNORECASE):
                print(f"  (INFO: Detectada estructura 'Artículo'. Usando chunk_by_article para {filename})")
                chunks = chunk_by_article(processed_text)
            else:
                print(f"  (INFO: No se detectó 'Artículo'. Usando chunker de normatividad general para {filename})")
                chunks = chunk_normatividad_general(processed_text)
        else: chunks = [chunk for chunk in processed_text.split('\n\n') if chunk.strip()]
        
        base_metadata = parse_filename_for_metadata(filename, dominio)
        base_metadata.update({"dominio": dominio, "source": filename})

        patron_enum = re.compile(r"^(\d+(\.\d+)*\.)")
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            if dominio == 'legislacion_unam':
                article_match = re.search(r'ART[IÍ]CULO (\d+)', chunk_text, re.IGNORECASE)
                enum_match = patron_enum.match(chunk_text)
                if article_match: chunk_metadata["article"] = article_match.group(1)
                elif enum_match: chunk_metadata["enumeration"] = enum_match.group(1).strip()
            chunk_metadata["chunk_index"] = i
            new_docs_to_embed.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    if not new_docs_to_embed: print("\nNo se generaron nuevos chunks."); return

    print(f"\nGenerando embeddings para {len(new_docs_to_embed)} nuevos chunks...")
    if db is None:
        db = FAISS.from_documents(new_docs_to_embed, embeddings)
        print("Creando nueva base de datos FAISS...")
    else:
        texts_to_add = [doc.page_content for doc in new_docs_to_embed]
        metadata_to_add = [doc.metadata for doc in new_docs_to_embed]
        db.add_texts(texts=texts_to_add, metadatas=metadata_to_add)
        print("Añadiendo nuevos documentos a la base de datos existente...")
    
    db.save_local(str(OUTPUT_DIR))
    print(f"\n✅ Base de datos FAISS guardada/actualizada en: '{OUTPUT_DIR}'")
    
    all_metadata.extend([doc.metadata for doc in new_docs_to_embed])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    print(f"✅ Archivo de metadatos actualizado en: '{metadata_path}'")

# 4. --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == "__main__":
    if not INPUT_DOCS_DIR.exists(): print(f"ERROR: Directorio de entrada '{INPUT_DOCS_DIR}' no existe.")
    else:
        start_time = time.time()
        process_documents()
        print(f"\n--- Proceso completado en {time.time() - start_time:.2f} segundos. ---")