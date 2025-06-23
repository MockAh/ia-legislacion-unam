# === SCRIPT DE PROCESAMIENTO Y GENERACIÓN DE EMBEDDINGS ===
# VERSIÓN 5.2: AÑADIDO DOMINIO 'r3d' Y SU LÓGICA DE PARSEO

import os
import re
import json
import time
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
import docx2txt
from tqdm import tqdm

# 1. --- CONFIGURACIÓN INICIAL ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DOCS_DIR = BASE_DIR / "Juegos"
OUTPUT_DIR = BASE_DIR / "procesado"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 2. --- FUNCIONES DE SOPORTE ---

def parse_filename_for_metadata(filename: str, dominio: str) -> dict:
    """
    Extrae metadatos de un nombre de archivo de forma adaptativa,
    dependiendo del dominio al que pertenece.
    """
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    metadata = {}

    if dominio == 'legislacion_unam':
        if len(parts) == 5:
            metadata["entidad_unam"] = parts[0].replace('-', ' ')
            metadata["area_interna"] = parts[1].replace('-', ' ')
            metadata["titulo"] = parts[2].replace('-', ' ')
            metadata["tipo_documento"] = parts[3].replace('-', ' ')
            metadata["año"] = parts[4]
        else:
            metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
            print(f"  (ADVERTENCIA: El archivo '{filename}' en 'legislacion_unam' no sigue el formato de 5 partes.)")

    elif dominio == 'teoria_politica':
        if len(parts) == 3:
            metadata["autor"] = parts[0].replace('-', ' ')
            metadata["titulo"] = parts[1].replace('-', ' ')
            metadata["año"] = parts[2]
        else:
            metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
            print(f"  (ADVERTENCIA: El archivo '{filename}' en 'teoria_politica' no sigue el formato de 3 partes.)")

    # --- NUEVA LÓGICA PARA EL DOMINIO 'r3d' ---
    elif dominio == 'r3d':
        # Formato esperado: r3d_Titulo-del-documento_año.pdf
        if len(parts) == 3:
            metadata["entidad"] = parts[0].upper() # Guardamos "R3D" en mayúsculas
            metadata["titulo"] = parts[1].replace('-', ' ')
            metadata["año"] = parts[2]
        else:
            metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
            print(f"  (ADVERTENCIA: El archivo '{filename}' en 'r3d' no sigue el formato de 3 partes.)")
            
    else: # Reglas genéricas para cualquier otro dominio futuro
        metadata["titulo"] = name_without_ext.replace('_', ' ').replace('-', ' ')
        metadata["autor"] = "no aplica"
        metadata["año"] = "s/f"

    return metadata


def convert_to_text(file_path):
    try:
        path_str = str(file_path)
        if path_str.endswith('.pdf'):
            try: return extract_text(path_str)
            except Exception:
                print(f"    (INFO: Usando OCR para {Path(path_str).name})")
                images = convert_from_path(path_str, dpi=200)
                return "".join(pytesseract.image_to_string(img, lang='spa') for img in images)
        elif path_str.endswith('.docx'): return docx2txt.process(path_str)
        elif path_str.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e:
        print(f"    (ERROR al procesar {Path(file_path).name}: {str(e)})")
        return ""

def clean_legal_text(text):
    text = re.sub(r'(Artículo\s*\d+[\.\:]?)', r'\n\n\1\n', text)
    # ... (puedes añadir más reglas de limpieza si es necesario)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# (El resto del script, la función process_documents, es idéntica y no necesita cambios)
def process_documents():
    print("--- Iniciando el Pipeline de Procesamiento de Documentos ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separators=[r"\n\nArtículo \d+\.", "\n\nArtículo", "\n\nSECCIÓN", "\n\nCAPÍTULO", "\n\n", " ", ""])
    db = None
    processed_files = set()
    all_metadata = []
    db_path_faiss = OUTPUT_DIR / "index.faiss"
    metadata_path = OUTPUT_DIR / "metadata.json"
    if db_path_faiss.exists() and metadata_path.exists():
        print(f"Base de datos existente encontrada en '{OUTPUT_DIR}'. Cargando...")
        db = FAISS.load_local(str(OUTPUT_DIR), embeddings, allow_dangerous_deserialization=True)
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
            processed_files = set(meta.get('source') for meta in all_metadata)
        print(f"Carga completa. {len(processed_files)} archivos previamente procesados.")
    else:
        print("No se encontró base de datos. Se creará una nueva.")
    all_files_in_dir = []
    if not INPUT_DOCS_DIR.exists():
        print(f"ERROR: El directorio de entrada '{INPUT_DOCS_DIR}' no existe. Por favor, créalo.")
        return
    for root, _, files in os.walk(INPUT_DOCS_DIR):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx', '.txt')):
                all_files_in_dir.append(Path(root) / filename)
    new_files_to_process = [f for f in all_files_in_dir if f.name not in processed_files]
    if not new_files_to_process:
        print("\nNo hay documentos nuevos para añadir. La base de datos está actualizada.")
        return
    print(f"\nSe encontraron {len(new_files_to_process)} documentos nuevos. Iniciando procesamiento...")
    new_chunks_with_metadata = []
    for file_path in tqdm(new_files_to_process, desc="Procesando documentos"):
        filename = file_path.name
        relative_path = file_path.relative_to(INPUT_DOCS_DIR)
        dominio_juego = relative_path.parts[0]
        raw_text = convert_to_text(file_path)
        if not raw_text: continue
        cleaned_text = clean_legal_text(raw_text)
        chunks = text_splitter.split_text(cleaned_text)
        base_metadata = parse_filename_for_metadata(filename, dominio_juego)
        base_metadata["dominio"] = dominio_juego
        base_metadata["source"] = filename
        for i, chunk_text in enumerate(chunks):
            article_match = re.search(r'Artículo (\d+)', chunk_text)
            chunk_metadata = base_metadata.copy()
            chunk_metadata["article"] = article_match.group(1) if article_match else None
            chunk_metadata["chunk_index"] = i
            new_chunks_with_metadata.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    if not new_chunks_with_metadata:
        print("\nNo se generaron nuevos chunks. No hay nada que añadir a la base de datos.")
        return
    print(f"\nGenerando embeddings para {len(new_chunks_with_metadata)} nuevos chunks... (esto puede tardar)")
    if db is None:
        db = FAISS.from_documents(new_chunks_with_metadata, embeddings)
        print("Creando nueva base de datos FAISS...")
    else:
        db.add_documents(new_chunks_with_metadata)
        print("Añadiendo nuevos documentos a la base de datos existente...")
    db.save_local(str(OUTPUT_DIR))
    print(f"\n✅ Base de datos FAISS guardada/actualizada exitosamente en: '{OUTPUT_DIR}'")
    new_metadata = [doc.metadata for doc in new_chunks_with_metadata]
    all_metadata.extend(new_metadata)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    print(f"✅ Archivo de metadatos actualizado en: '{metadata_path}'")

if __name__ == "__main__":
    start_time = time.time()
    process_documents()
    print(f"\n--- Proceso completado en {time.time() - start_time:.2f} segundos. ---")