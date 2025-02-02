import os
import time
import requests
import streamlit as st
from chromadb import PersistentClient  # Nueva importación para el cliente persistente

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# =====================================
# Configuración de la aplicación
# =====================================
st.set_page_config(
    page_title="Chatbot PDF: Ollama y Deepseek",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de colores
PRIMARY_COLOR = "#007BFF"
PRIMARY_COLOR_DARK = "#0056b3"
SECONDARY_COLOR = "#6C757D"
BACKGROUND_COLOR = "#F8F9FA"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#343A40"
ACCENT_COLOR = "#28A745"
ERROR_COLOR = "#DC3545"
LIGHT_GRAY = "#E0E0E0"

# Aplicar tema personalizado
st.markdown(f"""
    <style>
        body {{ color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; }}
        .stApp {{ background-color: {BACKGROUND_COLOR}; }}
        [data-testid="stSidebar"] {{
            background-color: {CARD_BACKGROUND_COLOR};
            border-right: 1px solid {LIGHT_GRAY};
        }}
        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 0.5rem;
            transition: background-color 0.3s ease;
        }}
        .stButton > button:hover {{ background-color: {PRIMARY_COLOR_DARK}; }}
        .stChatMessage {{
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}
        .stChatMessage.st-chat-message--user {{
            background-color: #E9F1F7;
            border-left: 3px solid {PRIMARY_COLOR};
        }}
        .stChatMessage.st-chat-message--assistant {{
            background-color: {CARD_BACKGROUND_COLOR};
            border-right: 3px solid {ACCENT_COLOR};
        }}
        .stChatInput textarea {{
            background-color: {CARD_BACKGROUND_COLOR} !important;
            border: 1px solid {LIGHT_GRAY};
            border-radius: 0.5rem;
        }}
        h1, h2, h3 {{ color: {TEXT_COLOR}; font-weight: 700; }}
        .streamlit-metric-container {{
            background-color: {CARD_BACKGROUND_COLOR};
            border: 1px solid {LIGHT_GRAY};
        }}
        .stFileUploader {{ border: 2px dashed {LIGHT_GRAY}; }}
    </style>
""", unsafe_allow_html=True)

# =====================================
# Parámetros de configuración
# =====================================
MODEL_NAME = "deepseek-r1:8b"
PDF_DIR = 'pdfs/'
DB_DIR = 'vectordb/'
OLLAMA_HOST = "http://localhost:11434"
RESET_VECTOR_STORE = False  # Cambiar a True si necesitas reiniciar la base de datos

# Crear directorios necesarios
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# =====================================
# Funciones principales (actualizadas)
# =====================================
def check_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_HOST}/")
        response.raise_for_status()
        models = requests.get(f"{OLLAMA_HOST}/api/tags").json()
        installed_models = [m['name'] for m in models.get('models', [])]
        if MODEL_NAME not in installed_models:
            raise ValueError(f"Ejecuta en terminal: ollama pull {MODEL_NAME}")
        return True
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False, hash_funcs={Chroma: id})
def initialize_components():
    check_ollama_connection()
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    model = OllamaLLM(model=MODEL_NAME)
    
    # Reiniciar la base de datos si es necesario
    if RESET_VECTOR_STORE and os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
    
    # Inicializar ChromaDB utilizando la nueva API
    try:
        client = PersistentClient(path=DB_DIR)
        vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="pdf_collection"
        )
    except Exception as e:
        st.error(f"Error inicializando ChromaDB: {str(e)}")
        st.stop()
    
    return embeddings, model, vector_store

@st.cache_data(show_spinner=False)
def process_pdf_data(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def process_pdf(file, vector_store):
    try:
        file_path = os.path.join(PDF_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        chunks = process_pdf_data(file_path)
        vector_store.add_documents(chunks)
        return True
    except Exception as e:
        st.error(f"Error procesando PDF: {str(e)}")
        return False

def generate_response(query, vector_store, model):
    for attempt in range(3):
        try:
            docs = vector_store.similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in docs])
            template = """Eres un asistente experto en documentos. Basado en este contexto:
            {context}
            
            Responde en español de forma precisa y profesional:
            {query}"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | model
            return chain.invoke({"context": context, "query": query})
        except requests.exceptions.ConnectionError:
            time.sleep(2 ** attempt)
        except Exception as e:
            st.error(f"Error generando respuesta: {str(e)}")
            return ""
    return ""

# =====================================
# Interfaz de usuario (corregida)
# =====================================
embeddings, model, vector_store = initialize_components()

st.title("Chatbot PDF: Ollama y Deepseek 📚")
st.caption("Conversación inteligente con documentos PDF usando IA")

with st.sidebar:
    st.title("⚙️ Configuración")
    uploaded_file = st.file_uploader("Subir PDF", type="pdf", help="Máximo 50MB")
    
    if uploaded_file:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("El archivo excede el límite de 50MB")
        elif process_pdf(uploaded_file, vector_store):
            st.success("PDF procesado exitosamente")
    
    st.divider()
    
    with st.expander("📊 Estadísticas en tiempo real"):
        data = vector_store.get()
        if data is None:
            st.info("No hay datos en la base de vectores.")
        else:
            documents = data.get('documents') or []
            embeddings_data = data.get('embeddings') or []
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documentos cargados", len(documents))
            with col2:
                st.metric("Fragmentos indexados", len(embeddings_data))
            
            if not documents:
                st.warning("No hay documentos cargados")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = generate_response(prompt, vector_store, model)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Pie de página
st.divider()
st.markdown(
    f"""<div style="text-align: center; color: {SECONDARY_COLOR}; padding: 1rem;">
    Sistema de análisis documental v1.0 | 
    <a href="https://github.com/RickyFer22" style="color: {SECONDARY_COLOR};">GitHub</a> | 
    <a href="https://www.linkedin.com/in/ricardo-fernández00/" style="color: {SECONDARY_COLOR};">LinkedIn</a>
    </div>""",
    unsafe_allow_html=True
)
