import os # Para interactuar con el sistema operativo
import time # Para funciones relacionadas con el tiempo
import requests # Para hacer peticiones HTTP
import streamlit as st # Para crear la interfaz de usuario web
from chromadb import PersistentClient  # Para la base de datos vectorial persistente

from langchain_community.document_loaders import PDFPlumberLoader # Para cargar documentos PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # Para dividir texto en fragmentos
from langchain_chroma import Chroma # Para la base de datos vectorial
from langchain_ollama import OllamaEmbeddings # Para generar embeddings usando Ollama
from langchain_core.prompts import ChatPromptTemplate # Para crear prompts
from langchain_ollama.llms import OllamaLLM # Para usar modelos de lenguaje de Ollama

# =====================================
# Configuración de la aplicación
# =====================================
st.set_page_config( # Configura la página de Streamlit
    page_title="Chatbot PDF: Ollama y Deepseek",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define colores de la interfaz de usuario
PRIMARY_COLOR = "#007BFF" # Azul primario
PRIMARY_COLOR_DARK = "#0056b3" # Azul primario oscuro para hover
SECONDARY_COLOR = "#6C757D" # Gris secundario
BACKGROUND_COLOR = "#F8F9FA" # Fondo claro
CARD_BACKGROUND_COLOR = "#FFFFFF" # Fondo blanco para tarjetas
TEXT_COLOR = "#343A40" # Texto gris oscuro
ACCENT_COLOR = "#28A745" # Acento verde
ERROR_COLOR = "#DC3545" # Error rojo
LIGHT_GRAY = "#E0E0E0" # Gris claro para bordes

# Aplica tema personalizado usando Markdown y CSS inyectado
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
MODEL_NAME = "deepseek-r1:8b" # Modelo de lenguaje a usar
PDF_DIR = 'pdfs/' # Directorio para PDFs subidos
DB_DIR = 'vectordb/' # Directorio para la base de datos vectorial
OLLAMA_HOST = "http://localhost:11434" # URL de la API de Ollama
RESET_VECTOR_STORE = False  # Poner en True para reiniciar la base de datos vectorial

# Crea directorios si no existen
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# =====================================
# Funciones principales
# =====================================
def check_ollama_connection(): # Verifica la conexión a Ollama
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

@st.cache_resource(show_spinner=False, hash_funcs={Chroma: id}) # Cachea el recurso para rendimiento
def initialize_components(): # Inicializa los componentes de la aplicación
    check_ollama_connection()
    embeddings = OllamaEmbeddings(model=MODEL_NAME) # Inicializa embeddings
    model = OllamaLLM(model=MODEL_NAME) # Inicializa el modelo de lenguaje

    # Reinicia la base de datos vectorial si es necesario
    if RESET_VECTOR_STORE and os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)

    # Inicializa ChromaDB
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

@st.cache_data(show_spinner=False) # Cachea los datos para rendimiento
def process_pdf_data(file_path): # Procesa los datos del PDF
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def process_pdf(file, vector_store): # Procesa el archivo PDF y lo añade al vector store
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

def generate_response(query, vector_store, model): # Genera respuesta usando vector store y modelo
    for attempt in range(3): # Lógica de reintento para robustez
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
        except requests.exceptions.ConnectionError: # Maneja errores de conexión
            time.sleep(2 ** attempt)
        except Exception as e:
            st.error(f"Error generando respuesta: {str(e)}")
            return ""
    return ""

# =====================================
# Interfaz de usuario
# =====================================
embeddings, model, vector_store = initialize_components() # Inicializa componentes

st.title("Chatbot PDF: Ollama y Deepseek 📚")
st.caption("Conversación inteligente con documentos PDF usando IA")

with st.sidebar: # Configuración de la barra lateral
    st.title("⚙️ Configuración")
    uploaded_file = st.file_uploader("Subir PDF", type="pdf", help="Máximo 50MB")

    if uploaded_file: # Procesa el PDF subido
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("El archivo excede el límite de 50MB")
        elif process_pdf(uploaded_file, vector_store):
            st.success("PDF procesado exitosamente")

    st.divider()

    with st.expander("📊 Estadísticas en tiempo real"): # Expansor de estadísticas en tiempo real
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

if "messages" not in st.session_state: # Inicializa los mensajes del chat en el estado de sesión
    st.session_state.messages = []

for message in st.session_state.messages: # Muestra los mensajes del chat
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz una pregunta sobre el documento..."): # Entrada del chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"): # Muestra el mensaje del usuario
        st.markdown(prompt)

    with st.chat_message("assistant"): # Muestra la respuesta del asistente
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
