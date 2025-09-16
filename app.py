import streamlit as st
import google.generativeai as genai
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ----------------------------
# Configuração da página
# ----------------------------
st.set_page_config(
    page_title="Agente SEI",
    page_icon="📑",
    layout="wide"
)

# Paleta de cores personalizada via CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fafb;
    }
    h1, h2, h3 {
        color: #004080; /* Azul escuro institucional */
    }
    .stButton button {
        background-color: #004080;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Cabeçalho com logos
# ----------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("assets/logo_prefeitura.png", width=100)  # sua logo prefeitura
with col2:
    st.title("📑 Agente SEI - Usuário")
    st.write("Converse com o assistente treinado em materiais do SEI.")
with col3:
    st.image("assets/logo_sei.png", width=100)  # logo SEI

# ----------------------------
# Inicializa Gemini
# ----------------------------
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ----------------------------
# Carregar base local (PDFs)
# ----------------------------
@st.cache_resource
def carregar_base():
    documentos = []
    folder = "data"
    for pdf in os.listdir(folder):
        if pdf.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, pdf))
            documentos.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
    chunks = splitter.split_documents(documentos)
    
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
    db = FAISS.from_documents(chunks, embeddings)
    return db

db = carregar_base()

# ----------------------------
# Prompt do Agente SEI
# ----------------------------
prompt_sistema = """
Você é o Agente SEI, assistente virtual da Prefeitura de Pedro Leopoldo.

Seu papel é ajudar servidores municipais a compreender e utilizar o Sistema Eletrônico de Informações (SEI 4.0) durante a implantação e no dia a dia.

Você é bem-informado, claro e didático, dá respostas mais curtas mas ao mesmo tempo muito esclarecedoras.

Usa linguagem simples, amigável e prática, sem jargões técnicos.

Sempre que possível, apresenta a resposta em passo a passo.

Quando houver materiais de apoio (manuais, decretos, cronogramas, tutoriais), você deve orientar o servidor a consultá-los e indicar os links/documentos.

Caso a pergunta não esteja na sua base de conhecimento, você deve:
- Responder de forma geral.
- Orientar o usuário a procurar a equipe de suporte do SEI local.

Tom de voz:
Próximo, colaborativo, paciente.
Sempre encoraja os servidores: “Você consegue fazer isso em poucos passos, vou te mostrar como…”.
"""

# ----------------------------
# Interface
# ----------------------------
pergunta = st.text_input("Digite sua dúvida sobre o SEI:")

if pergunta:
    docs = db.similarity_search(pergunta, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    {prompt_sistema}

    Contexto (materiais de apoio):
    {contexto}

    Pergunta do servidor:
    {pergunta}
    """

    resposta = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)

    if resposta.candidates:
        resposta_texto = resposta.candidates[0].content.parts[0].text
    else:
        resposta_texto = "❌ Não foi possível gerar resposta."

    st.markdown("### 📌 Resposta do Agente SEI:")
    st.write(resposta_texto)

    with st.expander("🔎 Fontes consultadas"):
        for d in docs:
            st.write(f"- {d.metadata.get('source')} (página {d.metadata.get('page', '?')})")

