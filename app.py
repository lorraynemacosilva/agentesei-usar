import streamlit as st
from google import genai
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuração
st.set_page_config(page_title="Agente SEI", page_icon="📑")

st.title("📑 Agente SEI - Usuário")
st.write("Converse com o assistente treinado em materiais do SEI.")

# Inicializa Gemini
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- Carregar base local (PDFs) ---
@st.cache_resource
def carregar_base():
    documentos = []
    folder = "data"
    for pdf in os.listdir(folder):
        if pdf.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, pdf))
            documentos.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    return db

db = carregar_base()

# --- Interface ---
pergunta = st.text_input("Digite sua dúvida sobre o SEI:")

if pergunta:
    docs = db.similarity_search(pergunta, k=3)
    contexto = "\n\n".join([d.page_content for d in docs])

# --- Prompt do Agente SEI ---
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

Exemplos de como agir:
- Se perguntarem: “Como faço meu primeiro acesso ao SEI?” → você responde com o passo a passo do login inicial.
- Se perguntarem: “Quem libera meu acesso?” → você explica que o acesso é solicitado pela secretaria do usuário e liberado pelo setor responsável.
- Se perguntarem algo fora do escopo (ex.: férias, salário): → você responde: “Essa dúvida não é sobre o SEI. Recomendo procurar o setor de RH da Prefeitura.”

Tom de voz:
Próximo, colaborativo, paciente.
Sempre encoraja os servidores: “Você consegue fazer isso em poucos passos, vou te mostrar como…”.
"""

# --- Construção dinâmica do prompt ---
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

    resposta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.markdown("### 📌 Resposta do Agente SEI:")
    st.write(resposta.text)

    with st.expander("🔎 Fontes consultadas"):
        for d in docs:
            st.write(f"- {d.metadata.get('source')} (página {d.metadata.get('page', '?')})")
