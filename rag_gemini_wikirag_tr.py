import os, pathlib
from typing import List
from dataclasses import dataclass

import datasets, google.generativeai as genai
from langchain.docstore.document import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

# API anahtarı ortam değişkeninden alınır
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY ortam değişkenini ayarla.")
genai.configure(api_key=GOOGLE_API_KEY)

# Kullanılacak sohbet motorunu seçer
def choose_model() -> str:
    # Tercih edilen modeller listesi (öncelik sırasına göre)
    prefer = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    # Erişilebilen ve uygun olan modelleri filtreler
    allowed = [
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
        and not any(bad in m.name.lower() for bad in ("vision", "translate", "deprecated"))
    ]
    # Tercih edilen modellerden ilk erişilebileni döner
    for p in prefer:
        if p in allowed:
            return p
    if not allowed:
        raise RuntimeError("Hiç sohbet modeline erişimin yok.")
    return allowed[0]

# Kullanılacak model adı (ortam değişkeninden ya da otomatik seçim)
LLM_NAME = os.getenv("GOOGLE_LLM", choose_model())

# Kullanılacak embedding (anlam vektörü) modeli
EMB_NAME = "models/embedding-001"

# FAISS dizin klasörü
INDEX_DIR = pathlib.Path("faiss_index")

# Aramada döndürülecek belge sayısı
K = 5

# Sistem mesajı: sohbet motorunun davranış kuralları
SYSTEM_PROMPT = """
Sen Türkçe konuşan, bağlam tabanlı bir asistansın. Kesin kurallar:
1) Kullanıcı mesajı yalnızca selam niteliğindeyse (“selam”, “merhaba”, “hey”,
   “sa”, “selâm” vb.) bağlama bakma. Sadece kısa bir selam ver ve
   “Size nasıl yardımcı olabilirim?” diye sor.
2) Diğer tüm girdilerde:
   • Yalnızca AŞAĞIDAKİ BAĞLAM'I kullanarak cevap ver.
   • Yanıt tek cümle, doğrudan olsun.
   • Eğer bağlamda cevap yoksa sadece “Bilmiyorum” yaz.
   • Gereksiz açıklama ya da mazeret ekleme. """
   
# Kullanıcı girdisi ve bağlam şablonu
HUMAN_PROMPT = """
BAĞLAM:
{context}

Soru: {question}
Cevap:
"""

# Sohbet motoruna gönderilecek mesaj şablonu
PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT.strip()), ("human", HUMAN_PROMPT.strip())]
)

# Wikipedia veri kümesinden alınan her bir belge parçasını temsil eder
@dataclass
class WikiChunk:
    id: str
    text: str

# FAISS dizinini oluşturur ya da yüklü dizini getirir
def build_or_load_index() -> FAISS:
    if INDEX_DIR.exists():
        # Eğer dizin daha önce oluşturulmuşsa yerelden yükle
        return FAISS.load_local(
            str(INDEX_DIR),
            GoogleGenerativeAIEmbeddings(model=EMB_NAME),
            allow_dangerous_deserialization=True
        )

    print("[i] Dizini oluşturuyor… (ilk sefer)")
    # Veri kümesini yükle
    ds = datasets.load_dataset("Metin/WikiRAG-TR", split="train")
    # Belgeleri hazırla (her biri bir Document nesnesi olur)
    docs: List[Document] = [
        Document(page_content=row["context"], metadata={"source": row["id"]})
        for row in ds
    ]
    # FAISS dizinini belgelerden oluştur
    vs = FAISS.from_documents(
        docs,
        GoogleGenerativeAIEmbeddings(model=EMB_NAME)
    )
    # Oluşturulan dizini kaydet
    vs.save_local(str(INDEX_DIR))
    return vs

# Soru-cevap zincirini kurar
def build_chain():
    # Sohbet motorunu başlat
    llm = ChatGoogleGenerativeAI(
        model=LLM_NAME,
        temperature=0.0,  # Yanıtları tutarlı ve deterministik hale getirir.İhtiyacınıza göre değişebilir
        convert_system_message_to_human=True,
    )
    # Vektör arama motorunu hazırla
    retriever = build_or_load_index().as_retriever(search_kwargs={"k": K})
    # Konuşma geçmişini tutacak hafıza nesnesi
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    # Soru, arama ve yanıt üretimini birleştiren zincir
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,  # Hangi belgelerin kullanıldığını da döner
    )
    print(f"[i] Kullanılan sohbet modeli: {LLM_NAME}")
    return chain

# Üretilen cevabı ve kullanılan kaynak parçaları ekrana yazdırır
def pretty_print(res: dict):
    print(f"\n[Cevap]\n{res['answer']}\n")
    print("[Kaynak Parçalar]")
    for i, doc in enumerate(res["source_documents"], 1):
        # Belge parçasının ilk 160 karakterini yazdır
        snippet = doc.page_content[:160].replace("\n", " ")
        if len(doc.page_content) > 160:
            snippet += "…"
        print(f"{i}. {snippet}")

# Etkileşimli kullanıcı döngüsü: soruları alır ve cevabı yazdırır
def interactive_loop():
    qa_chain = build_chain()
    try:
        while True:
            q = input("\n>>> ").strip()
            if not q:
                break
            result = qa_chain.invoke({"question": q})
            pretty_print(result)
    except (KeyboardInterrupt, EOFError):
        print("\n\n[+] Görüşmek üzere!")

# Tek bir soruyu cevaplar 
def answer_question(question: str):
    qa_chain = build_chain()
    result = qa_chain.invoke({"question": question})
    return result

# Program doğrudan çalıştırıldığında etkileşimli mod başlatılır
if __name__ == "__main__":
    interactive_loop()
