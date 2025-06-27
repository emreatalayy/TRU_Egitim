
````markdown
# TRÜ Yapay Zeka ve Robotik Kulübü — Ünides Programı Eğitim Kodu

Bu proje, **Tokat Gaziosmanpaşa Üniversitesi Yapay Zeka ve Robotik Kulübü** tarafından **Ünides Programı** kapsamında geliştirilmiştir.  
Amaç, Türkçe belgeler üzerinde çalışan bir **Retrieval-Augmented Generation (RAG)** tabanlı soru-cevap sistemini kurmak ve öğretmektir.

## Kurulum ve Çalıştırma

Python 3.8+ ve şu kütüphaneler gereklidir:

```bash
pip install langchain langchain_community langchain_google_genai datasets faiss-cpu google-generativeai
````

Google Generative AI API anahtarınızı alıp tanımlayın:

```bash
export GOOGLE_API_KEY="senin_google_api_anahtarın"
export GOOGLE_LLM="gemini-1.5-pro-latest"  # Opsiyonel
```

👉 API anahtarını almak için: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

Proje dosyasının olduğu dizinde:

```bash
python dosya_adı.py
```

İlk çalıştırmada sistem `faiss_index/` dizinini oluşturur ve belgeleri ekler.
Sonraki çalıştırmalarda dizin doğrudan yüklenir.

Kullanım örneği:

```plaintext
>>> Türkiye'nin başkenti neresidir?
[Cevap]
Ankara.

[Kaynak Parçalar]
1. Türkiye'nin başkenti Ankara'dır ve 1923 yılında başkent ilan edilmiştir...
```

Çıkmak için: boş satır girin veya `Ctrl + C` / `Ctrl + D`.

## RAG (Retrieval-Augmented Generation)

RAG, bir yapay zekanın yalnızca ezber bilgisini değil, dış belgeleri arayıp bunlarla yanıt üretmesini sağlayan sistemdir.

### RAG Nasıl Çalışır?

1️⃣ Soru anlam vektörüne dönüştürülür.
2️⃣ FAISS dizininden en yakın belgeler çekilir.
3️⃣ Yanıt sadece bu belgelerle üretilir.
4️⃣ Bağlam yoksa “Bilmiyorum” yanıtı verilir.

### Avantajları

✅ Güncel ve doğru kaynaklardan yanıt verir.
✅ Eğitim verisinde olmayan sorulara doğru yanıt verir.
✅ Halüsinasyon riski azalır.
✅ Belgeler değiştikçe sistem güncellenir.

### RAG Sistem Akışı

```plaintext
Soru → Vektör Arama (FAISS) → İlgili Belgeler → Yanıt Üretimi → Cevap
```

```mermaid
flowchart TD
    A[Soru] --> B[Vektör Arama (FAISS)]
    B --> C[İlgili Belgeler]
    C --> D[Sohbet Motoru]
    D --> E[Cevap]
```

### Bileşenler

| Bileşen       | Açıklama                                             |
| ------------- | ---------------------------------------------------- |
| Soru vektörü  | Sorunun anlamını sayısal olarak temsil eder.         |
| FAISS dizini  | Belgelerin anlam vektörlerini saklar ve arama yapar. |
| Retriever     | En alakalı belgeleri bulur.                          |
| Yanıt üretici | Sadece bulunan belgelerle yanıt üretir.              |

### Örnek Görsel

![RAG Akışı](https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/rag.png)

## Notlar

* Kullanılan veri kümesi: `Metin/WikiRAG-TR`.
* Yanıtlar sadece belgelerden gelen bilgiye dayanır, model uydurma yapmaz.Eğitim amamçlı tasarlanmıştır.Bu yüzden tam anlamıyla doğru çalışan bir model değildir.
* Sistem Türkçe sorular için tasarlanmıştır.

