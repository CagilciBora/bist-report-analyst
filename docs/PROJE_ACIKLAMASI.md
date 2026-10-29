# BIST Rapor Analisti — Proje Açıklaması ve Mülakat Çalışma Rehberi

Bu belge projeyi baştan sona anlatır: neden yapıldığını, her adımın hangi dosyada olduğunu, arkasındaki kavramları küçük örneklerle ve mülakatta gelebilecek soruları. Teknik terimlerin İngilizcesi ilk kullanımda parantez içinde verilmiştir.

---

## 1. Problem: Finansal raporlar neden basit RAG için zordur?

Basit bir RAG (Retrieval-Augmented Generation, erişimle güçlendirilmiş üretim) sistemi şöyle çalışır: metni sabit uzunlukta parçalara (chunk) böl, her parçayı bir vektöre (embedding) çevir, soruya en benzer 5 parçayı getir, dil modeline (LLM) "bu bağlama göre cevap ver" de. Faaliyet raporlarında bu yaklaşım beş yerden kırılır:

1. **Tablolar.** Bilançodaki "Hasılat | 955.472 | 745.430" satırı başlığından ayrılırsa anlamını kaybeder: hangi yıl? hangi birim? Sabit boyutlu bölme tabloyu ortasından keser.
2. **Sayılar ve birimler.** Türkçede `1.234` bin iki yüz otuz dört demektir, İngilizce okumada bir virgül iki üç dört. Tablo başlığında "milyon TL" yazar ama hücrede sadece `955.472` vardır. Model "955.472 bin TL" derse 1000 kat hata yapmış olur.
3. **Birbirine çok benzeyen belgeler.** ASELSAN 2024 ve 2025 raporları aynı şablonla yazılır; "hasılat %15 arttı" cümlesi iki yılda da vardır. Anlamsal (semantic) arama doğru konuyu ama yanlış yılı getirebilir.
4. **Birden çok yıl ve şirket.** "THY ile BİM'i karşılaştır" sorusunda top-5'in beşi de THY'den gelebilir; BİM tarafı boş kalır.
5. **Kesin terimler.** Ticker'lar (ASELS), kalem adları (FAVÖK) ve sayılar, vektör uzayında "bulanıklaşır"; anahtar kelime araması bunları daha iyi yakalar.

Bu proje her bir sorun için ayrı bir çözüm ekler ve bunların etkisini sayılarla ölçer.

---

## 2. Boru hattı (pipeline): her adım ve dosyası

```
PDF → PDFParser → (TableExtractor, SectionDetector, TextCleaner) → Chunker
    → Embedder → Qdrant (DenseIndex)       ┐
    → TurkishTokenizer → BM25Index          ├→ HybridRetriever → Reranker → PromptBuilder → LLMClient → NumberVerifier → AskResponse
Soru → QueryParser (ticker/yıl/dönem) ──────┘
```

| Adım | Ne yapar | Sınıf / dosya |
|---|---|---|
| 1. Ayrıştırma (parsing) | Sayfa sayfa metin + font bilgisi; taranmış (scanned) ve bozuk kodlamalı (garbled) sayfaları tespit eder; isteğe bağlı OCR | `PDFParser` — `src/kaprag/ingestion/pdf_parser.py` |
| 2. Temizlik | Çok sayfada tekrar eden üst/alt bilgi (header/footer), sayfa numaraları, satır sonu tirelemesi | `TextCleaner` — `ingestion/cleaner.py` |
| 3. Bölüm tespiti | Gövde fontundan büyük ya da tamamen kalın kısa satırlar başlıktır; sonraki metne `section_title` olarak eklenir | `SectionDetector` — `ingestion/section_detector.py` |
| 4. Tablolar | Çizgili tablolar için PyMuPDF `find_tables()`, çizgisiz mali tablolar için koordinat tabanlı çıkarıcı; Markdown + başlık (caption) | `TableExtractor` — `ingestion/table_extractor.py` |
| 5. Meta veri | Yol (path) `data/raw/ASELS/2025_faaliyet_raporu.pdf` → ticker, yıl, dönem, belge türü | `parse_path`, `CompanyRegistry` — `ingestion/metadata.py` |
| 6. Parçalama (chunking) | Bölüm içinde cümle sınırında ~400 token, 60 token örtüşme (overlap); tablolar ayrı parça, büyük tablolar başlık tekrarlanarak bölünür | `Chunker` — `chunking/chunker.py` |
| 7. Yoğun indeks (dense) | `bge-m3` vektörleri Qdrant'a (gömülü mod) | `Embedder`, `DenseIndex` — `indexing/` |
| 8. Seyrek indeks (sparse) | Türkçe tokenizer + BM25 | `TurkishTokenizer`, `BM25Index` — `indexing/` |
| 9. Artımlı indeksleme | Dosya özeti (hash) + ayar özeti aynıysa atla | `Indexer` — `indexing/indexer.py` |
| 10. Soru çözümleme | Ticker/alias, yıl, "son iki yıl", dönem, belge türü, karşılaştırma mı? | `QueryParser` — `retrieval/query_parser.py` |
| 11. Hibrit erişim | Dense top-30 + BM25 top-30 → RRF → (rerank top-20) → final 6 | `HybridRetriever` — `retrieval/hybrid.py` |
| 12. Yeniden sıralama | Cross-encoder ile soru-parça çiftlerini puanlar | `Reranker` — `retrieval/reranker.py` |
| 13. İstem (prompt) | Sürümlü şablon, kaynak etiketli bağlam | `PromptBuilder` — `generation/prompt.py`, `prompts/answer_tr.md` |
| 14. Üretim | Ollama (varsayılan), Anthropic, OpenAI | `LLMClient` — `generation/llm.py` |
| 15. Sayı doğrulama | Cevaptaki her sayıyı bağlamda arar | `NumberVerifier` — `generation/verifier.py` |
| 16. Orkestrasyon | Hepsini bağlar, süreleri ölçer | `RAGPipeline` — `src/kaprag/pipeline.py` |
| 17. Arayüzler | CLI, REST API, web arayüzü | `cli.py`, `api/main.py`, `ui/app.py` |

---

## 3. Kavramlar (kısa örneklerle)

### 3.1 Yoğun gömme (dense embedding)
Bir metni ~1024 boyutlu bir vektöre çeviren model (bi-encoder). Anlamca yakın metinler uzayda yakın düşer: "net kâr" ile "dönem karı" farklı kelimeler olsa da benzer vektörler üretir. Benzerlik kosinüs (cosine) ile ölçülür; vektörleri birim uzunluğa normalize ettiğimiz için kosinüs = iç çarpım.
*Güçlü yanı:* eş anlamlılar, farklı ifadeler. *Zayıf yanı:* "ASELS" ile "THYAO" ya da "955.472" ile "955.427" vektör uzayında neredeyse aynıdır.

### 3.2 BM25
Klasik anahtar kelime puanlaması. Belge *d* ve sorgu *q* için:

```
score(d, q) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) )        (rank_bm25 BM25Okapi)
```

- **tf (term frequency, terim sıklığı):** terim belgede kaç kez geçiyor. Daha çok geçmesi iyi ama azalan getiriyle (k1 = 1,5 doygunluğu ayarlar).
- **IDF (inverse document frequency, ters belge sıklığı):** terim kaç belgede geçiyor. Nadir terim (FAVÖK, ASELS) değerli; her yerde geçen terim değersiz.
- **Uzunluk normalizasyonu (length normalization, b = 0,75):** uzun parça sırf uzun olduğu için kazanmasın.

*Mini örnek:* N = 10 parça. "favök" 1 parçada geçiyor → IDF = ln(9,5/1,5) ≈ 1,85. "şirket" 4 parçada → IDF = ln(6,5/4,5) ≈ 0,37. Ortalama uzunluk avgdl = 100 token. Parça A (100 token) "favök"ü 2 kez içeriyor:
tf kısmı = 2·2,5 / (2 + 1,5·(1 − 0,75 + 0,75·1)) = 5 / 3,5 ≈ 1,43 → katkı ≈ 1,85 · 1,43 ≈ 2,64.
Parça B (200 token, yani 2 kat uzun) da 2 kez içeriyor: payda = 2 + 1,5·(0,25 + 1,5) = 4,625 → 5/4,625 ≈ 1,08 → katkı ≈ 2,0. Aynı sıklık, ama uzun parça daha az puan alır.

Not: BM25Okapi, parçaların yarısından fazlasında geçen terimlere negatif IDF verir (kütüphane bunu küçük pozitif bir değerle değiştirir). Testlerde bunu bizzat gördük (`tests/test_retrieval.py`).

### 3.3 Neden hibrit (hybrid)?
- "ASELS 2025 FAVÖK marjı" → kesin terimler: BM25 güçlü.
- "Şirketin yurt dışı satışlarının payı ne?" → raporda "ihracatın toplam hasılata oranı" yazıyor: kelimeler örtüşmüyor, dense güçlü.
İkisini birleştirince her iki soru tipinde de iyi sonuç alırız. Değerlendirme tablosundaki `dense`, `bm25`, `hybrid` satırları bunu sayılarla gösterir.

### 3.4 Reciprocal Rank Fusion (RRF)
```
RRF(d) = Σ_i 1 / (k + rank_i(d)),   k = 60, sıra 1'den başlar
```
*Çalışılmış örnek:* dense sırası [A, B, C], BM25 sırası [C, A, D].
- A: 1/(60+1) + 1/(60+2) = 0,01639 + 0,01613 = 0,03252
- C: 1/(60+3) + 1/(60+1) = 0,01587 + 0,01639 = 0,03226
- B: 1/62 = 0,01613 · D: 1/63 = 0,01587
Sonuç: A, C, B, D. İki listede de geçen belgeler, tek listede birinci olan belgeyi geçer. (Testi: `test_rrf_toy_example`.)

**Neden puanları toplamıyoruz?** BM25 puanı 0–30 arası ve soruya göre değişir; kosinüs 0–1 arası. Toplamak için normalize etmek gerekir ve bu kırılgandır (bir soruda BM25 maksimumu 8, diğerinde 25). RRF yalnızca sıraları kullanır, ölçek bağımsızdır, ayar gerektirmez. k büyüdükçe ilk sıraların avantajı azalır (1/61 ile 1/62 arasındaki fark küçüktür).

### 3.5 Meta veri filtreleme (metadata filtering): neden aramanın İÇİNDE?
"ASELSAN 2025 hasılatı" sorusunda tüm indeksten top-30 alıp sonra ASELS-2025 dışını atarsak, 30'un 25'i THY/BİM olabilir; elimizde 5 (belki 0) aday kalır. Filtre aramanın içinde uygulanırsa (Qdrant `query_filter`, BM25'te yalnızca filtreyi geçen alt kümenin puanlanması) en iyi 30 *ASELS-2025* parçası gelir. Aynı `SearchFilter` nesnesi iki arayıcıda da kullanılır (`indexing/filters.py`), böylece ikisi asla farklı filtrelerle çalışmaz.

### 3.6 Cross-encoder vs bi-encoder
- **Bi-encoder** (embedder): soru ve parça *ayrı ayrı* vektöre çevrilir. Parça vektörleri önceden hesaplanır; arama milisaniyeler sürer. Ama iki metin birbirini "görmez".
- **Cross-encoder** (reranker): soru ve parça *birlikte* modele girer; dikkat (attention) mekanizması "2025 hasılatı" ile tablodaki "Hasılat | 955.472" satırını doğrudan eşleştirebilir. Çok daha isabetli, ama her çift için çalışmalı. Bu yüzden yalnızca birleştirilmiş ilk 20 adaya uygulanır.

### 3.7 Yapıya duyarlı parçalama ve tabloların bütün tutulması
Metin bölümler içinde cümle sınırından bölünür; iki bölüm asla aynı parçaya girmez (aksi halde "Riskler" bölümünün sonu ile "Gelir Tablosu"nun başı karışır). Tablolar ayrı parçadır; büyük tablo satırlardan bölünür ve **her parçada başlık satırı + caption tekrarlanır**: "| Kalem | 31 Aralık 2025 | 31 Aralık 2024 |" olmadan "| 86.035 | 95.992 |" anlamsızdır. Token sayısı embedding modelinin kendi tokenizer'ı ile ölçülür, çünkü Türkçe kelimeler birçok alt-token'a bölünür.

Ek olarak her parçanın indekslenen metnine kısa bir bağlam başlığı eklenir: `Türk Hava Yolları A.O. (THYAO) 2025 finansal_rapor | Bölüm adı`. Böylece içinde şirket adı geçmeyen bir tablo parçası da şirket adıyla bulunabilir.

### 3.8 Türkçe noktalı/noktasız I sorunu
Türkçede dört harf vardır: İ/i ve I/ı. Python'un `.lower()` fonksiyonu İngilizce kuralını uygular: `"I".lower()` → `"i"` (Türkçede `"ı"` olmalı), `"İ".lower()` → `"i̇"` (i + birleşik nokta U+0307!). Sonuç: "İHRACAT" ile "ihracat" farklı token olur ve BM25 eşleşmez. `utils/turkish.py` içindeki `tr_lower` önce `I→ı`, `İ→i` dönüşümünü yapar, sonra küçültür. Ayrıca `tr_ascii_fold` ile "turk hava yollari" yazan kullanıcı da THYAO'yu bulur. Tokenizer ayrıca kesme işaretinden sonraki eki atar ("ASELSAN'ın" → "aselsan") ve sayıları tek token tutar ("955.472" → "955472", "%12,5" tek token).

### 3.9 Sayı doğrulama ve LLM'lerin sayı hataları
LLM'ler metni token token üretir; "955.472" birkaç token'dır ve model bir rakamı kaydırabilir, sütunları (2024/2025) karıştırabilir, yuvarlayabilir ya da birim ölçeğini (bin/milyon) yanlış yazabilir. Bu hatalar akıcı bir metnin içinde fark edilmez.
`NumberVerifier`: cevaptaki her sayıyı `parse_tr_number` ile çözer (değer × ölçek), bağlamdaki tüm sayılarla karşılaştırır.
- Biçim ve ölçek farkı tolere edilir: "1,2 milyar TL" = "1.200.000 bin TL".
- Yuvarlama toleransı cevabın son basamağının yarısıdır: "955,5 milyar TL" ↔ milyon TL tablosundaki "955.472" ✓.
- Parçanın kendi birimi okunur: başlıkta "milyon TL" varsa "955.472 bin TL" cevabı ✗ olarak işaretlenir.
- Gösterilen hesaplar yeniden hesaplanır: "(955.472 − 745.430) / 745.430 = %28,2" → sonuç doğruysa "calculated".
- Yıllar, tarihler, kaynak etiketleri ve madde numaraları atlanır.
Arayüz doğrulanmayan sayıları kırmızıyla vurgular. Sınırı: sayının bağlamda *var olduğunu* doğrular, *doğru kaleme* ait olduğunu değil.

---

## 4. Tasarım kararları ve alternatifler

| Konu | Seçim | Alternatifler | Neden |
|---|---|---|---|
| Vektör veritabanı | **Qdrant** (gömülü mod) | Chroma, pgvector, FAISS | Gerçek filtreli arama; aynı kod Qdrant sunucusuna URL değiştirerek geçer; Docker gerekmez. Chroma'nın filtreleri daha zayıf, pgvector için Postgres kurmak gerekir, FAISS'te meta veri filtresi yok |
| Embedding | **bge-m3** | multilingual-e5-base | Çok dilli erişimde daha güçlü, 8k bağlam. e5 ~3 kat hızlı (CPU'da cazip), config ile seçilebilir; `query:`/`passage:` önekleri otomatik eklenir |
| Birleştirme | **RRF** | Ağırlıklı puan toplamı | Ölçek bağımsız, ayarsız; ağırlıklı toplam normalizasyon ve ağırlık ayarı ister |
| Reranker | **bge-reranker-v2-m3**, top-20 | Yok / LLM ile sıralama | Benzer pasajlar arasında belirgin isabet artışı; maliyeti sınırlı |
| Tablo çıkarımı | **PyMuPDF + kendi çizgisiz tablo çıkarıcımız** | pdfplumber | Aynı sayfalarda karşılaştırdım: pdfplumber çizgisiz bilançoda harf harf bölünmüş çöp üretti, çizgili tablolarda üstünlüğü yoktu |
| Yerel model vs API | **Ollama qwen2.5:3b** varsayılan | Claude / GPT | Ücretsiz, veri makineden çıkmaz; API'ler config ile açılır, boru hattı değişmez |
| Soru çözümleme | **Kurallar**, LLM isteğe bağlı | Yalnız LLM | Açıklanabilir, test edilebilir, deterministik, maliyetsiz |
| "bilanço" gibi kelimeler | **Yumuşak ipucu** | Sert filtre | Faaliyet raporları da mali tabloları içerir; sert filtre onları gizlerdi |
| Donanım | CPU tekerlekleri + otomatik cihaz | Yalnız CUDA | Geliştirme makinesinde NVIDIA GPU yok; kod CUDA'yı görünce GPU'ya geçer |

---

## 5. Değerlendirme metriklerini okumak

- **Hit@k:** Soruların yüzde kaçında ilk k parça içinde en az bir doğru parça (doğru şirket/yıl/belge ve beklenen sayfa) var. Hit@1 "ilk sonuç doğru mu?", Hit@6 "LLM'e giden bağlamda doğru bilgi var mı?" sorusunu cevaplar. Hit@6 düşükse LLM ne kadar iyi olursa olsun doğru cevap veremez.
- **MRR (Mean Reciprocal Rank):** İlk doğru parçanın sırasının tersi, ortalaması. Doğru parça 1. sıradaysa 1, 2. ise 0,5, 3. ise 0,33. Sıralama kalitesini tek sayıda özetler.
- **Karşılaştırma kapsaması (side coverage):** Karşılaştırma sorularında her iki tarafın (şirket/yıl) da bağlamda temsil edilme oranı.
- **Filtre doğruluğu:** QueryParser'ın ticker ve yılları tam doğru çıkardığı soruların oranı.
- **Sayısal sadakat (numeric faithfulness):** Beklenen sayıların cevapta (biçim/ölçek farkı tolere edilerek) geçme oranı.
- **Doğrulanmamış sayı oranı:** Cevaplardaki sayılardan bağlamda bulunamayanların oranı; halüsinasyon göstergesi.
- **Cevaplanamaz soru başarısı:** Belgelerde olmayan sorularda "bu bilgi belgelerde yer almıyor" denme oranı.
- **Gecikme (latency):** Aşama başına ortalama ve p95 (isteklerin %95'i bundan hızlı).

### Benim sonuçlarım
Ortam: dizüstü CPU (Ryzen 5 5500U, GPU yok), 40 soru (34 cevaplanabilir, 6 cevaplanamaz), hepsi henüz `reviewed: false`.

| Mod (stemming / table boost) | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| dense | 0,53 | 0,88 | 0,88 | 0,681 |
| bm25 | 0,50 | 0,62 | 0,74 | 0,575 |
| bm25 + stemming | 0,50 | 0,68 | 0,76 | 0,604 |
| hybrid | 0,44 | 0,76 | 0,88 | 0,598 |
| hybrid + stemming | 0,53 | 0,79 | 0,91 | 0,677 |
| **hybrid_rerank** | **0,82** | **0,94** | **0,94** | **0,877** |
| hybrid_rerank + table boost | 0,68 | 0,88 | 0,91 | 0,775 |

Filtre doğruluğu: ticker 1,00 · yıl 0,97. Üretim (qwen2.5:3b, hybrid_rerank, boost açık): sayısal sadakat 0,58 · doğrulanmamış sayı oranı 0,21 · cevaplanamaz soru başarısı 1,00 · yanlış ret 0,09 · kaynak gösterme 0,76.

**Bu sonuçlar ne söylüyor?**
- **En büyük kazanç reranker'dan geliyor:** ilk sonucun doğru olma oranı 0,53'ten 0,82'ye çıkıyor. Bedeli CPU'da sorgu başına ~27 saniye.
- **Hibrit tek başına dense'i geçmedi.** Türkçe kök bulma (stemming) açılınca BM25 iyileşiyor ve hibrit dense'i hafifçe geçiyor (Hit@6 0,91'e karşı 0,88). Dürüst yorum: soruların çoğu raporla benzer kelimeleri kullandığı için dense de iyi çalışıyor; hibritin asıl katkısı reranker'a daha iyi bir aday havuzu (top-20) vermesi.
- **Table boost her modda zarar verdi.** Benzer görünen bağlı ortaklık tabloları doğru sayfanın önüne geçti. Bu yüzden varsayılanı kapattım. Ölçmeden "mantıklı" görünen bir özelliği açık bıraksaydım sistemi kötüleştirmiş olurdum: değerlendirmenin değeri tam olarak bu.
- **Darboğaz erişim değil, 3B model:** doğru sayfa %94 oranında bağlamda, ama beklenen sayıların sadece %58'i cevapta. Model doğru tablodan komşu sayıyı seçebiliyor. Çözüm: daha büyük model (7B veya API), ya da tablo hücresi seviyesinde cevap çıkarma.
- **Sayı doğrulayıcı uydurma sayıyı yakaladı** (q32: hiçbir belgede olmayan "140,0 Milyar TL" işaretlendi), ama bağlamda var olan bir sayının yanlış kaleme yazılmasını yakalayamadı (q33). Bu, mülakatta dürüstçe anlatılması gereken sınır.
- **Cevaplanamaz 6 sorunun 6'sı** doğru şekilde "belgelerde yok" ile cevaplandı; 4'ünde filtre sıfır parça döndürdüğü için LLM hiç çağrılmadı.


---

## 6. Sınırlamalar ve çözüm yolları

- **Taranmış raporlar:** THY 2024 raporunun 40 sayfası görüntüdür; ASELSAN 2025 finansal raporu Type-3 fontlarla Unicode eşlemesi olmadan basılmıştır (metin "garbled" olarak tespit edildi). Çözüm: `parsing.ocr_enabled: true` + `sudo apt install tesseract-ocr tesseract-ocr-tur`; daha iyisi düzen (layout) modelleri.
- **Karmaşık tablolar:** Birleşik hücreler, döndürülmüş başlıklar, çok katlı başlıklar sadeleşir; bazı döndürülmüş başlıklı tablolar kalite kontrolünden geçemez ve düz metne düşer. Çözüm: tablo yapısı tanıma modelleri (ör. Table Transformer) veya şirket bazlı şablonlar.
- **Konsolide vs solo tablolar:** Meta veride ayrılmıyor. Çözüm: başlıktan "konsolide/solo" etiketi çıkarıp filtre alanı yapmak.
- **Yeniden düzenlenmiş (restated) rakamlar:** 2025 raporundaki 2024 sütunu, 2024 raporundakiyle farklı olabilir (enflasyon muhasebesi TMS 29 ile yeniden ifade). Çözüm: her sayıyı "rapor yılı" ve "ait olduğu dönem" ile ayrı saklamak, cevapta ikisini de belirtmek.
- **Sayı doğrulayıcı** sayının var olduğunu doğrular, doğru kaleme bağlı olduğunu değil. Çözüm: tablo hücresi seviyesinde (satır, sütun) eşleştirme.
- **CPU gecikmesi:** Reranker ve 3B model dizüstü CPU'da saniyeler-dakikalar sürer; GPU'da bu süreler kat kat düşer.

---

## 7. Yeni bir şirket/rapor eklemek: 3 adım

1. Ticker yeniyse `data/companies.csv` dosyasına bir satır ekle (ad, takma adlar `;` ile, sektör).
2. PDF'i `data/raw/<TICKER>/<YIL>_<belge_türü>.pdf` olarak koy (ör. `data/raw/TUPRS/2025_faaliyet_raporu.pdf`).
3. `uv run kap ingest` çalıştır. Değişmeyen belgeler atlanır; `uv run kap inspect TUPRS 2025` ile ayrıştırmayı kontrol et.

---

## 8. Mülakat soru-cevapları

**1. Neden hibrit arama?**
Dense arama anlamı yakalar ama kesin terimleri (ticker, kalem adı, sayı) bulanıklaştırır; BM25 kesin terimlerde güçlü ama eş anlamlıları kaçırır. Dürüst sonuç: benim setimde hibrit tek başına dense'i ancak stemming ile ve az farkla geçti (Hit@6 0,91'e 0,88). En büyük kazanç, hibritin ürettiği aday havuzunu cross-encoder ile yeniden sıralamaktan geldi (Hit@1 0,82).

**2. Neden puanları toplamak yerine RRF?**
BM25 ve kosinüs farklı ölçeklerde ve soruya göre değişen dağılımlarda. Toplamak normalizasyon ve ağırlık ayarı gerektirir. RRF sadece sıraları kullanır, ayarsızdır ve sağlamdır.

**3. Modelin sayı uydurmasını nasıl engelliyorsun?**
Üç katman: (a) prompt sayıları birimiyle aynen kopyalamayı ve kaynak etiketini zorunlu kılıyor; (b) bağlam tablo başlığı ve birim satırıyla birlikte veriliyor; (c) üretimden sonra `NumberVerifier` her sayıyı bağlamda arıyor, bulamadıklarını işaretliyor ve UI kırmızıyla gösteriyor. Engellemiyorum, *görünür* kılıyorum; bu daha dürüst.

**4. Metadata filtresi neden aramadan sonra değil de içinde?**
Sonradan filtrelersek top-k'nın çoğu başka şirketlerden gelir ve elenince elimizde az aday kalır. İçeride filtrelemek, filtrelenmiş alt kümenin en iyi k parçasını garanti eder.

**5. Cross-encoder ile bi-encoder farkı?**
Bi-encoder metinleri ayrı kodlar, hızlıdır, önceden hesaplanır. Cross-encoder çifti birlikte işler, daha isabetlidir ama her sorguda çalışır; bu yüzden sadece ilk 20 aday için.

**6. Tabloları nasıl ele aldın?**
Çizgili tablolar için PyMuPDF `find_tables()`, çizgisiz mali tablolar için kelime koordinatlarından satır/sütun çıkaran kendi algoritmam (sayılar sağa yaslı olduğu için sütunları sağ kenara göre kümeliyorum). Bozuk çıkarımları (yapışık kelimeler, bölünmüş sayılar) kalite kontrolüyle eliyorum. Tablolar ayrı parça, büyükler başlık tekrarıyla bölünüyor.

**7. Türkçeye özgü ne yaptın?**
Noktalı/noktasız I için doğru küçültme, kesme işaretinden sonra ek atma, Türk sayı biçimi (1.234.567,89, (1.234) negatif, %12,5), bin/milyon/milyar ölçekleri, isteğe bağlı Snowball kök bulma, Türkçe takma adlar (THY, BİM).

**8. Karşılaştırma soruları nasıl çalışıyor?**
Soru (şirket, yıl) çiftlerine ayrılıyor, her çift için ayrı filtreyle eşit bütçeyle arama yapılıyor, sonuçlar gruplanıp prompt'a "karşılaştırma" notuyla veriliyor ve cevabın sonunda tablo isteniyor.

**9. Tüm BIST şirketleri ve 10 yıla nasıl ölçeklersin?**
~500 şirket × 10 yıl × birkaç belge ≈ 15–20 bin belge, birkaç milyon parça. Qdrant'ı sunucu moduna al (aynı kod, URL değişir), payload indeksleri gerçekten devreye girer; BM25'i Elasticsearch/OpenSearch'e ya da Qdrant'ın sparse vektörlerine taşı; ingestion'ı kuyruk + işçilerle (Celery/RQ) paralelleştir; embedding'i GPU'da toplu çalıştır.

**10. Yeni KAP açıklamalarıyla nasıl güncel tutarsın?**
Sitenin kullanım koşullarına ve robots.txt'e uyan, onaylı bir besleme/API varsa onu periyodik dinleyen bir görev; yeni dosya `data/raw`'a düşer, artımlı ingestion sadece yeniyi işler (hash + ayar özeti). Özel durum açıklamaları tarihle (`2025-03-14_ozel_durum.pdf`) saklanır.

**11. Üretimde ne değiştirirdin?**
Qdrant sunucusu, kimlik doğrulama, istek kotası, gözlemlenebilirlik (her aşamanın süresi zaten ölçülüyor → metrik sistemi), prompt/model sürümlerini cevapla birlikte loglama, değerlendirme setini CI'da regresyon testi olarak çalıştırma, GPU üzerinde vLLM gibi bir sunucu, önbellek.

**12. Neden yerel model?**
Ücretsiz, veri dışarı çıkmaz, demo internetsiz çalışır. Kalite gerekirse aynı boru hattı config ile Claude/GPT'ye geçer; değerlendirme ikisini karşılaştırmayı sağlar.

**13. Chunk boyutunu neden 350–450 token seçtin?**
Bir paragraf grubu veya orta boy tablo sığacak kadar büyük, tek bir konuya odaklanacak kadar küçük. Altı parça + talimat 3B modelin 8k bağlamına rahat sığar. Token sayısını embedding tokenizer'ı ile ölçüyorum.

**14. Overlap neden var?**
Bir bilgi iki parçanın sınırına denk gelirse ikisinde de bağlamıyla görünsün diye. Cümle düzeyinde yaptım ki cümle ortadan bölünmesin.

**15. Taranmış PDF'leri nasıl tespit ediyorsun?**
Sayfada 40 karakterden az metin varsa "scanned". Metin var ama kontrol karakterleri çok ve "kelimelerin" çoğu sesli harf içermiyorsa "garbled" (bozuk font kodlaması). İkisi de loglanıyor; OCR açıksa Tesseract ile okunuyor.

**16. Header/footer'ı şirkete özel kural yazmadan nasıl temizledin?**
Her sayfanın en üst ve en alt 3 satırını alıp rakamları `#` ile değiştiriyorum ("Sayfa 12" = "Sayfa 13"). Sayfaların en az %10'unda (ve en az 5 sayfada) tekrar eden satır boilerplate'tir.

**17. Değerlendirme setini nasıl oluşturdun, yanlılık riski?**
İndekslenmiş parçaları okuyarak, gerçekten gördüğüm sayfa ve sayılarla 40 soru; hepsi `reviewed: false`, insan incelemesi bekliyor. Soruları herhangi bir modu kayıracak şekilde yazmadım ve setle oynayıp sonuç iyileştirmedim. Risk: soruları yazan, metni gördüğü için terimleri kopyalayabilir; bu BM25'i kayırır. Bunu bilerek ifadeleri çeşitlendirdim.

**18. Reranker her zaman daha mı iyi?**
Bende en büyük kazancı o sağladı (Hit@1 0,53 → 0,82), ama CPU'da sorgu başına ~27 saniye ekliyor. GPU'da bu süre çok kısalır. Gecikme kritikse `hybrid` + stemming iyi bir orta yol (Hit@6 0,91, ~1 saniye).

**19. Table boost işe yaradı mı?**
Hayır, ve bunu ölçerek öğrendim. Fikir mantıklıydı: "hasılat" gibi kelimeler geçen sorularda tablo parçalarının puanını ×1,15 artırmak. Ama değerlendirmede her modda Hit@1 düştü (reranker ile 0,82'den 0,68'e), çünkü benzer görünen başka tablolar doğru sayfanın önüne geçti. Özelliği kodda bıraktım ama varsayılanı kapattım. Bu, değerlendirme olmadan alınacak kararların neden riskli olduğunun iyi bir örneği.

**20. En zayıf noktan ne?**
3B yerel model: erişim %94 doğru sayfayı getiriyor ama model beklenen sayıların sadece %58'ini doğru yazıyor. Sayı doğrulayıcı sayının varlığını kontrol ediyor, doğru kaleme ait olduğunu değil. Değerlendirme seti küçük (40 soru) ve henüz insan tarafından doğrulanmadı.
