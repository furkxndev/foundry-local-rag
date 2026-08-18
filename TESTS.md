# Testler

`python3 test_cases.py` ile çalışıyor. Aşağıdaki sonuçları macOS'ta, Foundry Local 0.8.119
ve `Phi-4-mini-instruct-generic-gpu` modeliyle, 6 doküman / 13 parçalık indeksle aldım.

Dokuz soru denedim: altısının cevabı dökümanlarda var, üçünün yok.

| Soru | Beklenen | Sonuç | En iyi kaynak | Süre |
|---|---|---|---|---|
| What is Foundry Local? | cevap versin | geçti | foundry-local.txt (0.313) | 13.8 sn* |
| How does the RAG pattern work? | cevap versin | geçti | rag-pattern.txt (0.151) | 10.9 sn |
| Why is SQLite used for local storage? | cevap versin | geçti | sqlite.txt (0.229) | 12.6 sn |
| What is cosine similarity? | cevap versin | geçti | embeddings.txt (0.286) | 7.8 sn |
| What chunk size should I use? | cevap versin | geçti | embeddings.txt (0.259) | 5.4 sn |
| Which instructions matter most in the system prompt? | cevap versin | geçti | prompt-engineering.txt (0.188) | 5.5 sn |
| Who won the World Cup in 2018? | bilmiyorum desin | geçti | — | 3.9 sn |
| What is the capital of Australia? | bilmiyorum desin | geçti | — | 0.0 sn |
| *(boş soru)* | bilmiyorum desin | geçti | — | 0.0 sn |

Dokuzu da geçti.

\* İlk soru modeli belleğe yüklediği için uzun sürüyor, sonrakiler hızlı.

## Gözlemler

Cevaplanabilen altı soruda da en yüksek puanlı parça doğru dosyadan geldi, cevabın içeriği
de o metinle uyuşuyordu.

Bilmediğini söylemesi iki ayrı yerde çalışıyor. Son iki soruda hiçbir parça 0.02 benzerlik
eşiğini geçemedi, o yüzden model hiç çağrılmadı ve cevap anında döndü. Dünya Kupası
sorusunda zayıf da olsa bir parça geldi ama prompt'taki kural sayesinde model yine de
cevap vermeyi reddetti.

Boş soru API'ye ulaştığında 400 dönüyor, sadece boşluktan oluşan bir soru ise bilmiyorum
cevabına düşüyor.

## Testler sırasında çıkan sorun

Baştaki halinde modelden cevabın sonuna kaynak dosyanın adını yazmasını istiyordum.
Phi-4-mini yaklaşık üç cevaptan birinde yanlış dosya adı yazdı — mesela cevabı
`embeddings.txt`'ten kurup altına `sqlite.txt` yazdı.

Çözüm olarak bu işi modelden tamamen aldım. Sunucu, modelin yazdığı dosya adlarını siliyor
ve aramanın gerçekten seçtiği dosyaları benzerlik puanlarıyla birlikte döndürüyor, arayüz
de onları gösteriyor.
