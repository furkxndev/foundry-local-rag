# Local RAG Assistant

Kendi metin dosyalarım hakkındaki soruları cevaplayan küçük bir uygulama. Cevabı üreten
model bulutta değil, Microsoft Foundry Local üzerinden kendi bilgisayarımda çalışıyor, o
yüzden wi-fi kapalıyken de çalışıyor.

Yaz okulundaki Foundry Local ve RAG projesi için yaptım.

## Çalıştırmak için

```bash
foundry service start          # yerel model servisini başlat
foundry model run phi-4-mini   # sadece ilk seferde, ~3.7 GB iniyor, sonra Ctrl+C
python3 app.py                 # indeksi kurup tarayıcıyı açıyor
```

Sonra http://localhost:8000 adresine git.

`pip install` yapılacak bir şey yok, sadece Python'un standart kütüphanesini kullandım.
Python 3.8 ve üstü yeterli.

Kendi dosyalarını kullanmak istersen `documents/` klasörüne `.txt` veya `.md` at ve
sayfadaki **Rebuild index** düğmesine bas.

## Nasıl çalışıyor

Uygulama ilk açıldığında `documents/` içindeki her dosyayı okuyor, paragraf büyüklüğünde
parçalara bölüyor, her parçayı bir vektöre çeviriyor ve hepsini `rag.db` adında bir SQLite
dosyasına yazıyor.

Bir soru geldiğinde:

1. soru da aynı şekilde vektöre çevriliyor
2. kayıtlı bütün parçalarla kosinüs benzerliğine göre karşılaştırılıyor
3. en yakın 3 parça prompt'un içine bağlam olarak konuyor
4. prompt yerelde çalışan Phi-4-mini'ye gidiyor ve cevap dönüyor

Hiçbir parça yeterince benzemiyorsa model hiç çağrılmıyor, uygulama direkt bilmediğini
söylüyor. Cevabın altında görünen kaynaklar 2. adımdan geliyor, modelden değil — sebebi
aşağıda.

## Dosyalar

- `rag.py` — asıl iş burada: parçalama, vektörler, SQLite, arama, modeli çağırma.
  Tek başına da çalışıyor (`python3 rag.py` ile terminalden sohbet).
- `app.py` — küçük bir web sunucusu, üç ucu var: status, ingest, ask.
- `static/index.html` — soruyu yazdığın sayfa.
- `documents/` — cevapların çıktığı dosyalar. Projenin kendisini anlatan altı kısa metin.
- `test_cases.py` — testler. Sonuçları `TESTS.md` içinde yazdım.

## Birkaç not

**Neden SQLite.** Tek bir dosya, kurulacak sunucu yok, Python'da zaten hazır geliyor. 13
parça için bütün vektörleri tek tek karşılaştırmak anlık sürüyor. Birkaç bin parça olsaydı
gerçek bir vektör indeksi gerekirdi.

**Neden düzgün bir embedding modeli yok.** Bendeki Foundry Local kataloğunda sadece sohbet
ve görsel modelleri var, embedding modeli yok. O yüzden vektörleri TF-IDF ile kendim
ürettim. Anlamdan çok kelime örtüşmesine bakıyor, projenin en zayıf tarafı bu. Katalogda
bir embedding modeli çıkarsa `rag.py` içinde sadece `build_vocabulary()` ve `embed()`
fonksiyonlarının değişmesi yeterli.

**Kaynakları neden model değil de uygulama yazıyor.** Önce modelden cevabın sonuna
kullandığı dosyanın adını yazmasını istemiştim. Phi-4-mini yaklaşık üç cevaptan birinde
yanlış yazdı — cevabı `embeddings.txt`'ten kurup altına `sqlite.txt` yazdığı oldu. Arama
adımı hangi parçaları seçtiğini zaten bildiği için, modelin yazdığı dosya adlarını silip
gerçek dosya adlarını benzerlik puanlarıyla birlikte gösteriyorum.

## Yapamadıkları

- Sorular arasında hafızası yok, her soru sıfırdan başlıyor
- Sadece `.txt` ve `.md` okuyor, PDF veya Word yok
- Cevaplar 3.8B'lik bir modelden ne kadar iyi olabilirse o kadar iyi, bazen konuyu
  serbestçe yeniden anlatıyor
- TF-IDF araması, kaynak metinden çok farklı kelimelerle sorulan soruları kaçırabiliyor
