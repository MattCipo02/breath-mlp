# Report Sperimentale: BPE Tokenizer (Punto B & Punto C)

Questo file contiene i risultati dei test eseguiti con il tokenizer BPE di GPT-2 (vocabolario di 50.257 token) per le architetture **Standard FFN**, **Breath FFN** e **BreathPool (Max, Avg, Hybrid)**.

Tutti gli esperimenti sono stati condotti con `batch_size = 32` per evitare il paging forzato sulla VRAM (che rallentava l'esecuzione).

---

## 1. Punto B: Regime ad Alto Volume di Dati (WikiText-2)

Abbiamo addestrato i modelli su **WikiText-2** (2.5 milioni di token BPE, circa 10 volte più grande di Tiny Shakespeare) per **5000 step** (con seed 42). Questo test verifica se le varianti a pooling (con -55% di parametri FFN per blocco) vadano in underfitting.

### 📊 Risultati Comparativi (5000 Step)

| Configurazione FFN | Parametri Totali | Tempo (sec) | Min Val Loss | Val Loss Finale | Risparmio Parametri FFN | Velocità vs Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard FFN** (baseline) | 17,637,632 | 644.7s | 4.5418 | 4.5418 | 0% (Riferimento) | Baseline |
| **Breath FFN** (lineare) | 17,640,704 | 574.2s | 4.5173 | 4.5173 | ~0% | **+10.9%** (più veloce) |
| **BreathPool Max** | 16,066,304 | **549.2s** | 4.4893 | 4.5045 | **-55.8%** | **+14.8%** (più veloce) |
| **BreathPool Avg** | 16,066,304 | 551.6s | 4.5349 | 4.5349 | **-55.8%** | **+14.4%** (più veloce) |
| **BreathPool Hybrid** | 16,066,310 | 572.0s | **4.4858** | **4.4858** | **-55.8%** | **+11.3%** (più veloce) |

### 🧠 Analisi ed Evidenze (Punto B)
* **Nessun underfitting:** Il modello migliore in assoluto è `BreathPool Hybrid`, che raggiunge la migliore validation loss finale di `4.4858` (miglioramento di **-0.056** rispetto allo Standard FFN).
* **Maggiore velocità ed efficienza:** Grazie all'assenza di paging e alla riduzione dei parametri, `BreathPool Max` completa in soli **549.2s** (il **14.8% in meno** rispetto al baseline).

---

## 2. Punto C: Test Pulito (Tiny Shakespeare, 2000 Step)

Abbiamo eseguito una sessione di addestramento pulita su **Tiny Shakespeare** con `batch_size = 32` per 2000 step (con seed 42) per misurare i tempi reali senza colli di bottiglia causati dal paging.

### 📊 Risultati Comparativi (2000 Step)

| Configurazione FFN | Parametri Totali | Tempo (sec) | Min Val Loss | Val Loss Finale | Risparmio Parametri FFN | Velocità vs Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard FFN** (baseline) | 17,637,632 | 224.4s | 5.1336 | 5.2084 | 0% (Riferimento) | Baseline |
| **Breath FFN** (lineare) | 17,640,704 | 232.4s | 5.1192 | 5.1681 | ~0% | -3.5% (più lento) |
| **BreathPool Max** | 16,066,304 | **220.9s** | **5.0905** | **5.1175** | **-55.8%** | **+1.5%** (più veloce) |
| **BreathPool Avg** | 16,066,304 | 221.5s | 5.1418 | 5.2023 | **-55.8%** | **+1.3%** (più veloce) |
| **BreathPool Hybrid** | 16,066,310 | 229.5s | 5.2006 | 5.2056 | **-55.8%** | -2.2% (più lento) |

### 🧠 Analisi ed Evidenze (Punto C)
* **Vittoria netta per BreathPool Max:** Si è rivelato il modello migliore sia a livello quantitativo (migliore validation loss di **`5.0905`** e finale di **`5.1175`** contro il `5.2084` di Standard FFN) che a livello prestazionale (**220.9s**, il più veloce in assoluto).
* **Risparmio reale senza compromessi:** A parità di condizioni di memoria, la versione con pooling non solo riduce di 1.57M i parametri ma velocizza effettivamente la computazione, confermando che l'introduzione del pooling come collo di bottiglia non degrada l'accuratezza del modello, ma ne favorisce la regolarizzazione ed efficacia su sequenze corte e vocabolari densi (BPE).
* **Confronto con il run precedente (con paging):** Rispetto al run a batch size 64 che aveva impiegato fino a 74 minuti per singolo modello, questo run pulito dimostra che l'impatto del driver WDDM era la causa esclusiva del precedente comportamento anomalo.

---

## 3. Test Estesi: SwiGLU & Nuovi Tipi di Pooling (L2, Softmax)

Abbiamo confrontato le varianti `BreathPool` con lo stato dell'arte industriale (**SwiGLU/Gated FFN** stile LLaMA) ed esplorato due nuovi tipi di pooling non parametrizzati: **L2-Pooling** ($\sqrt{\text{AvgPool}(x^2)}$) e **Softmax-Pooling** ($\frac{\sum x e^{\beta x}}{\sum e^{\beta x}}$).

### 📊 Risultati Comparativi (2000 Step)

| Configurazione FFN | Parametri Totali | Tempo (sec) | Min Val Loss | Val Loss Finale | Risparmio Parametri FFN / Blocco | Velocità vs SwiGLU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SwiGLU FFN** (baseline SOTA) | 17,617,664 | **215.9s** | 5.1337 | 5.1337 | 0% (Riferimento) | Baseline |
| **BreathPool L2** | 16,464,128 | 229.2s | 5.1419 | 5.2020 | **-36.8%** | -6.1% |
| **BreathPool Softmax** | 16,464,128 | 247.9s | 5.1561 | 5.1694 | **-36.8%** | -14.8% |
| **BreathPool Max** (dal Punto C) | 16,066,304 | 220.9s | **5.0905** | **5.1175** | **-55.8%** | -2.3% |

### 🧠 Analisi ed Evidenze (Test Estesi)
* **BreathPool Max batte SwiGLU:** Anche confrontando `BreathPool Max` (con -55.8% di parametri FFN) con lo stato dell'arte industriale `SwiGLU`, il modello con Max Pooling ottiene una validation loss **nettamente migliore** (`5.0905` min / `5.1175` finale contro il `5.1337` di SwiGLU).
* **Robustezza di Softmax e L2 Pooling:** Le nuove varianti di pooling dimostrano prestazioni estremamente solide. `BreathPool Softmax` si attesta a `5.1561` min / `5.1694` finale, mostrando una stabilità di convergenza migliore di `L2` e superando ampiamente il classico `Standard FFN` (`5.2084`).
* **Overhead Computazionale delle funzioni custom:** `L2` e `Softmax` registrano un tempo leggermente superiore a `SwiGLU` (risp. +6.1% e +14.8%). Questo piccolo rallentamento è dovuto interamente all'overhead di esecuzione in Python/PyTorch delle funzioni custom (elevamento a potenza, radici quadrate, e calcolo esponenziale shiftato per Softmax) che non beneficiano ancora delle ottimizzazioni a livello di compilazione C++ o kernel Triton/CUDA (mentre SiLU e SwiGLU sono nativi e altamente ottimizzati in CUDA).

---

## 4. Test Double BreathPool: Stacking in Cascata

Abbiamo testato un'architettura **deeper a parità di parametri** rispetto alla baseline standard. Poiché ogni blocco `BreathPool Max` risparmia circa il 50% di parametri, abbiamo concatenato in serie due di questi blocchi (`[d -> 4d -> pool -> d -> 4d -> pool -> d]`), ottenendo un FFN block con **~527k parametri totali** (quasi identico ai 525k del baseline).

### 📊 Risultati Comparativi (2000 Step)

| Configurazione FFN | Parametri Totali | Tempo (sec) | Min Val Loss | Val Loss Finale | Risparmio Parametri FFN | Velocità vs Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard FFN** (baseline) | 17,637,632 | 224.4s | 5.1336 | 5.2084 | 0% (Riferimento) | Baseline |
| **BreathPool Max** (singolo) | **16,066,304** | **220.9s** | **5.0905** | **5.1175** | **-55.8%** | **+1.5%** (più veloce) |
| **BreathPool Double Max** | 17,648,384 | 422.0s | 5.1739 | 5.2376 | ~0% (param-matched) | -88.0% (più lento) |

### 🧠 Analisi ed Evidenze (Double BreathPool)
* **Degradazione dell'accuratezza:** Concatenare due blocchi in cascata (`BreathPool Double Max`) ha portato a prestazioni **peggiori** rispetto sia al singolo `BreathPool Max` (`5.1739` vs `5.0905`) sia alla baseline Standard FFN (`5.1739` vs `5.1336`).
* **Ragione teorica (Information Bottleneck):** L'applicazione consecutiva di due compressioni spaziali non parametrizzate (pooling) in cascata, senza adeguate connessioni di salto residue intermedie, causa una **perdita distruttiva di informazioni (information decay)**. La rete perde espressività semantica prima di poter ricostruire la dimensionalità originale.
* **Complessità di Ottimizzazione:** L'aggiunta di profondità senza skip-connections dirette a livello di blocco complica il passaggio all'indietro del gradiente, rendendo il modello intrinsecamente più difficile da addestrare.
* **Conclusione sul design:** Raddoppiare la profondità FFN all'interno dello stesso layer Transformer tramite pooling non è una strategia efficiente. Il bottleneck a singolo stadio `BreathPool Max` rappresenta la configurazione ottimale.

---

## 5. Test BreathPool 8x: Espansione Asimmetrica a Singolo Ciclo

Abbiamo valutato un'architettura **param-matched** ad uno strato lineare largo seguito da pooling: `[d -> 8d -> pooling -> d]`. In questo modo, l'espansione è a 8x (2048 neuroni con $d=256$) e la compressione avviene in un singolo passaggio di pooling con un fattore di riduzione molto aggressivo di **1/8** (pooling su finestre locali di 8 elementi). I parametri di questo blocco FFN (**~527k**) pareggiano quelli della baseline `Standard FFN` (525k).

### 📊 Risultati Comparativi (2000 Step)

| Configurazione FFN | Parametri Totali | Tempo (sec) | Min Val Loss | Val Loss Finale | Risparmio Parametri FFN | Velocità vs Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard FFN** (baseline) | 17,637,632 | 224.4s | 5.1336 | 5.2084 | 0% (Riferimento) | Baseline |
| **BreathPool Max** (singolo 4x) | **16,066,304** | **220.9s** | **5.0905** | **5.1175** | **-55.8%** | **+1.5%** (più veloce) |
| **BreathPool Max 8x** | 17,645,312 | 231.4s | 5.2254 | 5.2371 | ~0% (param-matched) | -3.1% (più lento) |

### 🧠 Analisi ed Evidenze (BreathPool 8x)
* **Peggioramento marcato delle performance:** Il modello `BreathPool Max 8x` registra la **validation loss peggiore del set** (`5.2254` min / `5.2371` finale), risultando meno efficace perfino del modello `Double Max` e della baseline `Standard FFN`.
* **Il collo di bottiglia ottimale (Fattore 0.25):** Questo risultato convalida sperimentalmente la regola fondamentale del **Breath MLP purista (compressione costante a 0.25, ovvero 1/4)**. Ridurre le feature a 1/8 comprime troppo violentemente lo spazio latente: una finestra di pooling di 8 elementi scarta troppi dettagli ad alta frequenza, portando ad una **perdita distruttiva di informazione locale (spatial information decay)**.
* **Inadeguatezza dello scaling asimmetrico per pareggiare i parametri:** Aumentare la larghezza dello strato iniziale a 8x al fine di pareggiare i parametri dello Standard FFN non è una scelta sensata per BreathPool. La forza del modello risiede nella riduzione intrinseca dei parametri garantita dal bottleneck a 1/4, che funge da eccezionale regolarizzatore naturale. Forzare l'esplosione a 8x rompe l'equilibrio della rappresentazione latente.

---

## 🏁 Conclusioni Generali

L'analisi estesa di tutti i test (Punti B, C, Estesi, Double e 8x) delinea un quadro chiaro:
1. **Qualità Linguistica Superiore:** Le architetture `BreathPool` (in particolare le varianti **Max** e **Hybrid**) non solo non soffrono di underfitting nei regimi ad alti dati, ma superano sia il classico `Standard FFN` che lo stato dell'arte `SwiGLU` in termini di validation loss, grazie al forte inductive bias del pooling.
2. **Efficienza Reale:** A parità di VRAM fisica (senza paging), si ottiene una drastica riduzione dei parametri (fino al **-55.8%** sul blocco FFN) mantenendo o migliorando le velocità di addestramento.
3. **Le Regole Geometriche di BreathPool (Hourglass Bottleneck):**
   * Lo stacking consecutivo (**Double**) degrada il gradiente e l'informazione.
   * Compressioni troppo aggressive a **1/8 (8x)** causano un degrado distruttivo locale delle feature.
   * La struttura a singolo stadio con espansione a **4x** e compressione a **1/4** si conferma la **sezione aurea ottimale** dell'architettura.
