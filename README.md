## Installazione e Configurazione dell'Ambiente

Nel file `report_cta.pdf` è contenuta la relazione tecnica del lavoro svolto.

Per garantire la completa riproducibilità, questo progetto utilizza **Conda** per la gestione dell'ambiente e delle dipendenze.

### Prerequisiti
*   Git
*   Anaconda o Miniconda installato.
*   Una GPU NVIDIA con driver compatibili con CUDA.

### Procedura di Installazione

1.  **Clonare il repository:**
    ```bash
    git clone https://github.com/mattlos00/cta_clustering.git
    cd cta_clustering
    ```

2.  **Creare l'ambiente Conda:**
    Il file `environment.yml` contiene l'elenco di tutte le dipendenze. Per ricreare l'ambiente, eseguire il seguente comando:
    ```bash
    conda env create -f environment.yml
    ```
    Questo comando creerà un nuovo ambiente Conda `clustering_cta`.

3.  **Attivare l'ambiente:**
    Prima di eseguire gli script, attivare l'ambiente appena creato:
    ```bash
    conda activate clustering_cta
    ```
    
---

## Utilizzo

 Una volta configurato e attivato l'ambiente virtuale, è possibile proseguire con gli esperimenti.

 Di seguito viene illustrato l'utilizzo del programma.

 1. Il primo passo è quello della **generazione degli embedding**, svolta dallo script `generate_embeddings.py`.

     Lo script permette di configurare l'embedding dei dati fornendo diverse flag all'eseguibile.
    
     * `--ideal_scenario`: attiva lo scenario ideale con header puliti e non ambigui.
     * `--noisy_headers`: attiva lo scenario con header rumorosi.
     * `--stats_features`: abilita l'utilizzo di feature statistiche dui valori nel calcolo degli embedding.
     * `--headers_only`: abilita l'utilizzo del solo header nella generazione degli embedding.
       
    Senza nessuna flag, lo script esegue l'embedding (Header, Valore) sullo scenario baseline.
    
    Lo script produce in output gli embedding prodotti, `embeddings.txt`, e le label di GT codificate, `labels.txt`.

2. Successivamente, si prosegue con la generazione del grafo k-NN, a partire dal file `embeddings.txt`, con lo script `generate_graph.py`.
   Con la flag `--method` è possibile specificare il metodo di calcolo della similarità inter-nodo tra:
   * *cosine*: similarità del coseno, il metodo di default
   * *dot*: prodotto scalare.
   * *heat*: heat kernel Gaussiano.
     
   Lo script produce in output il file `embeddings_graph.txt`, contenente il grafo k-NN prodotto.

3. La fase successiva è quella di pre-addestramento dell'Autoencoder, insieme all'esecuzion del clustering con Birch sulla rappresentazione dei dati appresa.

   Questa funzione è svolta dallo script `ae_pretrain.py`. Lo script salva il modello pre-addestrato, `ae_sdcn.pkl`, il miglior modello Autoencoder, `ae.pkl`, e le migliori label predette dall'algoritmo Birch nel file `predicted_labels_ae.txt`.

   L'addestramento dell'Autoencoder prosegue finché la loss di ricostruzione migliora. Data l'assenza di una metrica non supervisionata e affidabile per il clustering (la Silhouette converge a valori bassi in poche epoche), spetta all'utente decidere quando interrompere il training.

4. La fase finale del workflow è quella dell'addestramento dell'algoritmo **SDCN**, effettuata tramite lo scritp `sdcn_train.py`.

   Un argomento obbligatorio da fornire allo script è quello relativo alla dimensionalità dei dati in input, tramite la flag `--n_input`.

   Lo script produce in output i seguenti file:
   * `predicted_labels_sdcn.txt`: gli assegnamenti ai cluster prodotti dall'algoritmo.
   * `sdcn_best_model.pkl`: pesi del modello associato alla Silhouette Score più alta.
   * `sdcn_best_logits.txt`: proiezione dei dati nello spazio latente associata alla Silhouette Score più alta.
  
   L'addestramento si interrompe automaticamente alla convergenza della Silhouette.

5. La fase finale consiste nella visualizzazione e l'interpretazione dei risultati.

   Tramite il file `analyze_results.py` è possibile calcolare la matrice di confusione relativa alle predizioni dell'algoritmo e la proiezione **t-SNE** delle label predette.
   Il file prevede delle flag obbligatorie:
   * `--pred_labels_file`: file `.txt` relativo alle predizioni che si vogliono analizzare.
   * `--tsne`: attiva la visualizzazione delle poiezioni t-SNE.
   * `--output_dir`: directory in cui salvare i grafici.
  
   Infine, il file `visualize_geometry.py` mostra i grafici delle proiezioni t-SNE relative agli spazi latenti appresi dagli approcci **AE + Birch** e **SDCN**, spostando il focus dell'analisi su una valutazione geometrica dei cluster prodotti.

   

   



