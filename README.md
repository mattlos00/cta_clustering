## Installazione e Configurazione dell'Ambiente

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

TODO
