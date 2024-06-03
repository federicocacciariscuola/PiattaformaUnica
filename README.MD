# Gemini IA

Un programma in python che, utilizzando le API di Gemini riesce permette di avere una chat con l'IA di Google.

## Table of content

* [Gemini](#gemini-ia)
* [Setup](#setup)
* [Utilizzo](#utilizzo)
  * [Comandi](#comandi)

## Setup

1. Il programma richiede una versione recente di python (>3.9)
2. Il programma richiede già che sia installato l'[SDK di Google Cloud](https://cloud.google.com/sdk/docs/install) e sia già [configurato](https://cloud.google.com/sdk/docs/authorizing?) con un account che possa accedere all'API di [Vertex AI](#https://cloud.google.com/vertex-ai/docs/reference/rest).
3. Configurato l'account, è consigliabile installare le libreria all'interno di un ``virtualenv``, utilizzando ad esempio la libreria ``venv``, installabile e attivabile col comando
   ```
   mkdir -p ~/.venvs
   python3 -m venv ~/.venvs/gemini-cli
   ```
4. Attivare l'ambiente virtuale
   ```
   source ~/.venvs/gemini-cli/bin
   ```
5. Installare le librerie necessarie dal file [requirements.txt](https://github.com/federicocacciariscuola/PiattaformaUnica/main/requirements.txt)

## Utilizzo

Eseguire il file [CLI.py](https://github.com/federicocacciariscuola/PiattaformaUnica/main/CLI.py) con il comando

```
python3 CLI.py
```

Dopo avere inserito quelle che sono le istruzioni di sistema che Gemini seguirà come linee guida per avere informazioni sull'utilizzo del programma digitare ``/help`` e verrà fuori il menù della guida.

### Comandi

**/help** -> Mostra questo messaggio di aiuto
**/file** -> Invia un file tra quelli consentiti
**/exit** -> Stoppa la conversazione/save -> Salva la conversazione
**/load** -> Carica la conversazione

| Immagini     | Audio        | Video         | Documento                   |
| ------------ | ------------ | ------------- | --------------------------- |
| - image/png  | - audio/wav  | - video/mp4   | - text/plain                |
| - image/jpeg | - audio/mp3  | - video/mpeg  | - text/html                 |
| - image/webp | - audio/aiff | - video/mov   | - text/css                  |
| - image/heic | - audio/aac  | - video/avi   | - text/javascript           |
| - image/heif | - audio/ogg  | - video/x-flv | - application/x-javascript  |
|              | - audio/flac | - video/mpg   | - text/x-typescript         |
|              |              | - video/webm  | - application/x-typescript  |
|              |              | - video/wmv   | - text/csv                  |
|              |              | - video/3gpp  | - text/markdown             |
|              |              |               | - text/x-python             |
|              |              |               | - application/x-python-code |
|              |              |               | - application/json          |
|              |              |               | - text/xml                  |
|              |              |               | - application/rtf           |
|              |              |               | - text/rtf                  |
|              |              |               | - application/pdf           |

N.B. Il programma salva le conversazioni usando la libreria pkl, perciò non è possibile aprirle come normali file testuali.
