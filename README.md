# Youtube-Mining
Analisi delle traccie audio per lo studio del fenomeno del over politically correct

Analisi video "Carlsen vs Caruana Norway Chess 2018 | Game Analysis"
Estrazione traccie audio da video youtube:

1) Creazione funzione extract_video_from_url() il cui fine è estrarre il video tramite un percorso (path) stabilito, conservandolo in una cartella "Videos", creata appositamente per estrarre il file video ed il conseguente file audio;
2) Utilizzo extract_audio_from_video() per estrapolare la traccia audio dal video;
3) Creazione funzione slicing_audio() per dividere le tracce audio in file più piccoli (segmenti), in una nuova cartella "Audios", così da poter eseguire la speech recognition su tutti i file;
Conversione speech2text:

1) Speech recognition (operazioni preliminari): creazione funzione get_file_paths() per automatizzare la speech recognition nei vari audio;
2) recog_multiple(): funzione per eseguire la speech recognition in tutte le varie tracce;
Text cleaning:

1) Creazione di una lista contenente tutte le parole ottenute dagli audio;
2) Rimozione punteggiatura, stopwords, caratteri non alfanumerici, conversione in lowercase e controllo synsets;
Creazione dataframe con Pandas;

Sentiment analysis:

1) Riconoscimento polarità delle parole attraverso l'assegnazione di un sentiment (positivo, neutro o negativo) utilizzando Vader Sentiment;
2) Applicazione di un secondo tool di Sentiment Analysis e confronto risultati con il precedente;
3) Rappresentazione grafica dei risultati;
Addestramento Algoritmi di Classificazione:

1) Bernoulli;
2) Random Forest;
Clustering analysis;

Applicazione algoritmo K-Means con k=7;
Rappresentazione grafica dei risultati tramite WordCloud;
