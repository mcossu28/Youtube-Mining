# DOWNLOAD VIDEO DA URL
def extract_video_from_url():
    url = "https://www.youtube.com/watch?v=XUVIewfjGr8&ab_channel=GMHuschenbeth"
    path = "Videos"
    ydl_opts = {}
    os.chdir(path)
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ESTRAZIONE AUDIO DAL FILE VIDEO
def extract_audio_from_video():
    clip = mp.VideoFileClip(
        r"C:\Users\marco\PycharmProjects\WAAT-2021\WAAT-2021\Videos\Carlsen vs Caruana Norway Chess 2018 _ Game Analysis-XUVIewfjGr8.mp4")
    clip.audio.write_audiofile(r"Videos/estratto.wav")


# SLICING AUDIO ORIGINALE
def slicing_audio():
    os.makedirs('Videos/Audios')
    audio = AudioSegment.from_wav("Videos/estratto.wav")
    n = len(audio)
    counter, interval, overlap, start, end, flag = 1, 40 * 1000, 1.5 * 1000, 0, 0, 0
    for i in range(0, n, interval):
        if i == 0:
            start = 0
            end = interval
        else:
            start = end - overlap
            end = start + interval
        if end >= n:
            end = n
            flag = 1
        chunk = audio[start:end]
        filename = 'Videos/Audios/pezzo' + str(counter) + '.wav'
        chunk.export(filename, format="wav")
        print("Processing chunk " + str(counter) + ". Start = " + str(start) + " end = " + str(end))
        counter = counter + 1


def get_file_paths(dirname):
    file_paths = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


# SPEECH RECOGNITION
def recog_multiple(file):
    r = sr.Recognizer()
    r_types = ['recognize_google']
    results = []
    for r_type in r_types:
        result = ''
        with sr.AudioFile(file) as source:
            audio = r.record(source)
            try:
                result = str(getattr(r, r_type)(audio))
            except sr.UnknownValueError:
                result = 'Speech Recognition could not understand audio'
            except sr.RequestError as e:
                result = 'Speech Recognition could not understand audio; {0}'.format(e)
        results.append(result.split(' '))
    return results


# PULIZIA TESTO
def remove_stopwords(text):
    translator = str.maketrans(",", " ", string.punctuation)
    stopwords = nltk.corpus.stopwords.words('english')
    output = [i for i in text if i not in stopwords]
    output = [w for w in output if w.isalpha()]
    output = [word.translate(translator).lower() for word in output]
    output = [w for w in output if w and w not in stopwords]
    output = [w for w in output if wn.synsets(w)]
    return output


# CREAZIONE DATAFRAME
def Get_DF():
    with open('Videos/Audios/recognized.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        df = pd.DataFrame([word for word in enumerate(reader)], columns=["Colonna1", "Testo"])  # creazione dataframe
    del df['Colonna1']
    df = df.iloc[1:]
    df["Testo"] = df["Testo"].apply(remove_stopwords)
    return df


# TOPIC MODELING
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


# CLUSTER ANALYSIS K-MEANS
def cluster_analysis(sentences):
    tfidfVectorizer = TfidfVectorizer()
    X = tfidfVectorizer.fit_transform(sentences)
    Sum_of_squared_distances = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    true_k = 7  # abbiamo scelto 7 in quanto risulta essere l'unico gomito visibile
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels = model.labels_
    cl = pd.DataFrame(list(zip(df.index, labels)), columns=['N° Frasi', 'cluster'])
    result = {'cluster': labels, 'testo': sentences}
    result = pd.DataFrame(result)
    for k in range(0, true_k):
        s = result[result.cluster == k]
        text = s['testo'].str.cat(sep=' ')
        text = text.lower()
        text = ' '.join([word for word in text.split()])
        wordcloud = WordCloud(max_font_size=75, max_words=100, background_color="black").generate(text)
        print('Cluster: {}'.format(k))
        print('N° Frasi')
        titles = cl[cl.cluster == k]['N° Frasi']
        print(titles.to_string(index=False))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    os.makedirs('Videos')
    extract_video_from_url()
    extract_audio_from_video()
    slicing_audio()
    DIRNAME = r'Videos/Audios'
    OUTPUTFILE = r'Videos/Audios/recognized.csv'
    files = get_file_paths(DIRNAME)
    for file in files:
        (filepath, ext) = os.path.splitext(file)
        file_name = os.path.basename(file)
        if ext == '.wav':
            a = recog_multiple(file)
            with open('Videos/Audios/recognized.csv', 'a', newline='') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_NONE)
                writer.writerows(a)
    df = Get_DF()

    analyzer = SentimentIntensityAnalyzer()
    df["Negative"], df["Neutral"], df["Positive"], df["Vader Sentiment"], df["Sentiment"] = "", "", "", "", ""
    sentences = []

    df_mr = pd.read_csv('movie_review.csv')  # addestriamo un sentiment analyzer
    X, y = df_mr['text'], df_mr['tag']
    random.seed(10)
    vect = CountVectorizer(ngram_range=(1, 2))
    X = vect.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = BernoulliNB()
    model.fit(X_train, y_train)
    p_test = model.predict(X_test)
    errori = accuracy_score(y_test, p_test)

    for i in range(1, (df.shape[0] + 1)):
        sentence = [",".join(df["Testo"][i]).replace(",", " ")]
        sentences.append(str(sentence))
        q = array(sentence)
        q = vect.transform(q)
        analyzer.polarity_scores(sentence)
        df["Negative"][i] = (analyzer.polarity_scores(sentence)["neg"])
        df["Neutral"][i] = (analyzer.polarity_scores(sentence)["neu"])
        df["Positive"][i] = (analyzer.polarity_scores(sentence)["pos"])
        if df["Positive"][i] > df["Neutral"][i] and df["Positive"][i] > df["Negative"][i]:
            df["Vader Sentiment"][i] = "Positive"
        elif df["Neutral"][i] > df["Positive"][i] and df["Neutral"][i] > df["Negative"][i]:
            df["Vader Sentiment"][i] = "Neutral"
        else:
            df["Vader Sentiment"][i] = "Negative"
        df["Sentiment"][i] = str(model.predict(q))
        with ExcelWriter("Dataframe.xlsx") as writer:
            df.to_excel(writer)

    # TOPIC MODELING
    no_topics = 10
    no_features = 1000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(sentences)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    display_topics(lda, tf_feature_names, 5)

    print("I DUE MODELLI: \n")
    # MODELLO BERNOULLIANO
    print("MODELLO BERNOULLIANO")
    random.seed(10)
    X, y = sentences, df['Sentiment']
    vect = CountVectorizer(ngram_range=(1, 2))
    X = vect.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = BernoulliNB()
    model.fit(X_train, y_train)
    p_test = model.predict(X_test)
    print("Accuratezza del modello: ", accuracy_score(y_test, p_test))
    print("Tasso di errata classificazione: ", 1 - (accuracy_score(y_test, p_test)))
    print("\n")

    # RANDOM FOREST
    print("RANDOM FOREST")
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Matrice di confusione: \n", confusion_matrix(y_test, y_pred))
    print("Report di classificazione: \n", classification_report(y_test, y_pred))
    print("Accuratezza del modello Random Forest: ", accuracy_score(y_test, y_pred))
    print("Tasso di errata classificazione del modello Random Forest: ", 1 - (accuracy_score(y_test, y_pred)))
    print("\n")

    # CLUSTER ANALYSIS K-MEANS
    cluster_analysis(sentences)

    # CREAZIONE BAR PLOT FREQUENZE SENTIMENT
    plt.figure()
    df.plot.bar()
    plt.ylabel("Percentuale di Sentiment rilevato")
    plt.xlabel("Numero riga Dataframe")
    plt.title('Composizione del Sentiment per riga')
    plt.show()

    # WORDCLOUD DEL TESTO TOTALE
    allWords = ' '.join([str(word) for word in df['Testo']])
    wordCloud = WordCloud(width=600, height=400, random_state=21, max_font_size=110, max_words=50).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
