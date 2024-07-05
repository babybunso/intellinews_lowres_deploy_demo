__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import spacy
import spacy_streamlit
from wordcloud import WordCloud
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


EMBEDDING_MODEL = 'text-embedding-3-large'  # 'text-embedding-3-small'
SPACY_MODEL = spacy.load(os.path.join(os.getcwd(), 'en_core_web_sm/en_core_web_sm-3.7.1')) # 'en_core_web_lg'
ENTITY_LABELS = ['PERSON', 'EVENT', 'DATE', 'GPE', 'ORG', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL']
CANDIDATE_LABELS = ['Economic Growth', 'Healthcare Reform', 'Education Initiatives', 'Infrastructure Development', 'Environmental Policies', 'Agricultural Support', 'Employment and Labor', 'Social Welfare Programs', 'Foreign Relations', 'Public Safety and Security']
SUPPORTED_LANGUAGES_T = ['Finnish', 'Tagalog', 'Cebuano', 'Ilocano', 'Hiligaynon']
TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag kapwa karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung".split())
APP_NAME = 'SENTINEL: Semantic Evaluation and Natural Text Intelligence Learning System'
APP_DESC = ' `by @Team IntelliNews`'
ABOUT_SENTINEL_1 = """SENTINEL is a powerful document analysis and comparison tool, driven by cutting-edge Large Language Models (LLMs) and advanced Natural Language Processing (NLP) technologies. It excels in conducting semantic evaluations to uncover similarities, differences, and nuanced relationships within textual data. Whether analyzing documents for investigative journalism, content comparison, or extracting key insights, SENTINEL delivers precise and actionable results."""
ABOUT_SENTINEL_2 = """Ideal for newsrooms and investigative journalists, SENTINEL enhances research capabilities by swiftly identifying patterns, sentiments, and critical information buried within extensive text corpora. Its intelligent learning system continuously refines accuracy, ensuring reliable and efficient analysis across diverse document types and sources. SENTINEL empowers users to uncover hidden connections and trends, making it an indispensable tool for driving informed decisions and impactful storytelling."""
K = 10
APP_DOCS = os.path.join(os.getcwd(), 'documents')
DF_CSV = 'intellinews.csv'
DB_PATH = "intellinews.db"
COLLECTION_NAME = "intellinews"


# load_dotenv()
# OPENAI_APIKEY = os.environ['OPENAI_APIKEY']
OPENAI_APIKEY = st.secrets['OPENAI_APIKEY']
def get_openai_client():
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client
#######################################################################################################

def init_chroma_db(collection_name, db_path=DB_PATH):
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name=EMBEDDING_MODEL)

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection
#######################################################################################################

def semantic_search(Q, k=K, collection=None,  titles=[]):
    n_K = len(titles) * k
    results = collection.query(
        query_texts=[Q], # Chroma will embed this for you
        n_results=n_K, # how many results to return,
        where={ 'title': {'$in': titles} }
    )
    return results

# def semantic_search(Q, k=5, collection=None):
#     # Query the collection
#     results = collection.query(
#         query_texts=[Q], # Chroma will embed this for you
#         n_results=k # how many results to return
#     )
#     return results
#######################################################################################################

def upsert_documents_to_collection(collection, documents):
    # Every document needs an id for Chroma
    last_idx = len(collection.get()['ids'])
    ids = list(f'id_{idx+last_idx:010d}' for idx, _ in enumerate(documents))
    docs = list(map(lambda x: x.page_content, documents))
    mets = list(map(lambda x: x.metadata, documents))

    # Update/Insert some text documents to the db collection
    collection.upsert(ids=ids, documents=docs,  metadatas=mets)
#######################################################################################################


def generate_response(task, prompt, llm):
    response = llm.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': f"Perform the specified task: {task}"},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content
#######################################################################################################

def generate_summarization(doc, llm):
    task = 'Text Summarization'
    prompt = f"Summarize this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_translation(doc, target_lang, llm):
    task = 'Text Translation'
    prompt = f"Translate this document from English to {target_lang}:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_topic_labels(doc, llm, top_k=K):
    task = 'Topic Modeling or keyword extraction'
    prompt = f"Extract and list the top {top_k} main keywords in this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_sentiment_analysis(doc, llm):
    task = 'Sentiment Analysis'
    prompt = f"Classify the sentiment analysis of this document:\n\n{doc}\n\n\n Use labels: Positive, Negative, Neutral, Mixed"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_document_analysis(Q, df, llm):
    task = 'Document analysis and comparison'

    titles = df['title'].to_list()
    documents = df['documents'].to_list()
    doc_input = ''
    for i in range(len(df)):
        doc_input += f"""
        Document {i} Title: {titles[i]}
        Document {i} Content: {documents[i]}
        """

    prompt = f"""
        You are an unbias, fair, honest, intelligent, and an expert jounalist-researcher that is very knowledgeable in different domain of expertise encompassing investigative journalism.
        You are not representing any party or organization and you would treat the documents as research materials for intelligent context searching for you to be able to report the similiraties and differences of what has been written in those documents.
        Your main task is to compare sets of documents that discuss several topics.

        Given these documents, you are tasked to compare and contrast the key points of each document relative to the research question: '{Q}':

        To accomplish this, you would first list down key points based on the given research question as these key points will serve as the context of queries that you would search in each of research document in this list:
        {doc_input}.

        Then, for each keypoint item relative to the search result that you have found given the same context, it is important to describe they're differences and similarities in terms of how was it align to its original context. If no similar context found, just note that keypoint was not found in the document but still include the keypoint in the item list.
        You would highlight the major keypoints in terms of statistics, action points, and policies.
        Finally, provide a brief explanation for each keypoint. Make sure that no keypoint is duplicated, no important keypoint is missed and that the summary is concise and informative.

        Likewise, for each keypoint item, you would include a reference to a phrase where you have found the keypoint.

        More importantly, you to always provide a final summary of the results from your findings where in you would highlight the overall similarities and differences of each keypoint and make a final recommendation or action items as necessary.


        The final output should be in the following markdown format:


            Title: Title

            Executive Summary: Executive Summary

            Keypoints:
                Keypoint 1 Title: Keypoint 1 Title
                    Context: Context Summary
                    Give Context About Descriptive Statistics if available
                    Give Context About Policies if available
                        Policies Context: Identify which group of people will benefit and affected from the policy.
                    Similarities: Similarities
                    Differences: Differences
                    Reference: Reference phrase

                Keypoint 2 Title: Keypoint 2 Title
                    Context: Context Summary
                    Give Context About Descriptive Statistics if available
                    Give Context About Policies if available
                        Policies Context: Identify which group of people will benefit and affected from the policy.
                    Similarities: Similarities
                    Differences: Differences
                    Reference: Reference phrase

                ...

                Keypoint N Title: Keypoint N Title
                    Context: Context Summary
                    Give Context About Descriptive Statistics if available
                    Give Context About Policies if available
                        Policies Context: Identify which group of people will benefit and affected from the policy.
                    Similarities: Similarities
                    Differences: Differences
                    Reference: Reference phrase

            Recommendations: Recommendations / Action items
        """

    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def plot_wordcloud(df, column):
    # Data with filled of additonal stop words
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))

    # Fit vectorizers
    count_vectorizer = CountVectorizer(stop_words=my_stop_words)
    cv_matrix = count_vectorizer.fit_transform(df[column])

    tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

    # Create dictionaries for word cloud
    count_dict = dict(zip(count_vectorizer.get_feature_names_out(),
                                cv_matrix.toarray().sum(axis=0)))

    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(),
                                tfidf_matrix.toarray().sum(axis=0)))

    # Create word cloud and word frequency visualization
    count_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(count_dict))

    tfidf_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(tfidf_dict))

    # Plot the word clouds and word frequency visualizations
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(count_wordcloud, interpolation='bilinear')
    plt.title('Count Vectorizer Word Cloud')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tfidf_wordcloud, interpolation='bilinear')
    plt.title('TF-IDF Vectorizer Word Cloud')
    plt.axis("off")

    plt.tight_layout()
    plt.show();
    return fig
#######################################################################################################

# @st.cache_data()
def init_data():
    df = pd.DataFrame(columns=['url', 'title', 'speech'])
    try:
        df = pd.read_csv(DF_CSV)
    except:
        pass
    return df
#######################################################################################################

def save_uploadedfile(uploadedfile):
    if not os.path.exists(APP_DOCS):
        os.makedirs(APP_DOCS)
    file_path = uploadedfile.name

    with open(os.path.join(APP_DOCS, file_path), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
#######################################################################################################

# Create a Streamlit app
st.set_page_config(layout="wide", page_icon='📰', page_title=APP_NAME)
st.title(APP_NAME)
st.write(APP_DESC)

# Initialize chroma db
collection = init_chroma_db(collection_name=COLLECTION_NAME, db_path=DB_PATH)

# Initialize OpenAI client
llm = get_openai_client()

# Load the dataset
df = init_data()



options = st.sidebar.radio("", ["🏠 Home", "📊 The Dataset", "📚 Document Analysis"])
if options == "🏠 Home":
    st.write('___')
    st.subheader("About")
    st.write(ABOUT_SENTINEL_1)
    st.write(ABOUT_SENTINEL_2)
    st.write('___')
    st.caption("For more information, visit our GitHub repository: [SENTINEL](github.com/Team-IntelliNews/SENTINEL)")

if options == "📊 The Dataset":
    st.write('___')
    st.write("##### Upload a document to add to the dataset:")
    pdf = st.file_uploader('Upload PDF files', type=['pdf'], label_visibility='hidden')
    btn_upload = st.button("Upload")
    if pdf and btn_upload:
        file_path = save_uploadedfile(pdf)
        st.success(f"File uploaded successfully: {file_path}")
        loader = PyPDFLoader(f'{APP_DOCS}/{file_path}')
        docs = loader.load_and_split()

        metadata = {'url':f'file://{file_path}', 'title':file_path}

        # for dataframe
        doc_input = ''
        for i_tmp, doc in enumerate(docs):
            doc_input += str(doc.page_content)
            doc.metadata = metadata
        upsert_documents_to_collection(collection, docs) # NOTE: Run Once to insert the documents to the vector database
        new_df = pd.DataFrame([{'url': f'file://{file_path}', 'title': file_path, 'speech': doc_input}])
        df = init_data()
        df = pd.concat([df, new_df], axis=0).reset_index(drop=True)
        df.to_csv(DF_CSV, index=False)

    st.write("___")
    df = init_data()
    c1, c2 = st.columns([2, 1])
    c2.subheader("Word Count:")
    c2.write(f"{df['speech'].apply(lambda x: len(x.split())).sum(): ,}")
    c1.subheader("The Dataset:")
    c1.write(f"The dataset contains {len(df)} documents.")
    display_df = df.rename(columns={'speech': 'content'}).copy()
    st.dataframe(display_df, height=750, width=1400)

if options == "📚 Document Analysis":
    if len(df) < 2:
        st.error('Please upload at least two documents in the "📊 The Dataset" page to start the comparison analysis.')
    else:
        with st.form(key='query_form'):
            Q = ''
            QA = ''
            QT = []
            QDOCS = []
            cf1, _, cf2 = st.columns([11, 1, 4])
            with cf1:
                QA = st.text_area("Ask a Question:", placeholder="Type your question here...", height=100, max_chars=5000)
                _, center, _ = st.columns([5, 1, 5])
                center.subheader('OR')
                QT = st.multiselect("Select a Topic:", CANDIDATE_LABELS)
                st.markdown('<div></div>', unsafe_allow_html=True)
                _, center, _ = st.columns(3)
                center.subheader('FROM DOCUMENTS')
                QDOCS = st.multiselect("Select Document(s):", df['title'].unique(), max_selections=5)
            with cf2:
                st.markdown('<div></div>', unsafe_allow_html=True)
                st.write("###### Output Options:")
                st.markdown('<div></div>', unsafe_allow_html=True)
                st.write("Select Translation Language(s):")
                for lang in SUPPORTED_LANGUAGES_T:
                    st.checkbox(lang, value=True, key=f'chk_{lang.lower()}')
                st.markdown('<div></div>', unsafe_allow_html=True)
                st.markdown('<div></div>', unsafe_allow_html=True)
                st.markdown('<div></div>', unsafe_allow_html=True)
                K = st.number_input('Number of Results(k) per Document:', min_value=5, max_value=50, value=K, step=5)

            if len(QT) > 0:
                Q = ', '.join(QT)
                QA = ''
            else:
                Q = QA

            btn_ask = st.form_submit_button("Analyze Documents")

        if btn_ask and Q.strip() != '':
            if len(QDOCS) <= 1:
                st.error("Please select at least two documents for comparison.")
            else:
                # Semantic Search Results
                results = semantic_search(Q, k=K, collection=collection, titles=QDOCS)

                # Inspect Results
                data_dict = {
                    'ids': results['ids'][0],
                    'distances': results['distances'][0],
                    'documents': results['documents'][0],
                    'title': [eval(str(m))['title'] for m in results['metadatas'][0]],
                    'url': [eval(str(m))['url'] for m in results['metadatas'][0]],
                    'metadata': results['metadatas'][0]
                }

                results_df = pd.DataFrame(data_dict)
                with st.expander("Semantic Data Analysis:", expanded=True):
                    st.subheader('Query:')
                    st.write(Q)
                    st.subheader(f'Sources({results_df["title"].nunique()}):')
                    st.write('; '.join(results_df['title'].unique()))
                    st.subheader(f'Semantic Search Results Data (k={len(results_df)}):')
                    st.dataframe(results_df)
                    st.subheader('Word Clouds:')
                    st.pyplot(plot_wordcloud(results_df, 'documents'))


                cols = st.columns(results_df['title'].nunique())
                unique_titles = results_df['title'].unique()

                for i in range(len(cols)):
                    with cols[i]:
                        title = unique_titles[i]
                        tmp_df = results_df[results_df['title'] == title]
                        source = ''
                        text = ''

                        for x in range(tmp_df.shape[0]):
                            source = f"Source: {tmp_df['url'].iloc[x]}"
                            text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'

                        st.header(title)
                        st.write(f"Document Result Index: {i}")
                        st.caption(f"Source: {results_df['url'].iloc[i]}")
                        st.write('___')

                        st.subheader('Summary *(English)*:')
                        summary = generate_summarization(text, llm)
                        st.write(summary)

                        for lang in SUPPORTED_LANGUAGES_T:
                            if st.session_state[f'chk_{lang.lower()}']:
                                st.subheader(f'Summary: *({lang})*')
                                st.write(generate_translation(summary, lang, llm))
                                st.write('___')

                        st.subheader('Sentiment Analysis:')
                        st.write(generate_sentiment_analysis(text, llm))
                        st.write('___')

                        st.subheader('Keywords:')
                        st.write(generate_topic_labels(text, llm, top_k=10))
                        st.write('___')

                        st.subheader('Semantic Search Results *(Full-Text)*:')
                        doc = SPACY_MODEL(text)
                        spacy_streamlit.visualize_ner(
                            doc,
                            labels = ENTITY_LABELS,
                            show_table = False,
                            title = '',
                            key=f'ner{i}'
                        )

                document_analysis = generate_document_analysis(Q, results_df, llm)
                st.write('___')
                st.header('SENTINEL Document Analysis:')
                st.write(document_analysis.replace('```markdown', '').replace('```', ''))
                # st.markdown(document_analysis)
                st.write('___')


