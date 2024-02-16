import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
st.header("Customer Review Analyysis")
# Model Selection
model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])

# Load the model
@st.cache_data
def load_model(model_select):
    if model_select == "SVC":
        sentiment_model = joblib.load(r'SVC_sentiment_model.sav')
    elif model_select == "Logistic Regression":
        sentiment_model = joblib.load(r'LR_sentiment_model.sav')
    elif model_select == "Naive Bayes":
        sentiment_model = joblib.load(r'NB_sentiment_model.sav')
    return sentiment_model

sentiment_model = load_model(model_select)

@st.cache_data
def load_vectorizer():
    vectorizer = joblib.load(r'tfidf_vectorizer_sentiment_model.sav')
    return vectorizer

vectorizer = load_vectorizer()
if st.checkbox("Please check this if you have a CSV available to be analysed", False, key=0):
    with st.spinner("Loading App..."):
        if st.sidebar.checkbox("Text Input", True, key=1):
            with st.expander('Analyze Text', expanded=False):
                st.subheader("Enter the statement that you want to analyze")

                text_input = st.text_input("Enter sentence")

                vec_inputs = vectorizer.transform([text_input])

                # Make the prediction
                if sentiment_model.predict(vec_inputs):
                    st.write("This statement is **Positve**")
                else:
                    st.write("This statement is **Negative**")
        if st.sidebar.checkbox("Csv Upload", True, key=2):
            with st.expander('Analyze CSV', expanded=True):
                # Upload a csv or excel file
                uploaded_file = st.file_uploader("Upload a csv or excel file", type=["csv", "xlsx"])
                if uploaded_file:
                    df = pd.read_excel(uploaded_file)
                    # Select the column to predict
                    col_to_predict = st.selectbox("Select the column to predict sentiment for", df.columns)

                    # Predict the sentiment
                    vec_inputs = vectorizer.transform(df[col_to_predict].astype(str))
                    sentiment = sentiment_model.predict(vec_inputs)

                    # Add a new column to the dataset
                    df['sentiment'] = sentiment
                    df['sentiment'] = df['sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative')



        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')


        if 'df' in locals():
            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )
        if uploaded_file:
            sentiments = df['sentiment'].value_counts()
            if 'sentiments' in locals() or 'sentiments' in globals():
                fig = px.pie(sentiments, values=sentiments.values, names=sentiments.index)
            else:
                print("")
            if st.checkbox("Show pie chart"):
                st.plotly_chart(fig)
            else:
                st.write("Pie chart is hidden")

            if st.sidebar.checkbox("Review Filter", True, key=3):

                if not uploaded_file:
                    st.write("Please upload the file")
                elif "df" not in locals():
                    st.write("Please upload the file")
                else:
                    # Apply the filter
                    filter_ = st.selectbox("Filter by Sentiment", ['All'] + df["sentiment"].unique().tolist())
                    if "filter_" not in locals():
                        st.write()
                    else:
                        if filter_ == 'All':
                            filtered_df = df
                        else:
                            filtered_df = df[df['sentiment'] == filter_]
                        st.write(filtered_df)

                if st.sidebar.checkbox("Show/hide Sentiment Summary", True, key=4):
                    with st.expander('Summary of reviews', expanded=True):
                        if "df" not in locals():
                            st.write("Please upload the file")
                        else:
                            pos = len(df[df["sentiment"] == "Positive"])
                            neg = len(df[df["sentiment"] == "Negative"])

                            st.sidebar.write("Sentiment Summary:")
                            st.sidebar.write(f"Total Reviews :", pos + neg)
                            st.sidebar.write(f"Positive:", pos)
                            st.sidebar.write(f"Negative:", neg)
if st.checkbox("Please check this if you have a URL from which you want to import review data from", False, key=5):
    with st.spinner("Loading App..."):
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import streamlit as st

        url = st.text_input("Enter product review URL:")
        page_num = int(st.number_input("Enter number of pages to scrape:", value=10))

        # Validate URL
        if not url.startswith("https://"):
            st.write("")
        else:
            data = []
            for i in range(1, page_num + 1):
                # URL setup and HTML request
                r = requests.get(url + '&pageNumber=' + str(i))
                soup = BeautifulSoup(r.text, 'html.parser')
                reviews = soup.find_all('div', {'data-hook': 'review'})

                for item in reviews:
                    review = {
                        'title': item.find('a', {'data-hook': 'review-title'}).text.strip() if item.find('a', {
                            'data-hook': 'review-title'}) else None,
                        'text': item.find('span', {'data-hook': 'review-body'}).text.strip() if item.find('span', {
                            'data-hook': 'review-body'}) else None,
                    }
                    if review['text'] is not None:
                        data.append(review)

            df = pd.DataFrame(data)
            if 'title' in df.columns:
                vec_inputs = vectorizer.transform(df['title'].astype(str))
            else:
                vec_inputs = vectorizer.transform(df['text'].astype(str))



            sentiment = sentiment_model.predict(vec_inputs)
            df['sentiment'] = sentiment
            df['sentiment'] = df['sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
            import plotly.express as px

            labels = df['sentiment'].value_counts().index
            values = df['sentiment'].value_counts().values
            fig = px.pie(names=labels, values=values)
            if st.checkbox("Show pie chart"):
                st.plotly_chart(fig)
            else:
                st.write("Pie chart is hidden")

            # Total number of reviews
            total_reviews = df.shape[0]
            st.sidebar.write("Total Reviews:", total_reviews)

            # Number of unique positive, negative, and neutral reviews
            positive_reviews = df[df['sentiment'] == "Positive"].shape[0]
            negative_reviews = df[df['sentiment'] == "Negative"].shape[0]

            st.sidebar.write("Positive Reviews:", positive_reviews)
            st.sidebar.write("Negative Reviews:", negative_reviews)

            # Add a filter to select sentiment label
            sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative"])

            if sentiment_filter == "Positive":
                df = df[df['sentiment'] == "Positive"]
            elif sentiment_filter == "Negative":
                df = df[df['sentiment'] == "Negative"]

            st.write(df)
            if st.button('Download as CSV'):
                st.write(df.to_csv(index=False))


