import base64
import time
from matplotlib import pyplot as plt
import streamlit as st
import os
from wordcloud import WordCloud
from functions import category_count_graph, authenticate_client, sentiment_analyzer_genai
from PIL import Image
import pandas as pd
import seaborn as sns
import plotly.express as px

from prompt import neg_summarizer, pos_summarizer

# Setting up page configuartion
logo=Image.open("logo.png")
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="collapsed",
)

## Setting up the page indents
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, 
        unsafe_allow_html=True
        )

## Setting up the background
background_image ="paymeclip.png"
def set_bg_hack(main_bg):

    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
set_bg_hack(background_image)    

## Hiding the default footer
st.markdown("""
<style>
            
            footer {visibility: hidden;}
</style>
            """
            ,unsafe_allow_html=True)

## Setting up the PayMe logo in toolbar
def sidebar_bg(side_bg):

   img_ext = 'png'

   st.markdown(
      f"""
      <style>
      header.css-18ni7ap.e13qjvis2 {{
          background: url(data:image/{img_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: cover
      }}
      </style>
""",
      unsafe_allow_html=True,
      )

   
sidebar_bg("payMe-logo-header.png")

## Page Title
st.write("<div style='font-size: 42px; font-family: Univers;'><b>Sentiment Analyzer-GenAI</b>", unsafe_allow_html=True)

## File Uploader
uploaded_file = st.file_uploader("", type=["xlsx","csv"])

## Placeholder for clearing screen
task_message = st.empty()

## Custom styling for Submit button
custom_css = """
<style>
div.stButton > button:first-child {
    background-color: #df1c2b;
    color: white
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

## Process after uploading the file and clicking on Submit button
if task_message.button("Submit"):    
    if uploaded_file is not None:
        # To show the progress
        progress_bar = st.progress(0)
        # Reading the data from uploaded file
        data_df = pd.read_excel(uploaded_file)

        task_message.write("File successfully uploaded!")
        time.sleep(2)  
        progress_bar.progress(20)

        # Selecting column for analyzation
        for column in data_df.columns:
            if 'comment' in column.lower() or 'text' in column.lower()  or 'reviews' in column.lower() or 'review' in column.lower():
                col=column
        
        task_message.write("Analyzing data...")
        time.sleep(3)  
        progress_bar.progress(50)

        # Calling the GenAI function to analyze sentiment, confidence score and category
        sentiments_list,confidence,topic= sentiment_analyzer_genai(data_df[col])

        task_message.write("Almost there...")
        time.sleep(2)  
        progress_bar.progress(70)

        data_df = data_df.reset_index(drop=True)

        # Creating columns in dataframe
        data_df['Sentiment'] =sentiments_list
        data_df["Confidence Score"]= confidence
        data_df["Topic"] = topic

        time.sleep(2)
        progress_bar.progress(90)

        # Saving the dataframe into a csv file
        data_as_csv= data_df.to_csv(index=False).encode("utf-8")

        task_message.write("Process complete! You may download the generated file")

        progress_bar.progress(100)

        # Download button
        st.download_button(
                            label="Download",
                            data=data_as_csv,
                            file_name=os.path.splitext(uploaded_file.name)[0]+".csv",
                            key="download_button"
                        )
        

                                                   ### Dashboard #####

        # Title
        st.write("<div style='font-size: 30px;color:#df1c2b; text-align: center; font-family: UniversNextforHSBC-Bold,MHei HK-Heavy,MHei PRC-Heavy;;'><b>Dashboard</b>",unsafe_allow_html=True)
        st.write('')

        # TOTAL RECORDS, POSITIVE COUNT AND NEGATIVE COUNT DISPLAY
        sentiment_counts = len(data_df['Sentiment']) 
        positive_rev=data_df['Sentiment'].value_counts()["Positive"]
        negative_rev=data_df['Sentiment'].value_counts()["Negative"]
        col1,col2,col3,col4,col5= st.columns([2,1,1,1,2])
        # Display the total count in a box with text
        col2.markdown(
            f"""
            <div style="background-color: #F9E9E5 ; text-align:center; color: red ; padding: 20px; border-radius: 10px;">
                <h6>Total      Records</h6>
                <h2> {sentiment_counts}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
  
        col3.markdown(
            f"""
            <div style="background-color:#F9E9E5 ;text-align:center;color: red; padding: 20px; border-radius: 10px;">
                <h6>Positive Reviews</h6>
                <h2>{positive_rev}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col4.markdown(
            f"""
            <div style="background-color: #F9E9E5 ;text-align:center; color: red; padding: 20px; border-radius: 5px;">
                <h6>Negative Reviews</h6>
                <h2>{negative_rev}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write('\n')

        # Dividing the page into 3 columns for displaying graphs and charts
        col1,col2,col3=st.columns([1,1.25,1])
        col4,col5,col6= st.columns([1,1,1])
        
        # WORDCLOUD
        text = ' '.join(data_df['Topic'])
        colors = sns.color_palette("RdYlGn_r", n_colors=3)
        wordcloud = WordCloud(width=800, height=600, background_color='white').generate(text)
        fig_w, ax=plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        col1.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>Word cloud to identify topics/issues of interest:</b>",unsafe_allow_html=True)
        col1.image(wordcloud.to_array())

        # BAR CHART FOR RATINGS
        chart, ax = plt.subplots()
        ax.hist(data_df['Rating'], bins=5, edgecolor='k', alpha=0.7,color="goldenrod")
        ax.set_xlabel('Ratings')
        ax.set_ylabel('Frequency')
        ax.set_xticks([1, 2, 3, 4, 5], ['1 (Lowest)', '2', '3', '4', '5 (Highest)'])
        col4.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>PayMe Ratings Distribution</b>", unsafe_allow_html=True)
        col4.pyplot(chart)

      
        #SUMMARY
        positive_categories=[]
        negative_categories=[]
        for i in range(len(data_df)):
            if data_df['Sentiment'][i]=='Positive':
                positive_categories.append(data_df['Topic'][i])
            elif data_df['Sentiment'][i]=='Negative':
                negative_categories.append(data_df['Topic'][i])
        negative_summary=neg_summarizer(negative_categories)
        positive_summary=pos_summarizer(positive_categories)
        col3.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>Summarization of Analysis</b>",unsafe_allow_html=True)
        
        col3.write("<div style='font-size: 18px;color:Red; text-align: center; font-family: Arial;'><i>{}</i>".format(negative_summary), unsafe_allow_html=True)
        col3.write("<div style='font-size: 18px;color:Green; text-align: center; font-family: Arial;'><i>{}</i>".format(positive_summary), unsafe_allow_html=True)
        
        # HORIZONTAL BAR GRAPH FOR AVERAGE RATING BY CATEGORY
        sentiment_mapping = {
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0
                    }
        Sentiment_Numerical = data_df['Sentiment'].map(sentiment_mapping)
        word_ratings = []
        for _, row in data_df.iterrows():
            categories = row['Topic'].split(', ')
            rating = row['Rating']
            for category in categories:
                if category not in ['NA','']:
                    word_ratings.append((category, rating))

        # Converting the list of word ratings to a DataFrame
        word_df = pd.DataFrame(word_ratings, columns=['Word', 'Rating'])
        # Calculating the average rating for each word
        word_avg_ratings = word_df.groupby('Word')['Rating'].mean().reset_index()
        word_avg_ratings.drop_duplicates(inplace=True)
        # Sorting the words by average rating in descending order and getting the top 15 output
        word_avg_ratings = word_avg_ratings.sort_values(by='Rating', ascending=False)[:15]
        plt.figure(figsize=(5, 6))
        plt.barh(word_avg_ratings['Word'], word_avg_ratings['Rating'], color='goldenrod')
        plt.xlabel('Average Rating')
        plt.ylabel('Category Words')
        plt.gca().invert_yaxis() 

        col6.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>Average Customer Ratings by Category</b>", unsafe_allow_html=True)
        col6.pyplot(plt)

        # BOX PLOT FOR CONFIDENCE SCORE DISTRIBUTION FOR EACH SENTIMENT
        sentiment_order = ['Positive', 'Neutral', 'Negative']
        plt.figure(figsize=(7, 6))
        sns.boxplot(x='Sentiment', y='Confidence Score', data=data_df,palette=colors, order=sentiment_order)
    
        col5.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>Box plot of Sentiment and Confidence Score:</b>", unsafe_allow_html=True)
        col5.pyplot(plt)

        # Sunburst   
        data_df['Topic'] = data_df['Topic'].str.replace(r'\b(\w+)\.$', r'\1', regex=True)
        data_df['Topic'] = data_df['Topic'].str.split(', ')

        df = data_df.explode('Topic')
        df['Topic']= df['Topic'].str.title()

        sentiment_aspect_counts = df.groupby(['Sentiment', 'Topic']).size().reset_index(name='Count')
        colors = sns.color_palette("RdYlGn_r", n_colors=3)
        
        fig = px.sunburst(
            sentiment_aspect_counts,
            path=['Sentiment', 'Topic'],
            values='Count',
            color='Sentiment',
            color_discrete_map={'Positive': 'Green', 'Neutral': 'Goldenrod','Negative': 'Red'},
        )

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            width=500,
            height=300,
        )
        col2.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>PayMe Sentiment Analysis by Category</b>",unsafe_allow_html=True)
        col2.plotly_chart(fig)

        ## DISCLAIMER
        st.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: UniversNextforHSBC-Bold,MHei HK-Heavy,MHei PRC-Heavy;'><b>** The dataset employed comprises a mere 100 records, extracted from a 35,000+ feedback submissions received for PayMe-Hong Kong, with over a million downloads. This represents a fraction of approximately 0.29% of the total feedback, highlighting the substantial difference in sample size.\n It is essential to bear in mind that our analysis should not be solely reliant on this limited dataset, as it may not fully encapsulate the entire user experience. (less than 4%) </b>",unsafe_allow_html=True)

         # # STACKED BAR GRAPH FOR SENTIMENT DISTRIBUTION COUNT BY CATEGORY
        # reversed_cmap = plt.get_cmap("RdYlGn_r")
        # # Dictionary to store sentiment counts for each topic
        # sentiment_counts = {}
        # data = []
        # data_df['Topic'] = data_df['Topic'].fillna('NA')

        # for i in range(len(data_df)):
        #     sen = data_df['Sentiment'][i]
        #     top = data_df['Topic'][i].lower()
        #     if sen is not None and top is not None:
        #         data.append((sen, top))

        # for sentiment, topics in data:
        #     topics_list = topics.split(", ")
        #     for topic in topics_list:
        #         if topic in sentiment_counts:
        #             sentiment_counts[topic].append(sentiment)
        #         else:
        #             sentiment_counts[topic] = [sentiment]

        # # Function to assign numeric values to sentiments
        # def sentiment_to_value(sentiment):
        #     if sentiment == "Positive":
        #         return 0
        #     elif sentiment == "Neutral":
        #         return 1
        #     elif sentiment == "Negative":
        #         return 2

        # # Calculating the sum of sentiment counts for each topic
        # sum_counts = {topic: [0, 0, 0] for topic in sentiment_counts}
        # for topic, sentiments in sentiment_counts.items():
        #     for sentiment in sentiments:
        #         sum_counts[topic][sentiment_to_value(sentiment)] += 1

        # # Sorting the dictionary based on the sum of counts and the specified order, and getting top 15 values
        # sorted_counts = dict(sorted(sum_counts.items(), key=lambda item: (-sum(item[1]), item[1][0], item[1][1], item[1][2]))[:15])
        # category_count_graph(sorted_counts, sentiment_order,reversed_cmap)
        # # col1.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>PayMe Sentiment by Topic Distribution</b>",unsafe_allow_html=True)
        # # col1.pyplot(plt) 
    
        
          # PIE CHART FOR LABEL DISTRIBUTION
        # sentiment_counts = {
        #     "Positive": data_df['Sentiment'].value_counts()["Positive"],
        #     "Neutral": data_df['Sentiment'].value_counts()["Neutral"],
        #     "Negative": data_df['Sentiment'].value_counts()["Negative"],
        # }
        # labels = sentiment_counts.keys()
        # sizes = sentiment_counts.values()
        # explode = (0.05, 0, 0)
        # fig, ax = plt.subplots()
        # ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        # ax.axis('equal') 

        # col3.write("<div style='font-size: 18px;color:Black; text-align: center; font-family: Arial;'><b>PayMe Sentiment Labels Distribution:</b>", unsafe_allow_html=True)
        # col3.pyplot(fig)


        