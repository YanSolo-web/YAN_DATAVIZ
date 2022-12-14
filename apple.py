import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import snscrape.modules.twitter as snstwitter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

header = st.container()
dataset = st.container()
exploration = st.container()

with header :
    st.title("PROJECT DATACAMP")
    st.markdown("<h1 style='text-align: center; color: black;'>What do you think of the E-reputation of APPLE ?</h1>", unsafe_allow_html=True)
    st.markdown('''
            <a>
                <img src="https://www.incimages.com/uploaded_files/image/apple-talking_466826.gif" width="800px" />
            </a>''',
                        unsafe_allow_html=True
                        )
    st.write('##')



# function to load my data fil
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data (path) :
    df=pd.read_csv(path)
    return df

# Creating a function for finding the day of the month, the hour of the day and the weekday
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_hour(dt):
    return dt.hour

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_dom (dt) :
    return dt.day

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_weekday(dt):
    return dt.weekday() 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_month(dt):
    return dt.month

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_year(dt):
    return dt.year

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def date_transformation (df):
    df['Date']= df['Date'].map(pd.to_datetime)
    return df



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def new_col(df) :
    df['hour']= df['Date'].map(get_hour)
    df['day']= df['Date'].map(get_dom)
    df['weekday']= df['Date'].map(get_weekday)
    df['month']= df['Date'].map(get_month)
    return df


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def analyse(query):
    tweets = []
    limit = 500
    for tweet in snstwitter.TwitterSearchScraper(query).get_items():
    #print(vars(tweet))
    #break
        if len(tweets)==limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content, tweet.replyCount, tweet.retweetCount, tweet.likeCount])
    iphone = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Comment', 'RT', 'Like'])
    res = {}
    sia = SentimentIntensityAnalyzer()
    for i, row in tqdm(iphone.iterrows(), total=len(iphone)):
        text = row['Tweet']
        MyUser = row['User']
        res[MyUser] = sia.polarity_scores(text)
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index' : 'Id'})
    vaders = vaders.merge(iphone, right_index=True,left_index=True)
    
    return vaders


def main():

    nav = st.sidebar.radio("Project Data Visulisation",['The Project','Iphone14','Apple','Two CEO','Lets test your Data'])
    if nav == "The Project":
        st.markdown("<h1 style='text-align: center; color: black;'>LISA SANGLAR AND YANNICK BIA2</h1>", unsafe_allow_html=True)
        st.write('##')
        '''  
        Our Linkedin Account'''
        st.markdown('''
            <a href="https://www.linkedin.com/in/yannick-pierre-marie-742605177/">
                <img src="https://i.ibb.co/XsqvvmB/LinkedIn.gif" width="150px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        st.markdown('''
            <a href="https://www.linkedin.com/in/lisa-sanglar-6a943719b/">
                <img src="https://i.ibb.co/XsqvvmB/LinkedIn.gif" width="150px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        st.write('##')
        ''' 
        Nowadays we're already know that Apple is one of most famous company in the world. His brand and his politics have been a real leader around the word of 
        digital and business.Let's have a look very closly about how Apple have been represent around the world . With his PDG , his band , his look and all the things we could question about it " 

        LET'S FIND OUT TODAY ! '''

    if nav == "Iphone14":
        path_logo ="iphone14.csv"
        logo = load_data(path_logo)
        logo = date_transformation(logo)
        logo = new_col(logo)
        st.write("##")
        st.markdown("<h1 style='text-align: center; color: black;'>What do you think of the new Iphone 14 ?</h1>", unsafe_allow_html=True)
        st.write("##")
        
        st.write("Sentiment Score on Iphone 14 ")
        st.write("##")
        st.write("x = Date bewteen September and October ")
        st.write("y = Compound of the tweet about the Iphone 14")
        dfg = logo.groupby(['Date'])['compound'].mean()
        dfg.plot(title='Sentiment Score', ylabel='Mean Sentiment Score',
             xlabel='Period', figsize=(6, 5))

        st.bar_chart(dfg)
        
        st.write("#")
        st.write("Mean of each sentimal indicator for the Iphone14")
        st.write("#")
        st.write("Positif sentimal indicator")
        Positif14 = logo['pos'].mean()
        st.write(Positif14)
        st.write("#")
        st.write("Negtif sentimal indicator")
        Negatif14 = logo['neg'].mean()
        st.write(Negatif14)
        st.write("#")
        st.write("Neutral sentimal indicator")
        Neutre14 = logo['neu'].mean()
        st.write(Neutre14)
        st.write("#")
        st.write("Compound sentimal indicator")
        Compound14 = logo['compound'].mean()
        st.write(Compound14)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [Positif14 * 100 ,Negatif14*100, Neutre14*100]
        st.write("##")
        #The plot
        fig = go.Figure(
            go.Pie(
            labels = countries,
            values = values,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        st.header("Pie chart of each sentimal indicateur of the Iphone 14")
        st.plotly_chart(fig)
        
    if nav == "Apple":
        st.write("##")
        st.markdown("<h1 style='text-align: center; color: black;'>What do you think of Apple ?</h1>", unsafe_allow_html=True)
        st.write("##")
        path_apple006 ="apple2006.csv"
        apple2006 = load_data(path_apple006)
        #apple2006 = date_transformation(apple2006)
        #apple2006 = new_col(apple2006)
        
        path_apple21_2 ="apple2021_2.csv"
        apple2021_2 = load_data(path_apple21_2)
        #apple2021_2 = date_transformation(apple2021_2)
        #apple2021_2 = new_col(apple2021_2)
        
        path_applePop = "applepopular.csv"
        applepop = load_data(path_applePop)
        
        st.write("##")
        st.write("Negative comment rate versus likes in 2006 ")
        dfg = apple2006.groupby(['Like'])['neg'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
            xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)
        
        st.write("##")
        st.write("Negative comment rate versus likes in 2021/2022")
        dfg = apple2021_2.groupby(['Like'])['neg'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
            xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)


        st.write("##")
        st.write("Postive comment rate versus likes in 2006")
        st.line_chart(apple2006['pos'])
        st.write("##")
        st.write("Negative comment rate versus likes in 2006")
        st.line_chart(apple2006['neg'])
        st.write("##")
        st.write("Compound comment rate versus likes in 2006")
        st.line_chart(apple2006['compound'])
        
        st.write("##")
        st.write("Positive comment rate versus likes in 2021/2")
        st.line_chart(apple2021_2['pos'])
        st.write("##")
        st.write("Negative comment rate versus likes in 2021/2")
        st.line_chart(apple2021_2['neg'])
        st.write("##")
        st.write("Compound comment rate versus likes in 2021/2")
        st.line_chart(apple2021_2['compound'])
        
        st.write("##")
        st.write("Table of the most popular tweet likes/comment /Retweet in 2022") 

        st.write(applepop)
        st.write("##")
        st.write("Compound of the most popular Apple Tweets versus Like ") 
        Like = alt.Chart(applepop).mark_circle().encode(
            x='Like', y='compound', size='Like', tooltip=['compound', 'Like'])

        st.altair_chart(Like, use_container_width=True)
        
        st.write("Compound of the most popular Apple Tweets versus RT ")
        RT = alt.Chart(applepop).mark_circle().encode(
            x='RT', y='compound', size='RT', tooltip=['compound', 'RT'])

        st.altair_chart(RT, use_container_width=True)
        st.write("Compound of the most popular Apple Tweets versus Comment ")
        Comment = alt.Chart(applepop).mark_circle().encode(
            x='Comment', y='compound', size='Comment', tooltip=['compound', 'Comment'])

        st.altair_chart(Comment, use_container_width=True)
    
    if nav == "Two CEO":
        
        st.write("##")
        st.markdown("<h1 style='text-align: center; color: black;'>What do you think of the new Iphone 14 ?</h1>", unsafe_allow_html=True)
        st.write("##")
        
        path_timcook = "timcook.csv"
        applecook = load_data(path_timcook)
        
        path_jobs = "stevejobs.csv"
        applejobs = load_data(path_jobs)
        
        st.write('##')
        st.write("Most popular tweet talked about Tim Cook in 2022 versus the compound")
        st.bar_chart(applecook['compound'])
        st.write("Most popular tweet talked about Steve Jobs in 2022 versus the compound")
        st.bar_chart(applejobs['compound'])
        
        
        st.write("Compound of the most popular Tweets about Cook versus Like")
        Like = alt.Chart(applecook).mark_circle().encode(
            x='Like', y='pos', size='Like', tooltip=['compound', 'Like'])
        st.altair_chart(Like, use_container_width=True)
        
        st.write("Compound of the most popular Tweets about Jobs versus Like")
        Like = alt.Chart(applejobs).mark_circle().encode(
            x='Like', y='pos', size='Like', tooltip=['compound', 'Like'])
        st.altair_chart(Like, use_container_width=True)
        
        st.write('In g??n??ral , what does their client think of this two CEO ? ')
        st.write("Table of impression of Tim Cook and Steve Jobs combine ")
        # Define the base time-series chart.
        
        frames = [applecook, applejobs]
        
        result = pd.concat(frames)
        st.write(result)
        
        #The plot
        
        fig8 = px.line(
            result, #Data Frame
            x = "Date", #Columns from the data frame
            y = "Like",
            title = "Most popular Like Tweets about Jobs and Cook in 2022 "
        )
        
        fig8.update_traces(line_color = "maroon")
        st.plotly_chart(fig8)
        
        st.write("Table of the three most like tweet about them ")
        result2 = result.loc[result['Like'] == 261294]                       # Get rows with particular value
        st.write(result2)
        
        result3 = result.loc[result['Like'] == 264541]                       # Get rows with particular value
        st.write(result3)
        
        result4 = result.loc[result['Like'] == 217139]                       # Get rows with particular value
        st.write(result4)
        st.write('#')
        
        st.write("Links to those tweet , click on the image below")
        st.markdown('''
            <a href="https://t.co/W7EoLu8O30">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Question_mark_alternate.svg/800px-Question_mark_alternate.svg.png" width="50px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        
        st.markdown('''
            <a href="https://twitter.com/andykreed/status/1568260953212616707">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Question_mark_alternate.svg/800px-Question_mark_alternate.svg.png" width="50px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        
        st.markdown('''
            <a href="https://t.co/MfijlzTKHI">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Question_mark_alternate.svg/800px-Question_mark_alternate.svg.png" width="50px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        
        st.write('#')
        st.write('#')
        st.write('#')
        
        
        #Likeable or not 
        st.write("#")
        st.write("Postive sentimal indicator")
        PositifM = applecook['pos'].mean()
        st.write(PositifM)
        st.write("#")
        st.write("Negative sentimal indicator")
        NegatifM = applecook['neg'].mean()
        st.write(NegatifM)
        st.write("#")
        st.write("Neutral sentimal indicator")
        NeutreM = applecook['neu'].mean()
        st.write(NeutreM)
        
        st.write("#")
        st.write("Compound sentimal indicator")
        CompoundM = applecook['compound'].mean()
        st.write(CompoundM)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [PositifM * 100 ,NegatifM*100, NeutreM*100]

        #The plot
        fig = go.Figure(
            go.Pie(
            labels = countries,
            values = values,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        st.header("Pie chart of each sentimal indicateur of Steve Jobs and Tim Cook")
        st.plotly_chart(fig)
        
    if nav == 'Lets test your Data':
        st.write("##")
        st.markdown("<h1 style='text-align: center; color: black;'>What do you think of your Data?</h1>", unsafe_allow_html=True)
        st.write("##")
        st.write("Lets write want you want to search in Twitter ")
        query1 = st.text_input('Tweet', 'Wrote your research here')
        st.write('##')
        st.write('The word you want to search is', query1)
        st.write('This is data set : ')
        
        new = analyse(query1)
        st.write(new)
        
        
        
        st.write("BEGINNIN OF THE FIRST PART OF YOUR SEARCH")
        
        st.write("Sentiment Score on",query1)
        st.write("##")
        st.write("x = Date bewteen September and October ")
        st.write("y = Compound of the tweet about",query1)
        dfg = new.groupby(['Date'])['compound'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
        xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)
        
        st.write("#")
        st.write("Positives sentimal indicator")
        PositifS = new['pos'].mean()
        st.write(PositifS)
        st.write("#")
        st.write("Negatives sentimal indicator")
        NegatifS = new['neg'].mean()
        st.write(NegatifS)
        st.write("#")
        st.write("Neutral sentimal indicator")
        NeutreS = new['neu'].mean()
        st.write(NeutreS)
        st.write("#")
        st.write("Compound sentimal indicator")
        CompoundS = new['compound'].mean()
        st.write(CompoundS)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [PositifS * 100 ,NegatifS*100, NeutreS*100]

        #The plot
        fig = go.Figure(
            go.Pie(
            labels = countries,
            values = values,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        st.header("Pie chart of each sentimal indicateur of your search")
        st.plotly_chart(fig)
        
        
        
        
        

if __name__ == "__main__":
    main()
