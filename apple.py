from msilib.schema import Environment
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
    st.markdown("<h1 style='text-align: center; color: black;'>What do you think of APPLE ?</h1>", unsafe_allow_html=True)
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

    nav = st.sidebar.radio("Project Data Visulisation",['The Project','Iphone14','Apple','Les Deux PDGs','Lets test your Data'])
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
        st.markdown("<h1 style='text-align: center; color: black;'>What do you think of the new Iphone 14 </h1>", unsafe_allow_html=True)
        st.write("##")
        
        st.write("Sentiment Score on Iphone 14 ")
        dfg = logo.groupby(['Date'])['compound'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
             xlabel='Period', figsize=(6, 5))

        st.bar_chart(dfg)
        
        st.write("#")
        st.write("Mean of each sentimal indicator for the Iphone14")
        Positif14 = logo['pos'].mean()
        st.write(Positif14)
        Negatif14 = logo['neg'].mean()
        st.write(Negatif14)
        Neutre14 = logo['neu'].mean()
        st.write(Neutre14)
        
        Compound14 = logo['compound'].mean()
        st.write(Compound14)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [Positif14 * 100 ,Negatif14*100, Neutre14*100,Compound14*100]
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
        st.write("Le nombre de Like par rapport aux commentaires positifs/negatifs")
        dfg = apple2006.groupby(['Like'])['compound'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
            xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)
        
        dfg = apple2021_2.groupby(['Like'])['compound'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
            xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)
        
        st.write("##")
        st.write("Le nombre de personnes qui ont mis des commentaires sous une publication négatives") 
        
        fig = px.line( 
            apple2006,#DataFrame
            x='Comment',
            y='neg',
            title="Number of commentary under negetive post "
        )
        st.plotly_chart(fig)
        
        fig2 = px.line( 
            apple2021_2,#DataFrame
            x='Comment',
            y='neg',
            title="Number of commentary under negetive post "
        )
        st.plotly_chart(fig2)
        
        fig3 = px.line( 
            apple2021_2,#DataFrame
            x='Comment',
            y='pos',
            title="Number of commentary under negetive post "
        )
        st.plotly_chart(fig3)
        
        st.write("##")
        st.write("Le nombre de personne ayant mis des commentaires positifs/negatifs")
        
   
        st.line_chart(apple2006['pos'])
        st.line_chart(apple2006['neg'])
        st.line_chart(apple2006['compound'])
        
        st.line_chart(apple2021_2['pos'])
        st.line_chart(apple2021_2['neg'])
        st.line_chart(apple2021_2['compound'])
        
        st.write("##")
        st.write("Les nombres de Tweets positfs les plus likes derrières les Tweet d'Apple les populaires") 
        
        st.write(applepop)
        Like = alt.Chart(applepop).mark_circle().encode(
            x='Like', y='compound', size='Like', tooltip=['compound', 'Like'])

        st.altair_chart(Like, use_container_width=True)
        
        RT = alt.Chart(applepop).mark_circle().encode(
            x='RT', y='compound', size='RT', tooltip=['compound', 'RT'])

        st.altair_chart(RT, use_container_width=True)
        
        Comment = alt.Chart(applepop).mark_circle().encode(
            x='Comment', y='compound', size='Comment', tooltip=['compound', 'Comment'])

        st.altair_chart(Comment, use_container_width=True)
    
    if nav == "Les Deux PDGs":
        
        path_timcook = "timcook.csv"
        applecook = load_data(path_timcook)
        
        path_jobs = "stevejobs.csv"
        applejobs = load_data(path_jobs)
        
        st.write("##")
        st.write("Qui est le plus aimé ?")
        
        st.bar_chart(applecook['compound'])
        st.bar_chart(applejobs['compound'])
        
        
        
        Like = alt.Chart(applecook).mark_circle().encode(
            x='Like', y='pos', size='Like', tooltip=['compound', 'Like'])
        st.altair_chart(Like, use_container_width=True)
        
        Like = alt.Chart(applejobs).mark_circle().encode(
            x='Like', y='pos', size='Like', tooltip=['compound', 'Like'])
        st.altair_chart(Like, use_container_width=True)
        
        st.write('Sont-ils bien perçus par leurs clients')
        
        # Define the base time-series chart.
        
        frames = [applecook, applejobs]
        
        result = pd.concat(frames)
        st.write(result)
        
        #The plot
        fig8 = px.line(
            result, #Data Frame
            x = "Date", #Columns from the data frame
            y = "Like",
            title = "Line frame"
        )
        fig8.update_traces(line_color = "maroon")
        st.plotly_chart(fig8)
        
        result2 = result.loc[result['Like'] == 261294]                       # Get rows with particular value
        st.write(result2)
        
        result3 = result.loc[result['Like'] == 264541]                       # Get rows with particular value
        st.write(result3)
        
        result4 = result.loc[result['Like'] == 217139]                       # Get rows with particular value
        st.write(result4)
        st.write('#')
        
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
        PositifM = applecook['pos'].mean()
        st.write(PositifM)
        NegatifM = applecook['neg'].mean()
        st.write(NegatifM)
        NeutreM = applecook['neu'].mean()
        st.write(NeutreM)
        
        CompoundM = applecook['compound'].mean()
        st.write(CompoundM)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [PositifM * 100 ,NegatifM*100, NeutreM*100,CompoundM*100]

        #The plot
        fig = go.Figure(
            go.Pie(
            labels = countries,
            values = values,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        st.header("Pie chart")
        st.plotly_chart(fig)
        
    if nav == 'Lets test your Data':
        st.write("Lets write want you want to search in Twitter ")
        query1 = st.text_input('Movie title', 'Wrote your research here')
        st.write('##')
        st.write('The word you want to search is', query1)
        st.write('This is data set : ')
        
        new = analyse(query1)
        st.write(new)
        
        
        
        st.write("DEBUT D'ANALYSE DE VOTRE RECHERCHE")
        
        dfg = new.groupby(['Date'])['compound'].mean()
        dfg.plot( title='Sentiment Score', ylabel='Mean Sentiment Score',
        xlabel='Period', figsize=(6, 5))
        st.bar_chart(dfg)
        
        
        PositifS = new['pos'].mean()
        st.write(PositifS)
        NegatifS = new['neg'].mean()
        st.write(NegatifS)
        NeutreS = new['neu'].mean()
        st.write(NeutreS)
        
        CompoundS = new['compound'].mean()
        st.write(CompoundS)
        
        countries=['Positif', 'Negatif',
           'Neut', 'Moyenne']
 
        values = [PositifS * 100 ,NegatifS*100, NeutreS*100,CompoundS*100]

        #The plot
        fig = go.Figure(
            go.Pie(
            labels = countries,
            values = values,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        st.header("Pie chart")
        st.plotly_chart(fig)
        
        
        
        
        

if __name__ == "__main__":
    main()