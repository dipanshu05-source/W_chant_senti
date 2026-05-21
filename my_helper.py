import re
from wordcloud import WordCloud
import emoji
from collections import Counter
import pandas as pd
from urlextract import URLExtract
extract=URLExtract()


def fetch_stats(selected_user,df):
    
    if selected_user != 'Overall':
        df=df[df['user']==selected_user]
    
    
    num_messages=df.shape[0]
    words=[]
    for message in df['message']:
        words.extend(message.split())
    
    #fetch  no of media messages
    num_media_messages=df[df['message']=='<Media omitted>\n'].shape[0]
    
    #fetch links...
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    
    
    return num_messages,len(words),num_media_messages,len(links)
    


def most_active_users(df):
    x=df['user'].value_counts().head()
    df=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percent'})
    return x,df

#wordcloud
def create_wordcloud(selected_user, df):

    # stopwords load
    with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    # selected user filter
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # remove group notifications
    temp = df[df['user'] != 'group_notification']

    # remove media messages
    temp = temp[temp['message'] != '<Media omitted>\n']

    temp = temp.copy()

    def clean_message(message):

        # lowercase
        message = message.lower()

        # remove urls
        message = re.sub(r'http\S+|www\S+', '', message)

        # remove numbers
        message = re.sub(r'\d+', '', message)

        # remove punctuation/symbols/emojis
        message = re.sub(r'[^\w\s]', '', message)

        words = []

        for word in message.split():

            # remove short garbage words
            if len(word) < 3:
                continue

            # remove stopwords
            if word in stop_words:
                continue

            words.append(word)

        return " ".join(words)

    temp['message'] = temp['message'].apply(clean_message)

    text = temp['message'].str.cat(sep=" ")

    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white',
        collocations=False,
    )

    df_wc = wc.generate(text)

    return df_wc




    
    
def most_common_words(selected_user,df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    
    if selected_user != 'Overall':
        df=df[df['user']==selected_user]
        
    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
        
    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word) 
                
    most_common_df=pd.DataFrame(Counter(words).most_common(20))   
    return most_common_df            

def emoji_helper(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
        
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
            
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))        
    
    return emoji_df
        
def monthly_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
        
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
        
    timeline=df.groupby(['year','month_num','month']).count()['message'].reset_index()
    
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i])) 
    
    timeline['time']=time       
    return timeline        

def daily_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
        
    daily_timeline=df.groupby('only_date').count()['message'].reset_index() 
    
    return daily_timeline   
        
        
def week_activity_map(selected_user,df):
    
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    return df['day_name'].value_counts()    

def month_activity_map(selected_user,df):
    
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    return df['month'].value_counts()    

def activity_heat_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
        
    activity_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)    
    return activity_heatmap




import pickle

# Load model globally
with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def predict_sentiment(text):
    transformed = tfidf.transform([text])
    prediction = sentiment_model.predict(transformed)[0]
    mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return mapping[prediction]

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Remove media and notifications
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification']
    
    # Predict each message
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    # Count karo
    sentiment_counts = temp['sentiment'].value_counts()
    
    return sentiment_counts, temp


def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification'].copy()
    
    # Predict sentiment
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    # Month-Year column
    temp['month_year'] = temp['date'].dt.strftime('%b-%Y')
    temp['month_num'] = temp['date'].dt.month
    temp['year'] = temp['date'].dt.year
    
    # Group by month aur sentiment
    timeline = temp.groupby(
        ['year', 'month_num', 'month_year', 'sentiment']
    ).size().unstack(fill_value=0).reset_index()
    
    # Sort by date
    timeline = timeline.sort_values(['year', 'month_num'])
    timeline['month_year'] = timeline['month_year'].astype(str)
    
    return timeline


def most_toxic_day(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != 'This message was deleted\n']
    
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    negative_df = temp[temp['sentiment'] == 'Negative']
    
    toxic_days = negative_df.groupby('only_date').size().reset_index()
    toxic_days.columns = ['date', 'negative_count']
    toxic_days = toxic_days.sort_values('negative_count', ascending=False)
    
    return toxic_days

def most_positive_day(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != 'This message was deleted\n']
    
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    # Sirf positive messages
    positive_df = temp[temp['sentiment'] == 'Positive']
    
    # Per date count
    positive_days = positive_df.groupby('only_date').size().reset_index()
    positive_days.columns = ['date', 'positive_count']
    positive_days = positive_days.sort_values('positive_count', ascending=False)
    
    return positive_days





def happy_hours(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != 'This message was deleted\n']
    
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    # Sirf positive messages
    positive_df = temp[temp['sentiment'] == 'Positive']
    
    # Hour wise count
    happy_hour_counts = positive_df.groupby('hour').size().reset_index()
    happy_hour_counts.columns = ['hour', 'positive_count']
    
    # Full 24 hours ensure karo
    all_hours = pd.DataFrame({'hour': range(24)})
    happy_hour_counts = all_hours.merge(
        happy_hour_counts, on='hour', how='left'
    ).fillna(0)
    
    return happy_hour_counts


def sentiment_score(df):
    temp = df[df['message'] != '<Media omitted>\n']
    temp = temp[temp['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != 'This message was deleted\n']
    
    temp['sentiment'] = temp['message'].apply(predict_sentiment)
    
    # Per user sentiment count
    user_sentiment = temp.groupby(
        ['user', 'sentiment']
    ).size().unstack(fill_value=0).reset_index()
    
    # Columns ensure karo
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in user_sentiment.columns:
            user_sentiment[col] = 0
    
    user_sentiment['total'] = (
        user_sentiment['Positive'] +
        user_sentiment['Neutral'] +
        user_sentiment['Negative']
    )
    
    # Score formula - 0 to 100
    user_sentiment['sentiment_score'] = (
        (user_sentiment['Positive'] * 100 +
         user_sentiment['Neutral'] * 50) /
        user_sentiment['total']
    ).round(2)
    
    return user_sentiment.sort_values(
        'sentiment_score', ascending=False
    )