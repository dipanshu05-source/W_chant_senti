import streamlit as st
import my_helper
import seaborn as sns
from preprocessor import preprocess
import matplotlib.pyplot as plt
from wordcloud import WordCloud
plt.rcParams['font.family'] = 'Segoe UI Emoji'

import streamlit as st

st.markdown("""
<style>

/* Transparent top bar */
[data-testid="stHeader"] {
    background: transparent;
}

/* Remove top gap */
.block-container {
    padding-top: 1rem;
}

/* Main + Sidebar background */
.stApp, [data-testid="stSidebar"] {
    background: #0f2027;
    background-image:
        radial-gradient(rgba(255,255,255,0.08) 2px, transparent 2px);
    background-size: 40px 40px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.image("logo.png", width=1700)
#st.sidebar.title("Whatsapp Chant Analyzer")

st.image("good image.png")
st.title("Understanding the Whatsapp Text and Sentiments")

st.markdown(">For More Information and Manual refer to GUIDE on left sidebar ")


uploaded_file=st.sidebar.file_uploader("Please choose only '.txt' file")
if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()     #firstly the data was the stream of byte .. now we are converting this in utf-8
    data=bytes_data.decode("utf-8")
    df=preprocess(data)
    
    st.dataframe(df)
    
    #fetch unique user
    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")
    
    selected_user=st.sidebar.selectbox("Show analysis wrt",user_list)
    
    if st.sidebar.button("Show Chat Analysis"):
        num_messages,words,num_media_messages,num_links=my_helper.fetch_stats(selected_user,df)
        col1,col2,col3,col4=st.columns(4)
        
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words") 
            st.title(words)
        with col3:
            st.header("Media Shared") 
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared") 
            st.title(num_links)        
        
        
        #monthly timeline
        st.title("Monthly TimeLine")
        timeline=my_helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        
        
        
        # Daily timeline
        st.title("Daily TimeLine")
        daily_timeline=my_helper.daily_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(daily_timeline['only_date'],daily_timeline['message'],color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        
        #Activity Map
        st.title("Activity Map")
        col1,col2=st.columns(2)
        
        with col1:
            st.header("Most busy day")
            busy_day=my_helper.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
        with col2:
            st.header("Most busy Month")
            busy_month=my_helper.month_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)    
        
        #Heaat Map
        st.title("Weekly Activity MAP")
        user_heatmap=my_helper.activity_heat_map(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)
        
        #finding the busiest user in the group(Group Level)
        if selected_user=='Overall':
            st.title('Most Active Users')
            x,new_df=my_helper.most_active_users(df)
            fig, ax=plt.subplots()
            col1,col2=st.columns(2)  
            
            with col1:
                ax.bar(x.index,x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        
                    

        #form Word Cloud
        st.title("WORD CLOUD")
        df_wc=my_helper.create_wordcloud(selected_user,df)  
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)  
        
        
        # most common words
        most_common_df=my_helper.most_common_words(selected_user,df)    
        fig,ax=plt.subplots()
        ax.bar(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('MOST COMMON WORDS')
        st.pyplot(fig)
        st.dataframe(most_common_df)
        
        #emoji Analysis
        
        emoji_df=my_helper.emoji_helper(selected_user,df) 
        st.title("Emoji Analysis") 
        
        col1,col2=st.columns(2)   
        
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax=plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")  
            st.pyplot(fig)    
    
    
    
    
    
    
    
    
    
    
    
    if st.sidebar.button("Sentiment Analysis"):
    
        
        #SENTIMENT ANALYSIS
        st.title("Sentiment Analysis")
    
        sentiment_counts, sentiment_df = my_helper.sentiment_analysis(selected_user, df)
    
        # Overall pie chart
        col1, col2 = st.columns(2)
    
        with col1:
            st.header("Overall Sentiment")
            fig, ax = plt.subplots()
            colors = ['#95a5a6','#2ecc71' ,'#ff6b6b' ]  # grey, green, red
            ax.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                autopct="%0.1f%%",
                colors=colors
            )
            st.pyplot(fig)
    
        with col2:
            st.header("Sentiment Count")
            st.dataframe(sentiment_counts)
    
        # Per user sentiment (only overall mein)
        if selected_user == 'Overall':
            st.header("Sentiment Per User")
        
            user_sentiment = sentiment_df.groupby(
                ['user', 'sentiment']
            ).size().unstack(fill_value=0)
        
            fig, ax = plt.subplots(figsize=(10, 5))
            user_sentiment.plot(
                kind='bar',
                ax=ax,
                color=['#ff6b6b', '#95a5a6', '#2ecc71']
            )
            plt.xticks(rotation='vertical')
            plt.legend(loc='upper right')
            st.pyplot(fig)
        
            # Most positive aur most negative user
            col1, col2, col3 = st.columns(3)
        
            with col1:
                st.header("😊 Most Positive")
                if 'Positive' in user_sentiment.columns:
                    st.title(user_sentiment['Positive'].idxmax())
        
            with col2:
                st.header("😢 Most Negative")
                if 'Negative' in user_sentiment.columns:
                    st.title(user_sentiment['Negative'].idxmax())
        
            with col3:
                st.header("😐 Most Neutral")
                if 'Neutral' in user_sentiment.columns:
                    st.title(user_sentiment['Neutral'].idxmax())          
        
        
        
        
    
        
        
        
        #SENTIMENT TIMELINE
        st.title("Sentiment Timeline")
        
        timeline = my_helper.sentiment_timeline(selected_user, df)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if 'Positive' in timeline.columns:
            ax.plot(
                timeline['month_year'],
                timeline['Positive'],
                color='#2ecc71',
                marker='o',
                label='Positive'
            )
        if 'Negative' in timeline.columns:
            ax.plot(
                timeline['month_year'],
                timeline['Negative'],
                color='#ff6b6b',
                marker='o',
                label='Negative'
            )
        if 'Neutral' in timeline.columns:
            ax.plot(
                timeline['month_year'],
                timeline['Neutral'],
                color='#95a5a6',
                marker='o',
                label='Neutral'
            )
        
        plt.xticks(rotation='vertical')
        plt.legend()
        ax.set_xlabel("Month")
        ax.set_ylabel("Message Count")
        st.pyplot(fig)
        
        
       
       #MOST TOXIC DAY
        st.title("Most Toxic Days 🔥")
        
        toxic_days = my_helper.most_toxic_day(selected_user, df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            top10 = toxic_days.head(10)
            ax.bar(
                top10['date'].astype(str),
                top10['negative_count'],
                color='#ff6b6b'
            )
            plt.xticks(rotation='vertical')
            ax.set_xlabel("Date")
            ax.set_ylabel("Negative Messages")
            st.pyplot(fig)
        
        with col2:
            if not toxic_days.empty:
                worst_day = toxic_days.iloc[0]
                st.header("☠️ Most Toxic Day")
                st.title(str(worst_day['date']))
                st.metric(
                    label="Negative Messages",
                    value=int(worst_day['negative_count'])
                )
                
        #MOST POSITIVE DAY
        st.title("Most Positive Days 🌟")
        
        positive_days = my_helper.most_positive_day(selected_user, df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            top10 = positive_days.head(10)
            ax.bar(
                top10['date'].astype(str),
                top10['positive_count'],
                color='#2ecc71'
            )
            plt.xticks(rotation='vertical')
            ax.set_xlabel("Date")
            ax.set_ylabel("Positive Messages")
            st.pyplot(fig)
        
        with col2:
            if not positive_days.empty:
                best_day = positive_days.iloc[0]
                st.header("🌟 Most Positive Day")
                st.title(str(best_day['date']))
                st.metric(
                    label="Positive Messages",
                    value=int(best_day['positive_count'])
                )
                
        
        
        
        
        
        

    
        #SENTIMENT SCORE
        st.title("Sentiment Score Per User 🏆")
        
        score_df = my_helper.sentiment_score(df)
        
        
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if s >= 60 else '#e74c3c' 
                  if s < 40 else '#95a5a6' 
                  for s in score_df['sentiment_score']]
        ax.barh(
            score_df['user'],
            score_df['sentiment_score'],
            color=colors
        )
        ax.set_xlabel("Sentiment Score (0-100)")
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        st.pyplot(fig)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🏆 Most Positive User")
            st.title(score_df.iloc[0]['user'])
            st.metric(
                "Score",
                f"{score_df.iloc[0]['sentiment_score']}/100"
            )
        with col2:    
            st.header("😔 Most Negative User")
            st.title(score_df.iloc[-1]['user'])
            st.metric(
                "Score",
                f"{score_df.iloc[-1]['sentiment_score']}/100"
            )
    
        # ===== HAPPY HOURS - Overall aur individual dono =====
        st.title("Happy Hours Detection 🌟")
        
        happy_hour_data = my_helper.happy_hours(selected_user, df)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(
            happy_hour_data['hour'],
            happy_hour_data['positive_count'],
            color='#f39c12'
        )
        ax.set_xlabel("Hour of Day (0-23)")
        ax.set_ylabel("Positive Messages")
        ax.set_xticks(range(24))
        
        # Peak hour highlight
        peak_hour = happy_hour_data.loc[
            happy_hour_data['positive_count'].idxmax(), 'hour'
        ]
        st.pyplot(fig)
        st.metric("Peak Happy Hour ⭐", f"{int(peak_hour)}:00 - {int(peak_hour)+1}:00")







is_guide=st.sidebar.checkbox("GUIDE")
if is_guide:
    st.markdown("## 📤 How to Export WhatsApp Chat")
    st.markdown("""
    **On Android:**
    1. Open WhatsApp and go to the chat or group
    2. Tap the three dots (⋮) in the top right corner
    3. Select **More** → **Export Chat**
    4. Choose **Without Media**
    5. Save or share the `.txt` file to your device

    **On iPhone:**
    1. Open the chat or group in WhatsApp
    2. Tap the contact or group name at the top
    3. Scroll down and tap **Export Chat**
    4. Choose **Without Media**
    5. Save the `.txt` file to your device
    """)

    st.markdown("---")
    st.markdown("## ⬆️ How to Upload")
    st.markdown("""
    - Click **Browse files** in the sidebar
    - Select the exported `.txt` file
    - The analysis will load automatically
    - Use the dropdown to select **Overall** or a specific user
    - Click **Show Analysis** to view results
    """)

    st.markdown("---")
    st.markdown("## 📊 Features Overview")

    st.markdown("""
    ### 🔢 Basic Statistics
    Displays four key metrics at a glance:
    - **Total Messages** — Total number of messages exchanged
    - **Total Words** — Total word count across all messages  
    - **Media Shared** — Number of images, videos, or files shared
    - **Links Shared** — Number of URLs shared in the chat

    ---

    ### 📅 Monthly Timeline
    A line graph showing message frequency over each month.
    Helps identify which months had the most activity.

    ---

    ### 📆 Daily Timeline
    A line graph showing day-by-day messaging activity.
    Useful for spotting specific active or inactive periods.

    ---

    ### 🗺️ Activity Map
    - **Most Busy Day** — Which day of the week sees the most messages
    - **Most Busy Month** — Which month was most active overall
    - **Weekly Heatmap** — A grid showing activity across days and hours

    ---

    ### 👥 Most Active Users *(Overall only)*
    A bar chart showing who sends the most messages in the group,
    along with a percentage breakdown table.

    ---

    ### ☁️ Word Cloud
    A visual representation of the most frequently used words.
    Larger words appear more often. Stop words are filtered out.

    ---

    ### 📝 Most Common Words
    A bar chart of the top 20 most used words in the conversation,


    ---

    ### 😊 Emoji Analysis
    - A table of all emojis used and their frequency
    - A pie chart showing the top emojis at a glance

    ---

    ### 💬 Sentiment Analysis
    Analyzes the emotional tone of messages:
    - **Positive** — Happy, excited, or appreciative messages
    - **Negative** — Frustrated, sad, or angry messages  
    - **Neutral** — Informational or casual messages
    - Includes a **per-user breakdown** with most positive and negative user

    ---

    ### 📈 Sentiment Timeline
    A month-by-month line graph showing how the mood of the
    conversation changed over time across all three sentiment classes.

    ---

    ### ☠️ Most Toxic Day
    Identifies the day with the highest number of negative messages.

    ---

    ### 🌟 Most Positive Day
    Identifies the day with the highest number of positive messages.

    ---

    ### 🌟 Happy Hours Detection 
    A 24-hour bar chart showing which time of day sees the most
    positive messages. The peak happy hour is highlighted.

    ---

    ### 🏆 Sentiment Score Per User
    Each user receives a score from **0 to 100** based on their
    overall message sentiment:
    - 🟢 **60 and above** — Positive dominant
    - ⚪ **40 to 60** — Neutral
    - 🔴 **Below 40** — Negative dominant
    """)

    st.markdown("---")
    st.markdown("## ⚠️ Important Notes")
    st.markdown("""
    - Always export chat **Without Media** for compatibility
    - Only `.txt` format files are supported
    - Select **Overall** to analyze the entire group
    - Select a **specific user** to view their individual analysis
    - Very old chats may have a different date format and may not parse correctly
    """)
    
    st.markdown("---")
    st.markdown(">Disclaimer")
    st.markdown("""
    <small>The sentiment analysis results generated by this application are based on machine learning predictions and may not always be fully accurate. Messages can sometimes be misinterpreted due to sarcasm, slang, mixed languages, context,informal communication styles, or because of many other reason.
         This tool is intended only for educational and analytical purposes. Users are advised not to take the generated sentiments personally or treat them as definitive judgments about any individual or conversation.</small>
    """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    st.markdown("""
    <small>Developed as a B.Sc. Final Year Project By Dipanshu Jaiswal [23220CMP008] | 
    WhatsApp Chat & Sentiment Analyzer</small>
    """, unsafe_allow_html=True)
                    