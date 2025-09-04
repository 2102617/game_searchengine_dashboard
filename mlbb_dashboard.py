import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chromadb
import re
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import os

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data


# Page configuration
st.set_page_config(
    page_title="MLBB Reviews Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .search-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .review-card {
        color:black;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the MLBB reviews data"""
    try:
        df = pd.read_csv('mobile_legends_reviews.csv')
        df['at'] = pd.to_datetime(df['at'])
        df['year'] = df['at'].dt.year
        df['month'] = df['at'].dt.month
        df['day'] = df['at'].dt.day
        df['hour'] = df['at'].dt.hour
        df['weekday'] = df['at'].dt.day_name()
        
        # Clean content for better analysis
        df['content_clean'] = df['content'].astype(str).str.lower()
        df['content_clean'] = df['content_clean'].str.replace(r'[^\w\s]', '', regex=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound']

@st.cache_resource
def setup_chromadb(df):
    """Setup ChromaDB for semantic search using SentenceTransformer embeddings"""
    try:
        from chromadb import PersistentClient
        from sentence_transformers import SentenceTransformer

        # Load embedding model (MiniLM is fast & accurate)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Wrap it for ChromaDB
        class STEmbeddingFunction:
            def __call__(self, input_texts):
                return model.encode(input_texts, convert_to_numpy=True).tolist()

        # Initialize client
        client = PersistentClient(path="./chroma_db")

        # Create or get collection
        collection_name = "mlbb_reviews"
        try:
            collection = client.get_collection(collection_name)
        except:
            collection = client.create_collection(
                collection_name,
                embedding_function=STEmbeddingFunction()
            )

        # Prepare documents
        documents = df['content'].astype(str).tolist()
        metadatas, ids = [], []
        for idx, row in df.iterrows():
            metadata = {
                'reviewId': row['reviewId'],
                'userName': row['userName'],
                'score': row['score'],
                'thumbsUpCount': row['thumbsUpCount'],
                'date': str(row['at']),
                'year': row['year'],
                'month': row['month']
            }
            metadatas.append(metadata)
            ids.append(str(idx))

        # Add only if empty
        if collection.count() == 0:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return client, collection
    except Exception as e:
        st.error(f"Error setting up ChromaDB: {e}")
        return None, None


def search_reviews(collection, query, n_results=10):
    """Search reviews using ChromaDB"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return None
    
from groq import Groq

# -------------------------------
# AI Assistant with RAG
# -------------------------------
def ai_assistant(query, collection, top_k=5):
    """
    Retrieve relevant reviews from ChromaDB and answer using Groq API.
    """

    if not query or not collection:
        return "⚠️ No query or collection available."

    # Step 1: Retrieve context from ChromaDB
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        return f"⚠️ Retrieval error: {e}"

    retrieved_docs = results["documents"][0] if results and "documents" in results else []
    context = "\n".join(retrieved_docs)

    if not context:
        return "❌ No relevant reviews found."

    # Step 2: Create prompt for Groq
    system_prompt = """
    You are an AI assistant specialized in analyzing Mobile Legends Bang Bang (MLBB) user reviews.
    You ONLY use the provided reviews as your source of truth.
    - If the reviews contain dates, thumbs up counts, or trends, mention them explicitly.
    - If the query asks about a time period (e.g., November 2025), summarize reviews from that time.
    - Do NOT make assumptions or add general knowledge about the game outside the reviews.
    - If you cannot find enough information in the reviews, say: 
      "The available reviews do not provide enough information to answer this."
    """
    user_prompt = f"""
    You are an assistant analyzing game reviews. 
    Answer the user's question based only on the following reviews:

    {context}

    User Question: {query}
    Answer in a clear and concise way.
    """


    # Step 3: Call Groq API
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # You can also use "llama-3.1-8b-instant"
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        return f"⚠️ Groq API error: {e}"


def create_overview_metrics(df):
    """Create overview metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reviews",
            value=f"{len(df):,}",
            delta=f"{len(df[df['at'] >= datetime.now() - timedelta(days=30)]):,} (last 30 days)"
        )
    
    with col2:
        avg_score = df['score'].mean()
        st.metric(
            label="Average Rating",
            value=f"{avg_score:.2f}/5",
            delta=f"{avg_score - 3:.2f}" if avg_score != 3 else "0"
        )
    
    with col3:
        total_thumbs = df['thumbsUpCount'].sum()
        st.metric(
            label="Total Thumbs Up",
            value=f"{total_thumbs:,}",
            delta=f"{df[df['at'] >= datetime.now() - timedelta(days=30)]['thumbsUpCount'].sum():,} (last 30 days)"
        )
    
    with col4:
        unique_users = df['userName'].nunique()
        st.metric(
            label="Unique Users",
            value=f"{unique_users:,}",
            delta=f"{df[df['at'] >= datetime.now() - timedelta(days=30)]['userName'].nunique():,} (last 30 days)"
        )

def create_rating_distribution(df):
    """Create rating distribution chart"""
    rating_counts = df['score'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Rating Distribution",
        labels={'x': 'Rating', 'y': 'Number of Reviews'},
        color=rating_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_title="Rating (1-5)",
        yaxis_title="Number of Reviews",
        showlegend=False
    )
    return fig

def create_temporal_analysis(df):
    """Create temporal analysis charts"""
    # Reviews over time
    daily_reviews = df.groupby(df['at'].dt.date).size().reset_index()
    daily_reviews.columns = ['date', 'count']
    
    fig1 = px.line(
        daily_reviews,
        x='date',
        y='count',
        title="Reviews Over Time",
        labels={'date': 'Date', 'count': 'Number of Reviews'}
    )
    
    # Reviews by hour
    hourly_reviews = df['hour'].value_counts().sort_index()
    
    fig2 = px.bar(
        x=hourly_reviews.index,
        y=hourly_reviews.values,
        title="Reviews by Hour of Day",
        labels={'x': 'Hour', 'y': 'Number of Reviews'}
    )
    
    return fig1, fig2

def create_sentiment_analysis(df):
    """Create sentiment analysis"""
    # Analyze sentiment for a sample of reviews (for performance)
    sample_df = df.sample(min(1000, len(df)))
    sample_df['sentiment'] = sample_df['content'].apply(analyze_sentiment)
    
    # Categorize sentiment
    sample_df['sentiment_category'] = pd.cut(
        sample_df['sentiment'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    sentiment_counts = sample_df['sentiment_category'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution (Sample)",
        color_discrete_map={'Positive': '#00FF00', 'Neutral': '#FFA500', 'Negative': '#FF0000'}
    )
    
    return fig

def create_word_cloud_data(df):
    """Prepare data for word cloud"""
    # Combine all content
    all_text = ' '.join(df['content'].astype(str))
    
    # Remove common words and clean
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(all_text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = pd.Series(words).value_counts().head(50)
    return word_freq

def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 MLBB Reviews Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Setup ChromaDB
    client, collection = setup_chromadb(df)
    
    # Sidebar
    st.sidebar.title("🎮 MLBB Dashboard")
    st.sidebar.markdown("### Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "📈 Analytics", "🎯 Insights", "🔍 Search Reviews", "🤖 AI Assistant", "⚙️ Admin Panel"]
    )
    
    if page == "📊 Overview":
        st.header("📊 Overview Dashboard")
        
        # Overview metrics
        create_overview_metrics(df)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_rating_distribution(df), use_container_width=True)
        
        with col2:
            fig1, fig2 = create_temporal_analysis(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        # Word frequency
        st.subheader("📝 Most Common Words in Reviews")
        word_freq = create_word_cloud_data(df)
        
        # Create a bar chart for word frequency
        fig = px.bar(
            x=word_freq.values[:20],
            y=word_freq.index[:20],
            orientation='h',
            title="Top 20 Most Common Words",
            labels={'x': 'Frequency', 'y': 'Words'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "🤖 AI Assistant":
        st.header("🤖 AI Assistant")
        query = st.text_input("Ask a question about the reviews:")

        if query:
            with st.spinner("Thinking..."):
                answer = ai_assistant(query, collection, top_k=5)
            st.write("### Answer:")
            st.write(answer)
        
    elif page == "🔍 Search Reviews":
        st.header("🔍 Search Reviews")
        
        if collection is None:
            st.error("ChromaDB is not available. Please check the setup.")
            return
        
        # Search interface
        
        search_query = st.text_input("Enter your search query:", placeholder="e.g., matchmaking, graphics, lag, gameplay...")
        n_results = st.slider("Number of results:", min_value=5, max_value=50, value=10)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if search_query:
            with st.spinner("Searching..."):
                results = search_reviews(collection, search_query, n_results)
            
            if results and results['documents']:
                st.subheader(f"🔍 Search Results for: '{search_query}'")
                
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    with st.container():
                        st.markdown(f"""
                        <div class="review-card">
                            <p><strong>User:</strong> {metadata['userName']}</p>
                            <p><strong>Rating:</strong> {'⭐' * int(metadata['score'])} ({metadata['score']}/5)</p>
                            <p><strong>Thumbs Up:</strong> {metadata['thumbsUpCount']}</p>
                            <p><strong>Date:</strong> {metadata['date']}</p>
                            <p><strong>Content:</strong> {doc[:300]}{'...' if len(doc) > 300 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No results found for your query.")
    
    elif page == "📈 Analytics":
        st.header("📈 Detailed Analytics")
        
        # Filter options
        st.subheader("📊 Filter Options")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_years = st.multiselect(
                "Select Years:",
                options=sorted(df['year'].unique()),
                default=sorted(df['year'].unique())
            )
        
        with col2:
            selected_scores = st.multiselect(
                "Select Ratings:",
                options=sorted(df['score'].unique()),
                default=sorted(df['score'].unique())
            )
        
        # Filter data
        filtered_df = df[
            (df['year'].isin(selected_years)) &
            (df['score'].isin(selected_scores))
        ]
        
        # Analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trend
            monthly_data = filtered_df.groupby(['year', 'month']).agg({
                'score': 'mean',
                'reviewId': 'count'
            }).reset_index()
            monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
            
            fig = px.line(
                monthly_data,
                x='date',
                y='score',
                title="Average Rating Trend (Monthly)",
                labels={'date': 'Date', 'score': 'Average Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly distribution
            fig1, fig2 = create_temporal_analysis(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Sentiment analysis
        st.subheader("😊 Sentiment Analysis")
        sentiment_fig = create_sentiment_analysis(filtered_df)
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    elif page == "🎯 Insights":
        st.header("🎯 Key Insights")
        
        # Top insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Rating Insights")
            
            # Best and worst rated periods
            monthly_ratings = df.groupby(['year', 'month'])['score'].mean().reset_index()
            best_month = monthly_ratings.loc[monthly_ratings['score'].idxmax()]
            worst_month = monthly_ratings.loc[monthly_ratings['score'].idxmin()]
            
            st.info(f"**Best Month:** {best_month['month']}/{best_month['year']} (Rating: {best_month['score']:.2f})")
            st.warning(f"**Worst Month:** {worst_month['month']}/{worst_month['year']} (Rating: {worst_month['score']:.2f})")
            
            # Rating distribution
            rating_dist = df['score'].value_counts().sort_index()
            st.write("**Rating Distribution:**")
            for rating, count in rating_dist.items():
                percentage = (count / len(df)) * 100
                st.write(f"{'⭐' * rating}: {count:,} reviews ({percentage:.1f}%)")
        
        with col2:
            st.subheader("⏰ Time-based Insights")
            
            # Peak hours
            peak_hour = df['hour'].value_counts().index[0]
            st.info(f"**Peak Review Hour:** {peak_hour}:00")
            
            # Most active day
            peak_day = df['weekday'].value_counts().index[0]
            st.info(f"**Most Active Day:** {peak_day}")
            
            # Recent trend
            recent_avg = df[df['at'] >= datetime.now() - timedelta(days=30)]['score'].mean()
            overall_avg = df['score'].mean()
            trend = "📈 Improving" if recent_avg > overall_avg else "📉 Declining"
            st.info(f"**Recent Trend:** {trend} (Recent: {recent_avg:.2f}, Overall: {overall_avg:.2f})")
        
        # Top keywords analysis
        st.subheader("🔍 Keyword Analysis")
        word_freq = create_word_cloud_data(df)
        
        # Positive and negative keywords
        positive_words = ['good', 'great', 'awesome', 'love', 'amazing', 'best', 'excellent']
        negative_words = ['bad', 'terrible', 'horrible', 'worst', 'hate', 'awful', 'lag']
        
        positive_freq = {word: word_freq.get(word, 0) for word in positive_words}
        negative_freq = {word: word_freq.get(word, 0) for word in negative_words}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Keywords:**")
            for word, freq in sorted(positive_freq.items(), key=lambda x: x[1], reverse=True):
                if freq > 0:
                    st.write(f"✅ {word}: {freq} mentions")
        
        with col2:
            st.write("**Negative Keywords:**")
            for word, freq in sorted(negative_freq.items(), key=lambda x: x[1], reverse=True):
                if freq > 0:
                    st.write(f"❌ {word}: {freq} mentions")
    
    elif page == "⚙️ Admin Panel":
        st.header("⚙️ Admin Panel")
        
        # Admin authentication (simple demo)
        st.subheader("🔐 Admin Authentication")
        password = st.text_input("Enter admin password:", type="password")
        
        if password == "admin123":  # Demo password
            st.success("✅ Admin access granted!")
            
            # Data management
            st.subheader("📊 Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Statistics:**")
                st.write(f"Total Records: {len(df):,}")
                st.write(f"Date Range: {df['at'].min()} to {df['at'].max()}")
                st.write(f"Missing Values: {df.isnull().sum().sum()}")
                
                # Export data
                if st.button("📥 Export Data"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="mlbb_reviews_export.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.write("**System Health:**")
                st.write(f"ChromaDB Status: {'✅ Connected' if collection else '❌ Disconnected'}")
                st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Data quality metrics
                st.write("**Data Quality:**")
                duplicate_ratio = (len(df) - len(df.drop_duplicates())) / len(df) * 100
                st.write(f"Duplicate Ratio: {duplicate_ratio:.2f}%")
                
                avg_content_length = df['content'].str.len().mean()
                st.write(f"Avg Content Length: {avg_content_length:.0f} characters")
            
            # Advanced analytics
            st.subheader("🔬 Advanced Analytics")
            
            # Anomaly detection
            st.write("**Anomaly Detection:**")
            
            # Reviews with very low scores but high thumbs up
            anomalies = df[(df['score'] <= 2) & (df['thumbsUpCount'] > 10)]
            if not anomalies.empty:
                st.warning(f"Found {len(anomalies)} reviews with low scores but high thumbs up")
                st.dataframe(anomalies[['userName', 'content', 'score', 'thumbsUpCount']].head())
            
            # Spam detection (very short reviews)
            spam_reviews = df[df['content'].str.len() < 10]
            if not spam_reviews.empty:
                st.warning(f"Found {len(spam_reviews)} potentially spam reviews (very short)")
                st.dataframe(spam_reviews[['userName', 'content', 'score']].head())
        
        else:
            st.error("❌ Invalid password. Please try again.")

if __name__ == "__main__":
    main() 