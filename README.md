# 🎮 MLBB Reviews Dashboard

A comprehensive Streamlit dashboard for analyzing Mobile Legends: Bang Bang (MLBB) game reviews with advanced search capabilities using ChromaDB and meaningful visualizations.

## 🚀 Features

### 📊 Overview Dashboard
- **Key Metrics**: Total reviews, average rating, total thumbs up, unique users
- **Rating Distribution**: Visual breakdown of 1-5 star ratings
- **Temporal Analysis**: Reviews over time and by hour of day
- **Word Frequency**: Most common words in reviews

### 🔍 Advanced Search Engine
- **Semantic Search**: Powered by ChromaDB for intelligent review search
- **Flexible Queries**: Search by keywords like "matchmaking", "graphics", "lag", etc.
- **Rich Results**: Display user info, ratings, thumbs up, and review content
- **Configurable Results**: Adjust number of search results (5-50)

### 📈 Detailed Analytics
- **Filtered Analysis**: Filter by year and rating
- **Trend Analysis**: Monthly rating trends
- **Sentiment Analysis**: Positive/negative/neutral sentiment distribution
- **Interactive Charts**: Plotly-powered visualizations

### 🎯 Key Insights
- **Rating Insights**: Best/worst rated periods
- **Time-based Insights**: Peak hours and most active days
- **Trend Analysis**: Recent vs overall performance
- **Keyword Analysis**: Positive and negative keyword frequency

### ⚙️ Admin Panel
- **Data Management**: Export data, view statistics
- **System Health**: ChromaDB status, memory usage
- **Data Quality**: Duplicate detection, content analysis
- **Anomaly Detection**: Identify suspicious reviews
- **Spam Detection**: Flag potentially spam reviews

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data file is in the correct location**:
   - Place `mobile_legends_reviews.csv` in the project directory
   - The file should have columns: `reviewId`, `userName`, `content`, `score`, `thumbsUpCount`, `at`

## 🚀 Usage

1. **Run the dashboard**:
   ```bash
   streamlit run mlbb_dashboard.py
   ```

2. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`
   - The dashboard will automatically load and process your data

3. **Navigate through sections**:
   - Use the sidebar to switch between different pages
   - Each page offers different insights and functionality

## 🔍 Search Functionality

### How to Use the Search Engine:
1. Go to the "🔍 Search Reviews" page
2. Enter your search query (e.g., "matchmaking", "graphics", "lag")
3. Adjust the number of results if needed
4. View detailed results with user information and ratings

### Search Examples:
- `matchmaking` - Find reviews about matchmaking issues
- `graphics` - Search for graphics-related feedback
- `lag` - Find reviews mentioning lag or performance issues
- `gameplay` - Search for gameplay-related comments
- `update` - Find reviews about game updates

## 📊 Data Structure

The dashboard expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `reviewId` | String | Unique review identifier |
| `userName` | String | Name of the reviewer |
| `content` | String | Review text content |
| `score` | Integer | Rating (1-5 stars) |
| `thumbsUpCount` | Integer | Number of thumbs up |
| `at` | DateTime | Review timestamp |

## 🎨 Customization

### Styling
- The dashboard uses custom CSS for a modern gaming theme
- Colors and styling can be modified in the CSS section of the code

### Admin Access
- Default admin password: `admin123`
- Change the password in the code for production use

### ChromaDB Configuration
- The search engine uses ChromaDB with DuckDB backend
- Database is stored in `./chroma_db` directory
- First run will create the database and embeddings

## 🔧 Troubleshooting

### Common Issues:

1. **Data Loading Error**:
   - Ensure `mobile_legends_reviews.csv` is in the correct location
   - Check file format and column names

2. **ChromaDB Issues**:
   - Delete `./chroma_db` directory and restart
   - Ensure sufficient disk space for embeddings

3. **NLTK Data Missing**:
   - The app will automatically download required NLTK data
   - If issues persist, manually download: `python -m nltk.downloader punkt stopwords vader_lexicon`

4. **Memory Issues**:
   - For large datasets, consider sampling data
   - Increase system memory if available

## 📈 Performance Tips

- **Large Datasets**: The dashboard samples data for sentiment analysis to improve performance
- **Search Optimization**: ChromaDB embeddings are cached for faster searches
- **Memory Management**: Data is cached using Streamlit's caching mechanisms

## 🔒 Security Notes

- Admin password is hardcoded for demo purposes
- Implement proper authentication for production use
- Consider data privacy and GDPR compliance

## 🤝 Contributing

Feel free to contribute by:
- Adding new visualizations
- Improving search functionality
- Enhancing the UI/UX
- Adding new analytics features

## 📄 License

This project is open source and available under the MIT License.

## 🎮 About MLBB

Mobile Legends: Bang Bang (MLBB) is a mobile MOBA game developed by Moonton. This dashboard helps analyze player feedback and reviews to understand game performance and user satisfaction.

---

**Happy Gaming! 🎮** 