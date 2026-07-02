import streamlit as st

st.set_page_config(
    page_title="Gurugram Real Estate Intelligence",
    page_icon=":house:",
    layout="wide"
)

# ---------------------- CUSTOM CSS ---------------------- #

st.markdown("""
<style>

.main > div {
    padding-top: 2rem;
}

h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #1F2937;
}

div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

.feature-card{
    background-color:#ffffff;
    border:1px solid #E5E7EB;
    border-left: 4px solid #2563EB;
    border-radius:10px;
    padding:25px;
    min-height:320px;
    box-sizing:border-box;
    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
}

.feature-card h3{
    margin-top:0;
    margin-bottom:12px;
    color:#1F2937;
}

.feature-card p{
    margin-bottom:14px;
    line-height:1.5;
    color:#374151;
}

.feature-card ul{
    margin:0;
    padding-left:20px;
    line-height:1.8;
    color:#374151;
}

.tech-card{
    background-color:#ffffff;
    border:1px solid #E5E7EB;
    border-radius:10px;
    padding:20px 25px;
    box-sizing:border-box;
    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
}

.tech-card p{
    margin-bottom:6px;
    margin-top:16px;
}

.tech-card p:first-child{
    margin-top:0;
}

.tech-card ul{
    margin:0 0 4px 0;
    padding-left:20px;
    line-height:1.7;
    color:#374151;
}

.section-divider{
    margin-top: 10px;
    margin-bottom: 10px;
    border: none;
    border-top: 1px solid #E5E7EB;
}

.footer{
    text-align:center;
    color:#9CA3AF;
    font-size:14px;
}

.subtle-label{
    color:#6B7280;
    font-size:14px;
    letter-spacing:0.5px;
    text-transform:uppercase;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ---------------------- #

st.title("Gurugram Real Estate Intelligence Platform")

st.write("""
A machine learning based real estate analytics platform for exploring market trends,
estimating apartment prices, and discovering comparable properties across Gurugram
using data-driven recommendation techniques.
""")

st.divider()

# ---------------------- METRICS ---------------------- #

col1, col2, col3, col4 = st.columns(4)

col1.metric("Properties Analyzed", "4,000+")
col2.metric("Sectors Covered", "102+")
col3.metric("ML Models Used", "3")
col4.metric("Prediction R\u00b2 Score", "0.90")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- OVERVIEW ---------------------- #

st.header("Project Overview")

st.write("""
This application demonstrates an end-to-end data science workflow built on a
Gurugram residential real estate dataset. It combines exploratory data analysis,
feature engineering, machine learning, and recommendation systems to support
informed property decisions.

The platform is built using Python, Scikit-learn, Pandas, and Plotly, and is
deployed on AWS EC2 to simulate a production-style hosting environment.
""")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- FEATURES ---------------------- #

st.header("Platform Features")

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        '<div class="feature-card">'
        '<h3>Market Analysis</h3>'
        '<p>Interactive visualizations built from thousands of property listings '
        'across Gurugram sectors.</p>'
        '<ul>'
        '<li>Sector-wise price analysis</li>'
        '<li>Price distribution trends</li>'
        '<li>BHK-wise comparison</li>'
        '<li>Luxury feature insights</li>'
        '<li>Location-based analytics</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        '<div class="feature-card">'
        '<h3>Price Prediction</h3>'
        '<p>Estimates residential property prices using a trained Random Forest '
        'regression model.</p>'
        '<ul>'
        '<li>Multi-feature input pipeline</li>'
        '<li>Engineered dataset features</li>'
        '<li>Real-time price estimation</li>'
        '<li>Robust regression pipeline</li>'
        '<li>Consistent, repeatable results</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        '<div class="feature-card">'
        '<h3>Apartment Recommendation</h3>'
        '<p>Surfaces comparable apartments using a hybrid recommendation engine '
        'based on location and property attributes.</p>'
        '<ul>'
        '<li>Radius-based search</li>'
        '<li>Location similarity scoring</li>'
        '<li>Content-based filtering</li>'
        '<li>Cosine similarity matching</li>'
        '<li>Direct links to listings</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- WORKFLOW ---------------------- #

st.header("Machine Learning Workflow")

st.markdown("""
1. Data collection
2. Data cleaning and feature engineering
3. Exploratory data analysis
4. Feature selection
5. Model training and evaluation
6. Price prediction
7. Recommendation system
8. Deployment on AWS EC2
""")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- TECH STACK ---------------------- #

st.header("Technology Stack")

left, right = st.columns(2)

with left:

    st.markdown(
        '<div class="tech-card">'
        '<p><strong>Programming</strong></p>'
        '<ul><li>Python</li><li>SQL</li></ul>'
        '<p><strong>Machine Learning</strong></p>'
        '<ul><li>Scikit-learn</li><li>Random Forest</li><li>TF-IDF</li>'
        '<li>Cosine Similarity</li></ul>'
        '<p><strong>Data Processing</strong></p>'
        '<ul><li>Pandas</li><li>NumPy</li></ul>'
        '</div>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        '<div class="tech-card">'
        '<p><strong>Visualization</strong></p>'
        '<ul><li>Plotly</li><li>Matplotlib</li></ul>'
        '<p><strong>Deployment</strong></p>'
        '<ul><li>Streamlit</li><li>AWS EC2</li></ul>'
        '<p><strong>Development Tools</strong></p>'
        '<ul><li>Git</li><li>Jupyter Notebook</li></ul>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- PROJECT SUMMARY ---------------------- #

st.header("Project Summary")

st.write("""
This project demonstrates the practical application of machine learning in the
real estate domain, covering exploratory data analysis, feature engineering,
predictive modeling, and recommendation systems within a single deployed
application.

The platform is designed to reflect a complete machine learning lifecycle:
data preprocessing, model training, evaluation, and cloud deployment on AWS
EC2, with a focus on usability and clear presentation of results.
""")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------- DEVELOPER ---------------------- #

st.header("Developer")

st.subheader("Surbhit Parashar")

st.write("Machine Learning Engineer | Data Science")

st.write("""
Focused on building end-to-end machine learning solutions, including
predictive modeling, recommendation systems, and interactive data
applications, deployed using cloud infrastructure such as AWS EC2.
""")

st.write("**Email:** surbhitparashar7@gmail.com")
st.write("**GitHub:** https://github.com/SurbhitParashar")

st.divider()

st.markdown(
    '<p class="footer">Gurugram Real Estate Intelligence Platform &nbsp;|&nbsp; '
    'Built with Streamlit and Scikit-learn &nbsp;|&nbsp; Deployed on AWS EC2</p>',
    unsafe_allow_html=True
)