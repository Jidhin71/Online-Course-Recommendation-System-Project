#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:





# In[3]:


import warnings
warnings.filterwarnings("ignore")

#### content based recommendation
# In[4]:


def recom_course(similar_course, dif_level,n=5):
    
    # Step 1 — Check if course exists in catalog
    match = (course_catalog["course_name"] == similar_course) & (course_catalog["difficulty_level"] == dif_level) #returns as true or false
    
    if match.any():  # .any() checks if at least one row matches
        
        # Step 2 — Get the row index of the matched course
        index = match[match == True].index[0]  # gets the actual integer index
        
        # Step 3 — Get similarity scores for this course vs all others
        # content_sim[index] → one row → similarity of this course with every course
        similar_indices = sorted(
            list(enumerate(content_sim[index])),  # [(0, 0.9), (1, 0.3), ...]
            reverse=True,                          # highest similarity first
            key=lambda x: x[1]                    # sort by score not index
        )[1:n+1]  # skip index 0 (that's the course itself, always score=1.0)
        
        

        result=[]
        for rank, (index, score) in enumerate(similar_indices, start=1):
            row = course_catalog.iloc[index]  # fetch full course details by position
            result.append({
                'Rank'       : rank,
                'Course Name': row['course_name'],
                'Difficulty' : row['difficulty_level'],
                'Avg Rating' : round(row['avg_rating'], 2),
                'Avg Price'  : round(row['avg_price'],  2),
                'Similarity' : round(score, 4)
            })
        result = pd.DataFrame(result)
        return result, None
    
    else:
        return None, f"❌ Course '{similar_course} | {dif_level}' not found!"
    

#### collaborative based
# In[5]:


def similar_users(user_id, n=5):
    
    if user_id not in user_course_matrix.index:
        return None, f"❌ User '{user_id}' not found!"
    
    # Step 1 — Compute similarity on the fly (avoids memory error!)
    user_vector   = user_course_matrix.loc[[user_id]]
    sim_scores    = cosine_similarity(user_vector, user_course_matrix)[0]
    similar_idx   = np.argsort(sim_scores)[::-1][1:11]
    similar_users = user_course_matrix.index[similar_idx]
    
    # Step 2 — Get their ratings and average course by course
    collab_scores = user_course_matrix.loc[similar_users].mean(axis=0)
    
    # Step 3 — Normalize between 0 and 1
    collab_scores = (collab_scores - collab_scores.min()) / \
                    (collab_scores.max() - collab_scores.min() + 1e-9)
    
    # Step 4 — Find courses already taken
    courses_taken = set(user_course_matrix.loc[user_id][
        user_course_matrix.loc[user_id] > 0].index)
    
    # Step 5 — Remove already taken courses
    collab_scores = collab_scores[~collab_scores.index.isin(courses_taken)]
    
    # Step 6 — Sort and get top N
    top_courses = collab_scores.sort_values(ascending=False).head(n)
    
    # Step 7 — Build result
    result = []
    for rank, (course_key, score) in enumerate(top_courses.items(), start=1):
        parts       = course_key.split(" | ")
        course_name = parts[0]
        difficulty  = parts[1] if len(parts) > 1 else "N/A"
        result.append({
            "Rank"       : rank,
            "Course Name": course_name,
            "Difficulty" : difficulty,
            "Score"      : round(score, 4)  # ✅ added score
        })
    
    result = pd.DataFrame(result)  # ✅ capital D and F
    return result, None


# In[6]:


#### Hybrid


# In[7]:


def hybrid_recommend(user_id, top_n=5,
                     w_content=0.5, w_collab=0.3, w_engage=0.2):
    
    # ── Collaborative score ──
    if user_id in user_course_matrix.index:
        
        # Compute on the fly — avoids memory error ✅
        user_vector   = user_course_matrix.loc[[user_id]]
        sim_scores    = cosine_similarity(user_vector, user_course_matrix)[0]
        similar_idx   = np.argsort(sim_scores)[::-1][1:11]
        similar_users = user_course_matrix.index[similar_idx]
        
        collab_scores = user_course_matrix.loc[similar_users].mean(axis=0)
        collab_scores = (collab_scores - collab_scores.min()) / \
                        (collab_scores.max() - collab_scores.min() + 1e-9)
    else:
        collab_scores = pd.Series(0, index=course_catalog['course_key'])
    
    # ── Content score ──
    user_courses   = df[df['user_id'] == user_id]['course_key'].unique()
    content_scores = np.zeros(len(course_catalog))
    course_index   = pd.Series(course_catalog.index, 
                               index=course_catalog['course_key'])
    
    for ck in user_courses:
        if ck in course_index:
            idx = course_index[ck]
            content_scores += content_sim[idx]
    
    content_scores = (content_scores - content_scores.min()) / \
                     (content_scores.max() - content_scores.min() + 1e-9)
    
    # ── Engagement score ──
    engage_scores = course_catalog['engagement_score'].values
    
    # ── Hybrid ──
    collab_array = collab_scores.reindex(
        course_catalog['course_key']).fillna(0).values
    
    final_scores = (w_content * content_scores +
                    w_collab  * collab_array   +
                    w_engage  * engage_scores)
    
    # ── Remove already enrolled ──
    course_catalog['final_score'] = final_scores
    already_enrolled  = set(user_courses)
    recommendations   = course_catalog[
        ~course_catalog['course_key'].isin(already_enrolled)
    ].sort_values('final_score', ascending=False)
    
    result = recommendations[['course_name', 'difficulty_level',
                               'avg_rating', 'avg_price',
                               'final_score']].head(top_n)
    
    return result, None  # ✅ return None as error (consistent with other functions)


# In[8]:


# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title = "Course Recommendation System",
    page_icon  = "🎓",
    layout     = "wide"
)

# ── Load Pickle Files ─────────────────────────────────────
@st.cache_resource
def load_data():
    content_sim        = joblib.load("content_sim.pkl")
    user_course_matrix = joblib.load("user_course_matrix.pkl")
    course_catalog     = joblib.load("course_catalog.pkl")
    df = joblib.load("df.pkl")
    return content_sim, user_course_matrix, course_catalog,df

content_sim, user_course_matrix, course_catalog,df= load_data()


# ── Sidebar Navigation ────────────────────────────────────
st.sidebar.title("🎓 Course Recommender")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Navigate", [
    "📊 Dataset Overview",
    "📈 Visualizations",
    "🔍 EDA",
    "🎯 Prediction"
])

# ════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ════════════════════════════════════════════════════════
if menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Courses",  len(course_catalog))
    col2.metric("Total Users",    user_course_matrix.shape[0])
    col3.metric("Total Ratings", len(df) )
    
    st.markdown("---")
    
    # Show dataset
    st.subheader("📋 Course Catalog")
    st.dataframe(course_catalog.head(20))
    
    st.subheader("📋 Sample Ratings Data")
    st.dataframe(df.head(20))  # your original df

# ════════════════════════════════════════════════════════
# SECTION 2 — VISUALIZATIONS
# ════════════════════════════════════════════════════════
elif menu == "📈 Visualizations":
    st.title("📈 Visualizations")
    st.markdown("---")
    
    # Rating Distribution
    st.subheader("⭐ Rating Distribution")
    fig, ax = plt.subplots()
    course_catalog['avg_rating'].hist(bins=20, ax=ax, color='steelblue')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Difficulty Level Count
    st.subheader("📚 Courses by Difficulty Level")
    fig2, ax2 = plt.subplots()
    course_catalog['difficulty_level'].value_counts().plot(
        kind='bar', ax=ax2, color='coral')
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    
    st.markdown("---")
    
    # Price Distribution
    st.subheader("💰 Price Distribution")
    fig3, ax3 = plt.subplots()
    course_catalog['avg_price'].hist(bins=20, ax=ax3, color='green')
    ax3.set_xlabel("Price")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

# ════════════════════════════════════════════════════════
# SECTION 3 — EDA
# ════════════════════════════════════════════════════════
elif menu == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    # Top rated courses
    st.subheader("🏆 Top 10 Highest Rated Courses")
    top_rated = course_catalog.nlargest(10, 'avg_rating')[
        ['course_name', 'avg_rating', 'difficulty_level']]
    st.dataframe(top_rated)
    
    st.markdown("---")
    
    # Most popular courses
    st.subheader("🔥 Top 10 Most Popular Courses")
    top_popular = course_catalog.nlargest(10, 'engagement_score')[
        ['course_name', 'engagement_score', 'avg_rating']]
    st.dataframe(top_popular)
    
    st.markdown("---")
    
    # Rating vs Price
    st.subheader("💡 Rating vs Price")
    fig4, ax4 = plt.subplots()
    ax4.scatter(course_catalog['avg_price'], 
                course_catalog['avg_rating'], alpha=0.5)
    ax4.set_xlabel("Price")
    ax4.set_ylabel("Rating")
    st.pyplot(fig4)

# ════════════════════════════════════════════════════════
# SECTION 4 — PREDICTION
# ════════════════════════════════════════════════════════
elif menu == "🎯 Prediction":
    st.title("🎯 Course Recommendations")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([
        "📄 Content Based",
        "👥 Collaborative Based",
        "🔀 Hybrid Based"
    ])
    
    # ── Tab 1: Content Based ──────────────────────────────
    with tab1:
        st.subheader("📄 Content Based Recommendations")
        
        course_name = st.selectbox(
            "Select a Course",
            course_catalog['course_name'].unique(),
            key="content_course"
        )
        dif_level = st.selectbox(
            "Select Difficulty Level",
            ["Beginner", "Intermediate", "Advanced"],
            key="content_diff"
        )
        top_n_content = st.slider(
            "Number of Recommendations",
            min_value=1, max_value=10, value=5,
            key="content_slider"
        )
        
        if st.button("Get Content Based Recommendations"):
            with st.spinner("Finding similar courses..."):
                result, error = recom_course(course_name, dif_level, top_n_content)
                if error:
                    st.error(error)
                else:
                    st.success(f"Top {top_n_content} courses similar to '{course_name}'")
                    st.dataframe(result)
    
    # ── Tab 2: Collaborative Based ────────────────────────
    with tab2:
        st.subheader("👥 Collaborative Based Recommendations")
        
        user_id_collab = st.number_input(
            "Enter User ID",
            min_value=1, step=1,
            key="collab_user"
        )
        top_n_collab = st.slider(
            "Number of Recommendations",
            min_value=1, max_value=10, value=5,
            key="collab_slider"
        )
        
        if st.button("Get Collaborative Recommendations"):
            with st.spinner("Finding similar users..."):
                result, error = similar_users(user_id_collab, top_n_collab)
                if error:
                    st.error(error)
                else:
                    st.success(f"Top {top_n_collab} Recommendations for User {user_id_collab}")
                    st.dataframe(result)
    
    # ── Tab 3: Hybrid Based ───────────────────────────────
    with tab3:
        st.subheader("🔀 Hybrid Based Recommendations")
        
        user_id_hybrid = st.number_input(
            "Enter User ID",
            min_value=1, step=1,
            key="hybrid_user"
        )
        top_n_hybrid = st.slider(
            "Number of Recommendations",
            min_value=1, max_value=10, value=5,
            key="hybrid_slider"
        )
        
        if st.button("Get Hybrid Recommendations"):
            with st.spinner("Combining all models..."):
                result, error = hybrid_recommend(user_id_hybrid, top_n=top_n_hybrid)
                if error:
                    st.error(error)
                else:
                    st.success(f"Top {top_n_hybrid} Recommendations for User {user_id_hybrid}")
                    st.dataframe(result)


# In[ ]:




