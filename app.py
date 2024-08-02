import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_groq import ChatGroq
import requests

#####################################
#            FUNCTIONS              #
#####################################
@st.cache_data()
def load_data(url):
    """
    Load data from URL
    """
    df = pd.read_csv(url)
    return df

def prepare_data(df):
    """
    Lowercase columns
    """
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

def extract_relevant_info(df):
    """
    Extract relevant information from DataFrame
    """
    employee_names = df['employee_name'].tolist()
    return employee_names

def plot_individual_performance(df, rep_id):
    rep_data = df[df['employee_id'] == rep_id]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    days = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    for day in days:
        if day+'_text' in rep_data.columns:
            ax[0].plot(rep_data['dated'], rep_data[day+'_text'], label=day.capitalize())
            ax[0].set_title(f"Individual Sales for Representative {rep_id}")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Sales")
            ax[0].legend()
        if day+'_call' in rep_data.columns:
            ax[1].plot(rep_data['dated'], rep_data[day+'_call'], label=day.capitalize())
            ax[1].set_title(f"Individual Calls for Representative {rep_id}")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Calls")
            ax[1].legend()
    st.pyplot(fig)

def plot_team_performance(df):
    team_data = df.groupby('dated').agg({'revenue_confirmed': 'sum', 'revenue_pending': 'sum'}).reset_index()
    
    fig, ax = plt.subplots()
    ax.plot(team_data['dated'], team_data['revenue_confirmed'], label='Total Confirmed Revenue')
    ax.plot(team_data['dated'], team_data['revenue_pending'], label='Total Pending Revenue')
    ax.set_title("Overall Sales Team Performance Summary")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    st.pyplot(fig)
    
def plot_performance_trends(df):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    days = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    for day in days:
        if day+'_text' in df.columns:
            ax[0].bar(df['dated'], df[day+'_text'], label=day.capitalize())
            ax[0].set_title(f"Sales Performance Trends ({day.capitalize()})")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Sales")
            ax[0].legend()
        if day+'_call' in df.columns:
            ax[1].bar(df['dated'], df[day+'_call'], label=day.capitalize())
            ax[1].set_title(f"Call Performance Trends ({day.capitalize()})")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Calls")
            ax[1].legend()
    st.pyplot(fig)

#####################################
#        LOCALS & CONSTANTS         #
#####################################
table_name = 'sales_data'

#####################################
#            HOME PAGE              #
#####################################
st.title('Sales Team Performance Analysis')

# Read data
url = "https://raw.githubusercontent.com/zainulabideen3434/Sales_Analysis_Excercise/main/sales_performance_data.csv"
df = load_data(url)

# Display the entire dataset
show_entire_dataset = st.checkbox("Show entire dataset", False)
st.subheader('Raw Dataset')
if show_entire_dataset:
    st.write(df)
else:
    st.write(df.head(5))

# API key
groq_api_key = st.text_input(
    "API key", 
    placeholder='Type your Groq API Key',
    type='password',
    disabled=False,
    help='Enter your Groq API key.'
)

# User query
user_query = st.text_input(
    "User Query", 
    placeholder="Enter your query",
    help="Enter a question based on the dataset"
)

# Commit data to SQL
data = prepare_data(df)
conn = sqlite3.connect(':memory:')
data.to_sql(table_name, conn, if_exists='replace', index=False)

# Create DB engine
eng = create_engine(
    'sqlite://', 
    poolclass=StaticPool, 
    creator=lambda: conn
)

# Create Groq connection
if groq_api_key:
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="gemma2-9b-it"
    )

# Run query and display result
if groq_api_key and user_query:
    try:
        # Extract relevant information from the DataFrame
        relevant_info = extract_relevant_info(df)
        
        # Use Groq to generate text based on the extracted information
        prompt = f"Provide insights based on the following data: {relevant_info}. User query: {user_query}"
        response = llm.generate(prompts=prompt)
        st.write(response['choices'][0]['text'])
    except Exception as e:
        st.error(f"Error executing Groq query: {e}")

# FUNCTIONALITIES
if st.button("Individual Sales Representative Performance Analysis"):
    rep_id = st.text_input("Enter Employee ID")
    if rep_id:
        plot_individual_performance(df, rep_id)

if st.button("Overall Sales Team Performance Summary"):
    plot_team_performance(df)

if st.button("Sales Performance Trends and Forecasting"):
    plot_performance_trends(df)
