#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import json
import nltk
import pickle
import spacy
import en_core_web_sm

from datetime import date, timedelta
from IPython import get_ipython
from PIL import Image
from streamlit import caching
import matplotlib.dates as mdates
import plotly.graph_objects as go

from nltk.corpus import stopwords
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components

from collections import Counter
import re
import math
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from streamlit_embedcode import github_gist
from wordcloud import WordCloud

with open("style_img.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
image = Image.open('img/santa.png')
st.image(image, width = 500)

# st.markdown("<h1 style='text-align: center; color: #930101;'>Santa All : Grant a Wish</h1>", unsafe_allow_html=True)

image = Image.open('img/eskwelabs_logo.jpg')
st.sidebar.image(image, caption='', use_column_width=True)

add_selectbox = st.sidebar.radio(
    "",
    ("Introduction and Problem Statement", "List of Tools", "Data Sourcing", "Feature Engineering", 
     "Exploratory Data Analysis", "Recommender Engine", 
     "Recommendations", "Contributors")
)

if add_selectbox == 'Introduction and Problem Statement':
    st.write('')
    
    image = Image.open('img/intro_img.png').convert('RGB')
    st.image(image, caption='Source:https://unsplash.com/photos/2plsuhgV53I', width=680)
    
    st.write("""
    Gift giving during the holiday season is inherent in Filipinos and it is traditionally coupled with holiday 
    shopping in malls and other offline stores. But because of the pandemic, majority of this traditionally offline 
    ritual will be happening online and people will be looking to e-commerce marketplaces to make their holiday shopping 
    easier and safer.
    <br>
    <br>
    Our project aims to create a feature that will support the gift-giving activities from consumers in the form of gift 
    recommendations and public wishlists supplemented with features that will help users build wishlists. Having this 
    feature that supports this already inherent behavior will give an e-commerce player an advantage over competitors by 
    attracting shoppers to go to their platform due to ease and convenience instead of the competitors’.
    """, unsafe_allow_html=True)

elif add_selectbox == 'List of Tools':
    st.subheader('List of Tools')
    
    st.write('___') 
    
    st.write('''**Integrated Development Environment:**''', unsafe_allow_html=True)
    image = Image.open('img/anaconda.png').convert('RGB')
    st.image(image, caption='Anaconda', width=300, height=150)
    image = Image.open('img/jupyter.png').convert('RGB')
    st.image(image, caption='Jupyter Notebook', width=300, height=150)
    
    st.write('''<br>**Main Programming Language:**''', unsafe_allow_html=True)
    image = Image.open('img/python.png').convert('RGB')
    st.image(image, caption='Python', width=300, height=150)
    
    st.write('''<br>**Data Visualization and Exploratory Data Analysis:**''', unsafe_allow_html=True)
    image = Image.open('img/pandas.png').convert('RGB')
    st.image(image, caption='Pandas', width=300, height=150)
    image = Image.open('img/seaborn.png').convert('RGB')
    st.image(image, caption='Seaborn', width=300, height=150)
    image = Image.open('img/matplotlib.png').convert('RGB')
    st.image(image, caption='Matplotlib', width=300, height=150)
    
    st.write('''<br>**Modeling:**''', unsafe_allow_html=True)
    image = Image.open('img/spacy.png').convert('RGB')
    st.image(image, caption='spaCy', width=300, height=150)
    image = Image.open('img/networkx.png').convert('RGB')
    st.image(image, caption='NetworkX', width=300, height=150)
    image = Image.open('img/sklearn.png').convert('RGB')
    st.image(image, caption='Scikit-learn', width=300, height=150)
    
    st.write('''<br>**Deployment:**''', unsafe_allow_html=True)
    image = Image.open('img/streamlit.png').convert('RGB')
    st.image(image, caption='Streamlit', width=300, height=150)
    image = Image.open('img/heroku.png').convert('RGB')
    st.image(image, caption='Heroku', width=300, height=150)

elif add_selectbox == 'Data Sourcing':
    st.subheader('Data Sourcing')
    
    shopee_links = pd.read_csv('data/Mobiles-Gadgets-cat.24456_links.csv', error_bad_lines=False)
    shopee_data = pd.read_csv('data/Mobiles-Gadgets-cat.24456.csv', error_bad_lines=False)
    
    st.write("""
    For this project, our group chose to scrape product data from one of the Philippines' leading e-commerce platforms, Shopee.
    Since Shopee utilizes Lazy Loading, we used BeautifulSoup together with Selenium to automate the data scraping process.
    <br>
    <br>""", unsafe_allow_html=True)
    
    image = Image.open('img/main_categories.png').convert('RGB')
    st.image(image, caption='Shopee Main Categories', height=300, width=680)
    
    st.write("""
    The first step in our data scraping process was to make sure that all products of each Main Category in Shopee are represented
    equally. For this to happen, we needed to scrape all the URL link of each product. The image below shows a sample of the automated process
    of scraping URL links from each product in Mobiles & Gadgets Category. Our group repeated this process for all other categories in Shopee.
    <br>
    <br>""", unsafe_allow_html=True)
    
#     file_ = open("img/shopee_link.gif", "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()
#     st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="shopee_link gif">',unsafe_allow_html=True)
    st.image('img/shopee_link.gif', width=680)
    
    st.write("""
    <br>
    After the scraping process for URL links, a CSV file will be created with the following contents.
    <br>""", unsafe_allow_html=True)
    st.table(shopee_links.head(1))
    
    st.write('<b>Main Category List:</b>', unsafe_allow_html=True)
    st.markdown("<ul>"                "<li>Babies & Kids</li>"                "<li>Cameras</li>"
                "<li>Digital Goods and Vouchers</li>"\
                "<li>Gaming</li>"\
                "<li>Groceries</li>"\
                "<li>Health and Personal Care</li>"\
                "<li>Hobbies and Stationery</li>"\
                "<li>Home and Living</li>"\
                "<li>Home Appliances</li>"\
                "<li>Home Entertainment</li>"\
                "<li>Laptops and Computers</li>"\
                "<li>Makeup and Fragrances</li>"\
                "<li>Men's Apparel</li>"\
                "<li>Men's Bags and Accessories</li>"\
                "<li>Men's Shoes</li>"\
                "<li>Mobile Accessories</li>"\
                "<li>Mobiles and Gadgets</li>"\
                "<li>Motors</li>"\
                "<li>Pet Care</li>"\
                "<li>Sports and Travel</li>"\
                "<li>Toys, Games & Collectibles</li>"\
                "<li>Women's Accessories</li>"\
                "<li>Women's Apparel</li>"\
                "<li>Women's Bags</li>"\
                "<li>Women's Shoes</li>"\
                 "</ul>", unsafe_allow_html=True)
    
    st.write("""
    After scraping each product URL, we then proceed to the product details. Our group again used BeautifulSoup and Selenium to scrape 
    each product details that we have deemed useful in building our recommender engine. 
    <br>
    <br>""", unsafe_allow_html=True)
    
#     file_ = open("img/shopee_detail.gif", "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()
#     st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="shopee_detail gif">',unsafe_allow_html=True)
    st.image('img/shopee_detail.gif', width=680)
    
    st.write("""
    <br>
    Shown below is the CSV file that was created after scraping the details of each product under the category
    for Men's Bags & Accessories.
    <br>
    <br>""", unsafe_allow_html=True)
    st.dataframe(shopee_data, width=680)
    
elif add_selectbox == 'Feature Engineering':
    st.subheader('Feature Engineering')
    
    DATA_URL = ('data/cleaned_data_1000 (11.23.2020).csv')
    
    @st.cache
    def load_data(nrows):
        overall_data = pd.read_csv(DATA_URL, nrows=nrows)
        return overall_data
    overall_data = load_data(101)
    
    st.write("""
    After scraping all product details in each category, we now proceed in merging it all in to one CSV file. We used the code
    below to do this.
    <br>
    <br>""", unsafe_allow_html=True)

    github_gist("https://gist.github.com/robibanadera/f591993a9fbbe0777c156a3177fe5664/",height=300, width=680)
    
    st.write("""
    After this process, we can now proceed with data cleaning and feature engineering. As we all know, data cleaning is an
    important part of data preprocessing. It lets data scientist improve the results of their findings by removing or
    modifying data that is incorrect, incomplete, irrelevant, duplicated, or improperly formatted.
    <br>
    <br>
    For this dataset, we start this off by removing duplicated rows and also rows that have null values in the 'Product Name'
    column. We also chose to drop some columns that our group deemed to have little or no importance in the goal that we are
    trying to achieve. Through this process, we would have rows that have complete values for each column thus leaving us 
    with a more robust dataset.
    <br>
    <br>
    The next step would be cleaning up the data from columns having unneeded characters and/or repetitive symbols 
    (eg. Currency $). This process is importance especially for columns dealing with numbers but then can't be processed by
    Python because of the values being a string in nature rather than an integer. We used the code embedded below to accomplish
    this step.
    <br>
    """, unsafe_allow_html=True)
    
    github_gist("https://gist.github.com/robibanadera/fc4c21d81956fdd0759cbcd987553187/",height=300, width=680)
    
    st.write("""
    After removing unnecessary symbols/character in numerical columns, we now proceed to converting it to integer or float
    values. We also checked the data types for the other columns if it is correct so no further problems will occur during coding
    in the future.
    <br>
    <br>
    You can now see below the sample for final dataset that we will use moving forward.
    """, unsafe_allow_html=True)
    
    overall_data = overall_data.drop(['Description'], axis=1)
    st.dataframe(overall_data, width=680)
    st.write("""
    <b>Dataset Shape:</b>
    <br>
    Rows: 20918
    <br>
    Columns: 42
    <br>
    <br>
    """, unsafe_allow_html=True)
    
    st.write('<b>Dataset Features:</b>', unsafe_allow_html=True)
    st.markdown("<ul>"                "<li>Product ID</li>"                "<li>URL</li>"
                "<li>Page</li>"\
                "<li>Preferred</li>"\
                "<li>Mall</li>"\
                "<li>Product Name</li>"\
                "<li>Main Category</li>"\
                "<li>Sub Category 1</li>"\
                "<li>Sub Category 2</li>"\
                "<li>Current Rating</li>"\
                "<li>Total Rating</li>"\
                "<li>Total Sold</li>"\
                "<li>Favorite</li>"\
                "<li>Discount Range</li>"\
                "<li>Price Range</li>"\
                "<li>Discount Percentage</li>"\
                "<li>Free Shipping</li>"\
                "<li>Free Shipping Info</li>"\
                "<li>Shipping Location</li>"\
                "<li>Shipping Price Range</li>"\
                "<li>Brand Name</li>"\
                "<li>Store Name</li>"\
                "<li>Store Ratings</li>"\
                "<li>Store Products Count</li>"\
                "<li>Store Response Rate</li>"\
                "<li>Store Response Time</li>"\
                "<li>Store Joined</li>"\
                "<li>Store Followers</li>"\
                "<li>Shipping From</li>"\
                "<li>Vouchers Available</li>"\
                "<li>Bundle Details</li>"\
                "<li>Coins Available</li>"\
                "<li>Product Variation List</li>"\
                "<li>Lowest Price Guarantee</li>"\
                "<li>Whole Sale</li>"\
                "<li>Five Star</li>"\
                "<li>Four Star</li>"\
                "<li>Three Star</li>"\
                "<li>Two Star</li>"\
                "<li>One Star</li>"\
                "<li>With Comments</li>"\
                "<li>With Media</li>"\
                 "</ul>", unsafe_allow_html=True)
    
elif add_selectbox == 'Exploratory Data Analysis':
    st.subheader('Exploratory Data Analysis')
    
    df = pd.read_csv('data/cleaned_data_1000 (11.23.2020).csv', index_col=0)
    
    fig = plt.figure(figsize=(13,4))
    Rating = ['Five Star', 'Four Star', 'Three Star', 'Two Star', 'One Star']
    Count = list(df[['Five Star', 'Four Star', 'Three Star', 'Two Star', 'One Star' ]].sum())
    plt.bar(Rating, Count)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(fig)
    
    st.write("""
    The bar graph above shows the distribution of the rating. We can observe that the distribution of the data 
    in the current ratings is skewed to five stars. This shows that Shopee users tend to be generous in 
    providing their ratings to their purchased products from Shopee.
    <br>
    """, unsafe_allow_html=True)
    
    st.write("___")
    
    fig = plt.figure(figsize=(13,4))
    ax = sns.kdeplot(df['Current Rating'], shade=True, color='orange')
    st.pyplot(fig)
    
    st.write("""
    Looking at the distribution of the average rating of the products, we can observe that it is skewed as 
    expected. It validates our first observation that SHoppee users are very generous in proviing ratings to 
    their purchased products.
    <br>
    """, unsafe_allow_html=True)
    
    st.write("___")
    
    fig = plt.figure(figsize=(13,4))
    ax = sns.kdeplot(df['Total Rating'], shade=True, color='orange')
    st.pyplot(fig)
    
    st.write("""
    Plotting the distribution of the total ratings of each product, we can see that most of the products 
    receive around 500 to 2000 reviews. In addition, there are very few prodjucts with more than 20,000 
    total reviews.
    <br>
    """, unsafe_allow_html=True)
    
    st.write("___")
    
    fig = plt.figure(figsize=(13,4))
    ax = sns.kdeplot(df['Favorite'], shade=True, color='orange')
    st.pyplot(fig)
    
    st.write("""
    Plotting the distribution of the total favorites of each product, we can see that most of 
    the products receive around 500 to 2000 reviews. In addition, there are very few prodjucts with more 
    than 10,000 total reviews.
    <br>
    """, unsafe_allow_html=True)
    
    st.write("___")
    
    included = ['Preferred','Mall', 'Current Rating', 'Total Sold', 'Favorite', 'Lowest Price Guarantee']
    df1 = df[included]
    corr = df1.corr()
    fig = plt.figure(figsize=(13, 10))
    ax = sns.heatmap(corr, annot=True, center=0)
    st.pyplot(fig)
    
    st.write("""
    From the correlation heatmap, we can observe that most of the features are not highly correlated to 
    each other except total sold and favorite features with a correlatio of 0.78.
    <br>
    """, unsafe_allow_html=True)
    
    st.write("___")
    
    st.write("""
    From the word cloud below we generated from the product names, we can see that most of the products 
    contains the words kids, women, old, pajama and terno the most.
    <br>
    """, unsafe_allow_html=True)
    
    image = Image.open('img/wordcloud/Babies & Kids.png').convert('RGB')
    st.image(image, caption='Babies & Kids', width=680)
    
    st.write("___")

    image = Image.open('img/wordcloud/Cameras.png').convert('RGB')
    st.image(image, caption='Cameras', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Gaming.png').convert('RGB')
    st.image(image, caption='Gaming', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Health & Personal Care.png').convert('RGB')
    st.image(image, caption='Health & Personal Care', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Hobbies & Stationery.png').convert('RGB')
    st.image(image, caption='Hobbies & Stationery', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Home & Living.png').convert('RGB')
    st.image(image, caption='Home & Living', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Home Appliances.png').convert('RGB')
    st.image(image, caption='Home Appliances', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Home Entertainment.png').convert('RGB')
    st.image(image, caption='Home Entertainment', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Laptops & Computers.png').convert('RGB')
    st.image(image, caption='Laptops & Computers', width=680)
    
    st.write("___")
    
    image = Image.open('img/wordcloud/Makeup & Fragrances.png').convert('RGB')
    st.image(image, caption='Makeup & Fragrances', width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Men's Apparel.png").convert('RGB')
    st.image(image, caption="Men's Apparel", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Men's Bags & Accessories.png").convert('RGB')
    st.image(image, caption="Men's Bags & Accessories", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Men's Shoes.png").convert('RGB')
    st.image(image, caption="Men's Shoes", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Mobiles & Gadgets.png").convert('RGB')
    st.image(image, caption="Mobiles & Gadgets", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Motors.png").convert('RGB')
    st.image(image, caption="Motors", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Pet Care.png").convert('RGB')
    st.image(image, caption="Pet Care", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Sports & Travel.png").convert('RGB')
    st.image(image, caption="Sports & Travel", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Women's Accessories.png").convert('RGB')
    st.image(image, caption="Women's Accessories", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Women's Apparel.png").convert('RGB')
    st.image(image, caption="Women's Apparel", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Women's Bags.png").convert('RGB')
    st.image(image, caption="Women's Bags", width=680)
    
    st.write("___")
    
    image = Image.open("img/wordcloud/Women's Bags.png").convert('RGB')
    st.image(image, caption="Women's Bags", width=680)
    
#     main_category = list(df['Main Category'].unique())

#     for category in main_category:
#         categorystr = str(category)
#         categorystr = " ".join(prod_name for prod_name in df[df["Main Category"]== str(category)]['Product Name'])
#         wordcloud = WordCloud(width=1500, height=500, background_color="white").generate(categorystr)
#         fig = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
#         ax = plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis("off")
#         st.write(category)
#         st.pyplot(fig)
    
elif add_selectbox == 'Recommender Engine':
    
#     with open("style_img.css") as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
#     image = Image.open('img/santa.png')
#     st.image(image, width = 150)
    
    st.subheader('Recommender Engine')
    
    overall_data = pd.read_csv('data/cleaned_data_1000 (11.23.2020).csv', error_bad_lines=False)
    st.write("What's your wish?")
    
    sub_category_list_2 = list(overall_data['Sub Category 2'].str.lower().unique())
    
    with open('network_theory.pickle','rb') as fe_data_file:
         G = pickle.load(fe_data_file)

    with open('betweenness_centraility.json') as f:
         between_centrality_json = json.load(f)
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#     def remote_css(url):
#         st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

    def icon(icon_name):
        st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

    local_css("style.css")
#     remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
    
#########################   

    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words('english'))

    def get_tfidf(product_details):
        clean_product = []
        product_name = list(product_details)
        for i in range(len(product_name)):
            words = ""

            doc = nlp(product_name[i].lower())
            for token in doc:
                token.lemma_ = re.sub(r'\W',' ',token.lemma_)
                token.lemma_ = token.lemma_.strip()
                if not token.lemma_.endswith("ml") and not token.lemma_.endswith("ms") and not token.lemma_.isdigit() and not token.lemma_ in stop_words:
                    if len(token.lemma_) > 2 or token.lemma_ == 'uv': 
                        words += token.lemma_.lower() + " "
                    

            if len(words) > 0:
                clean_product.append(str(words.strip()))

        tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
        tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(clean_product)
        first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), 
                      columns=["tfidf"]) 
        df = df.sort_values(by=["tfidf"], ascending=False).reset_index()
    
        return df
#########################
    user_input = st.text_area("What are you looking for?")
    if st.button('Search'):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(user_input.strip())

        result_categories = []

        for token in reversed(doc):
            if token.text in list(G.nodes()):
#         print(token.lemma_)
                closeness_centrality_list = []
                betweness_centrality_list = []
                degree_list = []
                neighbor_list = []
                shortest_path_list = []
                length_list = []

                for _neighbors in list(G.neighbors(token.text)):
                    if _neighbors in sub_category_list_2:
                        neighbor_list.append(_neighbors)
                        betweness_centrality_list.append(between_centrality_json[_neighbors])
                        shortest_path = nx.shortest_path(G, source=_neighbors, target=token.lemma_)
                        shortest_path_list.append(len(shortest_path))
                        length_list.append(overall_data.loc[overall_data['Sub Category 2'] == _neighbors].shape[0])

                network_result = pd.DataFrame(neighbor_list, columns=['neighbor'])
                network_result['betweeness_centrality'] = betweness_centrality_list
                network_result['shortest_path'] = shortest_path_list

                if len(betweness_centrality_list) > 0:
                    if network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'].shape[0] < 2:
                        if list(network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'])[0] not in result_categories:
                            result_categories.append(list(network_result[network_result['shortest_path'] == min(shortest_path_list)]['neighbor'])[0])
                    else:
                        if list(network_result[network_result['betweeness_centrality'] == min(betweness_centrality_list)]['neighbor'])[0] not in result_categories:
                            result_categories.append(list(network_result[network_result['betweeness_centrality'] == min(betweness_centrality_list)]['neighbor'])[0]) 
        merge_products = []
        for _result_categories in result_categories:
            merge_products.append(overall_data.loc[(overall_data['Sub Category 2'] == _result_categories.title())])

        selected_category = pd.concat(merge_products).reset_index()

#         from sklearn.neighbors import NearestNeighbors
        vectorize = TfidfVectorizer(stop_words='english')
        tfidf_response= vectorize.fit_transform(selected_category['Product Name'])
        dtm = pd.DataFrame(tfidf_response.todense(), columns = vectorize.get_feature_names())

        nn = NearestNeighbors(n_neighbors=selected_category.shape[0])
        nn.fit(dtm)
        wishlist = [user_input]

# print(wishlist)
        new = vectorize.transform(wishlist)
        knn_model_result = nn.kneighbors(new.todense())

        knn_result = pd.DataFrame(knn_model_result[0][0][0:], columns=['Distance'])
        knn_result["Product Name"] = selected_category['Product Name'][knn_model_result[1][0][0:]]

        merged_result = pd.merge(selected_category, knn_result, on='Product Name', how='inner')
        merged_result = merged_result.drop_duplicates(subset='Product Name', keep="first")

#         import numpy as np
#         import pandas as pd
#         from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        scoring_criteria = ['Trusted', 'Highly Rated', 'Discounted', 'Top Selling', 'High Interest']
        df_eng = merged_result.copy()
        df_eng['Discount Percent'] = df_eng['Discount Range']-df_eng['Price Range']
        df_eng['Total Sold'] = scaler.fit_transform(df_eng[['Total Sold']])
        df_eng['High Interest'] = scaler.fit_transform(df_eng[['Favorite']])

# Conditions
        df_eng['Highly Rated'] = df_eng['Current Rating'].astype(float).map(lambda x: True if x>df_eng['Current Rating'].mean() else False)
        df_eng['Discounted'] = df_eng['Discount Percent'].astype(float).map(lambda x: True if x>0.03 else False)
        df_eng['Top Selling'] = df_eng['Total Sold'].map(lambda x: True if x>df_eng['Total Sold'].mean() else False)
        df_eng['High Interest'] = df_eng['High Interest'].map(lambda x: True if x>df_eng['High Interest'].mean() else False)

# New Columns
        df_eng['Trusted'] = df_eng.apply(lambda x: x['Preferred'] | x['Mall'], axis=1)

        model_features = ['Price Range']
        scoring_criteria = ['Trusted', 'Highly Rated', 'Discounted', 'Top Selling', 'High Interest']

        if df_eng.shape[0] < 10:
            prd_list = df_eng.sample(n=df_eng.shape[0])
        else:
            prd_list = df_eng.sample(n=10)
        prd_list['Relevance'] = prd_list['Distance']
        prd_list['score'] = prd_list['Relevance']

        scored_list = prd_list[prd_list['Current Rating'] > 3.8]

# Scoring System
        trusted_bias = 0.05
        highly_rated_bias = 0.05
        discounted_bias = 0.05
        top_selling_bias = 0.05
        high_interest_bias = 0.05

        scored_list['score'] = scored_list.apply(lambda x: x['score']-trusted_bias if x['Trusted'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']-highly_rated_bias if x['Highly Rated'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']-discounted_bias if x['Discounted'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']-top_selling_bias if x['Top Selling'] == True else x['score'], axis=1)
        scored_list['score'] = scored_list.apply(lambda x: x['score']-high_interest_bias if x['High Interest'] == True else x['score'], axis=1)
        scored_list['Performance'] = scored_list['score']
        scored_list2 = scored_list[['Product Name', 'Distance', 'Trusted', 'Highly Rated', 'Discounted', 'Top Selling', 'High Interest', 'Performance']].sort_values(by=['Performance'], ascending=True)
        st.table(scored_list2)
        
        nearest = scored_list
        
        st.write('''**Top-selling Products:**''', unsafe_allow_html=True)
        top_sellers = pd.DataFrame(nearest['Total Sold'].sort_values(ascending=False)[:3].reset_index())
        top_sellers = top_sellers['index'].tolist()

        for i in top_sellers:
            st.table(nearest.loc[[i]][['Product Name']])

        st.write('''**Most Popular Products:**''', unsafe_allow_html=True)
        top_popular = pd.DataFrame(nearest['Favorite'].sort_values(ascending=False)[:3].reset_index())
        top_popular = top_popular['index'].tolist()

        for i in top_popular:
            st.table(nearest.loc[[i]][['Product Name']])
        
        st.write('''**Top Discounted Products:**''', unsafe_allow_html=True)
        top_discount = pd.DataFrame(nearest['Discount Range'].sort_values(ascending=False)[:3].reset_index())
        top_discount = top_discount['index'].tolist()

        for i in top_discount:
            st.table(nearest.loc[[i]][['Product Name']])
        
        st.write('''**Check out this stores!**''', unsafe_allow_html=True)
        top_related = nearest[(nearest['Mall']==True) | (nearest['Preferred']==True)]['Store Name'].unique()[:3]
        top_related = pd.DataFrame(top_related)
        top_related = top_related.rename(columns={0: "Store Name"})
        st.table(top_related['Store Name'])
        
#########################

#         from collections import Counter

        recommend = selected_category['Sub Category 2'][knn_model_result[1][0][0:]].tolist()
        counter = Counter(recommend)

        to_recommend = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
        to_recommend = [x for x in to_recommend.keys() if str(x) != 'nan']

        st.write('''**You might also like:**''', unsafe_allow_html=True)
        for i in to_recommend:
            st.write(i)
            st.table(overall_data[overall_data['Sub Category 2'] == i]['Product Name'].sample(n=1, replace=True))
            
    if st.button('Random'):
        df_random = overall_data.loc[(overall_data['Total Sold'] > overall_data['Total Sold'].mean()) | (overall_data['Current Rating']>overall_data['Current Rating'].mean()) | ((overall_data['Preferred'] ==True) | (overall_data['Mall'] ==True))]
        main_cat_list = Counter(df_random['Main Category'].tolist())
        main_cat_list= [x for x in main_cat_list.keys() if str(x) != 'nan']
        for main_cat in main_cat_list:
            st.subheader(main_cat)
            st.table(df_random[df_random['Main Category'] ==  main_cat]['Product Name'].sample(n=3, replace=True))

elif add_selectbox == 'Recommendations':
    st.subheader('Recommendations')
    
    st.markdown("<ul>"                "<li>Consider a hybrid filtering approach to the recommender system.</li>"                "<BLOCKQUOTE><li>Transactions data is needed</li></BLOCKQUOTE>"                "<li>Sentiment analysis on comments because ratings aren’t accurate.</li>"                "<li>Utilize spacy linguistic features to filter out results.</li>"                "<li>Consider making a third-party wishlist site that gathers data from other e-commerce marketplaces.</li>"                "<li>Save information to neo4j or MongoDB, a graph theory database.</li>"                 "</ul>", unsafe_allow_html=True)
            
else:
    st.subheader('Contributors')
    st.write('___')
    
    with open("style_img.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    
    image = Image.open('img/dan.png')
    st.image(image, width = 300)
    
    st.markdown(
    """<a style='display: block; text-align: center;'>**Danilo Gubaton Jr.**</a>
    """,
    unsafe_allow_html=True,)

    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/dcgubatonjr/">LinkedIn</a>
    """,
    unsafe_allow_html=True,)
    
    st.write('___')
    
    image = Image.open('img/emer.png')
    st.image(image, width = 300)
    
    st.markdown(
    """<a style='display: block; text-align: center;'>**Fili Emerson Chua**</a>
    """,
    unsafe_allow_html=True,)

    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/fili-emerson-chua/">LinkedIn</a>
    """,
    unsafe_allow_html=True,)
    
    st.write('___')

    image = Image.open('img/ran.png')
    st.image(image, width = 300)

    st.markdown(
    """<a style='display: block; text-align: center;'>**Rhey Ann Magcalas**</a>
    """,
    unsafe_allow_html=True,)

    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/rhey-magcalas-47541490/">LinkedIn</a>
    """,
    unsafe_allow_html=True,)
    
    st.write('___')
    
    image = Image.open('img/rob.png')
    st.image(image, width = 300)
    
    st.markdown(
    """<a style='display: block; text-align: center;'>**Roberto Bañadera Jr.**</a>
    """,
    unsafe_allow_html=True,)

    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/robertobanaderajr/">LinkedIn</a>
    """,
    unsafe_allow_html=True,)

