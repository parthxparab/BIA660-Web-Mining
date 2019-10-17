#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Web Scraping

# ## Q1. Scrape Book Catalog 
# - Scape content of http://books.toscrape.com 
# - Write a function getData() to scrape **title** (see (1) in Figure), **rating** (see (2) in Figure), **price** (see (3) in Figure) of all books (i.e. 20 books) listed in the page.
#   * For example, the figure shows one book and the corresponding html code. You need to scrape the highlighted content. 
#   * For star ratings, you can simply scrape One, Two, Three, ... 
# - The output is a list of 20 tuples, e.g. [('A Light in the ...','Three','£51.77'), ...] 
#     <img src='assign3_q1.png' width='80%'>
# 

# ## Q2. Data Analysis 
# - Create a function preprocess_data which 
#   * takes the list of tuples from Q1 as an input
#   * converts the price strings to numbers 
#   * calculates the average price of books by ratings 
#   * plots a bar chart to show the average price by ratings. 

# ### Q3 (Bonus) Expand your solution to Q1 to scrape the full details of all books on http://books.toscrape.com
# - Write a function getFullData() to do the following: 
#    * Besides scraping title, rating, and price of each book as stated in Q1, also scrape the **full title** (see (4) in Figure), **description** (see (5) in Figure), and **category** (see (6) in Figure) in each individual book page. 
#      * An example individual book page is shown in the figure below.
#        <img src='assign3_q3a.png' width='60%'>
#    
#    * Scape all book listing pages following the "next" link at the bottom. The figure below gives an screenshot of the "next" link and its corresponding html code. 
#    * <b>Do not hardcode page URLs </b>(except http://books.toscrape.com) in your code. 
#       <img src='assign3_q3.png' width='80%'>
#    * The output is a list containing 1000 tuples, 
#      - e.g. [('A Light in the ...','Three','£51.77', 'A Light in the Attic', "It's hard to imagine a world without A Light in the Attic. This now-classic collection ...",'Poetry'), ...]
#     

# In[33]:


import requests
import re
from bs4 import BeautifulSoup  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions 
from selenium.webdriver.common.keys import Keys
InteractiveShell.ast_node_interactivity = "all"
from selenium.common.exceptions import NoSuchElementException

def preprocess_data(data):
    
    # add your code
    for index, item in enumerate(data):
        itemlist = list(item)
        var = itemlist[2]
        itemlist[2] = float(var[1:])
        item = tuple(itemlist)

        data[index] = item
    #print(data)
    
    df = pd.DataFrame()
    avgOne=0
    cOne=0
    avgTwo=0
    cTwo=0
    avgThree=0
    cThree=0
    avgFour=0
    cFour=0
    avgFive=0
    cFive=0
    
    for i in range(0,len(data)):
        if(data[i][1]=='One'):
            avgOne = avgOne + data[i][2]
            cOne=cOne+1
        if(data[i][1]=='Two'):
            avgTwo = avgTwo + data[i][2]
            cTwo=cTwo+1
        if(data[i][1]=='Three'):
            avgThree = avgThree + data[i][2]
            cThree=cThree+1
        if(data[i][1]=='Four'):
            avgFour = avgFour + data[i][2]
            cFour=cFour+1
        if(data[i][1]=='Five'):
            avgFive = avgFive + data[i][2]
            cFive=cFive+1
    avgOne=avgOne/cOne
    avgTwo=avgTwo/cTwo
    avgThree=avgThree/cThree
    avgFour=avgFour/cFour
    avgFive=avgFive/cFive
    
    avgPrice=[avgOne,avgTwo,avgThree,avgFour,avgFive]
    rating=['One','Two','Three','Four','Five']

    df = pd.DataFrame({
         'Average Price': avgPrice,
         'Rating': rating
    })
    
    
    
   # print(df)
    ax = plt.gca()

    df.plot(kind='bar',x='Rating',y='Average Price',title="average price by rating",ax=ax)
    ax.set(ylabel="Average Price", xlabel="Rating");
    ax.get_legend().remove()



    plt.show()    

def getData():

    page = requests.get("http://books.toscrape.com/")    # send a get request to the web page
    data=[]
    if page.status_code==200:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs=soup.select("body div div div div div ol li article.product_pod")    
        #print(divs)
    for idx, div in enumerate(divs):
        title=None
        rating=None
        price=None
        
        p_title=div.select("a")


        if p_title!=[]:
            title=p_title[1].get_text()
        
        p_price=div.select("p.price_color")
        if p_price!=[]:
            price=p_price[0].get_text()
        
        p_star=div.select("p.star-rating")
        if p_star!=[]:
            rating=p_star[0].get("class")[1]
            
        data.append((title, rating, price))
    return data

def getFullData(): 
    
    data=[]# variable to hold all book data
    data1 = []
    
    page_url="http://books.toscrape.com"
    new_page = page_url
    
   # add your code


    def getBooksURLS(url):
        soup = getAndParseURL(url)
        return(["/".join(url.split("/")[:-1]) + "/" + x.div.a.get('href') for x in soup.findAll("article", class_ = "product_pod")])
    def getAndParseURL(url):
        result = requests.get(url)
        soup = BeautifulSoup(result.text, 'html.parser')
        return(soup)
        
    executable_path = '/Users/parthxparab/Documents/Fall 2019/BIA660/Lab2/Web_Scraping/geckodriver'

    # initiator the webdriver for Firefox browser
    driver = webdriver.Firefox(executable_path=executable_path)


 #   driver.implicitly_wait(20)

    driver.get('http://books.toscrape.com')
    
    name = None
    prices = None
    description = None
    categories = None
    ratings = None
    a = None
    
    pages_urls = []
    
    while requests.get(new_page).status_code == 200:
        pages_urls.append(driver.current_url)
        try:
            more_link=driver.find_element_by_css_selector('div ul li.next a')

            if requests.get(new_page).status_code ==200:
                soup = BeautifulSoup(requests.get(new_page).content, 'html.parser')
                divs=soup.select("body div div div div div ol li article.product_pod")    
    #print(divs)
            for idx, div in enumerate(divs):
                title=None

                p_title=div.select("a")

                if p_title!=[]:
                    title=p_title[1].get_text()


                data1.append(title)

#####
#     new_page = pages_urls[-1].split("-")[0] + "-" + str(int(pages_urls[-1].split("-")[1].split(".")[0]) + 1) + ".html"
            new_page=driver.current_url    


            more_link.click()    
        except NoSuchElementException:
            if requests.get(new_page).status_code ==200:
                soup = BeautifulSoup(requests.get(pages_urls[-1]).content, 'html.parser')
                divs=soup.select("body div div div div div ol li article.product_pod")    
    #print(divs)
            for idx, div in enumerate(divs):
                title=None

                p_title=div.select("a")

                if p_title!=[]:
                    title=p_title[1].get_text()


                data1.append(title)
            break
    count = 0
    booksURLs = []
    for page in pages_urls:
        booksURLs.extend(getBooksURLS(page))
        
    for url in booksURLs:
        soup = getAndParseURL(url)
    
        name = soup.find("div", class_ = re.compile("product_main")).h1.text
        
        prices = soup.find("p", class_ = "price_color").text[1:]

        categories = soup.find("a", href = re.compile("../category/books/")).text

        ratings = soup.find("p", class_ = re.compile("star-rating")).get("class")[1]
        
        description = str(soup.find("div",class_ = "sub-header").find_next_sibling("p"))

        a = data1[count]
        count = count + 1
        data.append((a, ratings,prices, name , description[3:-4], categories))            


    

    ############
           
    return data


# In[34]:


if __name__ == "__main__":  
    
    # Test Q1
    data=getData()
    print(data)
    
    # Test Q2
    preprocess_data(data)
    
    # Test Q3
    data=getFullData()
    print(len(data))
    
    # randomly select one book
    print(data[899])


# In[ ]:




