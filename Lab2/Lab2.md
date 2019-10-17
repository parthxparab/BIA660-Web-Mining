Q1. Scrape Book Catalog

Scape content of http://books.toscrape.com (http://books.toscrape.com)
Write a function getData() to scrape title, rating, price of all books (i.e. 20 books) listed in the page.
For example, the figure shows one book and the corresponding html code. You need to scrape the highlighted content.
For star ratings, you can simply scrape One, Two, Three, ...
The output is a list of 20 tuples, e.g. [('A Light in the ...','Three','£51.77'), ...]

Q2. Data Analysis

Create a function preprocess_data which
takes the list of tuples from Q1 as an input
converts the price strings to numbers
calculates the average price of books by ratings
plots a bar chart to show the average price by ratings.

Q3 (Bonus) Expand your solution to Q1 to scrape the full details of all books on http://books.toscrape.com (http://books.toscrape.com)
Write a function getFullData() to do the following:
Besides scraping title, rating, and price of each book as stated in Q1, also scrape the full title, description, and category in each individual book page.
Scape all book listing pages following the "next" link at the bottom. The figure below gives an screenshot of the "next" link and its corresponding html code.
Do not hardcode page URLs (except http://books.toscrape.com (http://books.toscrape.com)) in your code.
The output is a list containing 1000 tuples,
e.g. [('A Light in the ...','Three','£51.77', 'A Light in the Attic', "It's hard to imagine a world without A Light in the Attic. This now-classic collection ...",'Poetry'), ...]
