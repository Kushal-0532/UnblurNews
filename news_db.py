# This file deals with putting articles in postgres db running in docker. IDK MAN I'M BORED

import psycopg2
from psycopg2.extras import execute_values


# creating a function that saves article data to database.

def save_article(title, url, source, publishedAt, content):
    """
    Saves the article data (title, url, source, publishedAt, content) to the PostgreSQL database.
    Creates the table if it doesn't exist.
    """
    conn = psycopg2.connect(
        host="localhost",
        port="5353",
        database="unblurDB",
        user="Admin",
        password="password123"
    )
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT,
            url TEXT UNIQUE,
            source TEXT,
            publishedAt TIMESTAMP,
            content TEXT
        )
    """)
    
    cur.execute(
        "INSERT INTO articles (title, url, source, publishedAt, content) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (url) DO NOTHING",
        (title, url, source, publishedAt, content)
    )
    
    conn.commit()
    cur.close()
    conn.close()
