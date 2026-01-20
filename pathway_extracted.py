import pathway as pw
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm.splitters import RecursiveSplitter
from pathway.io.python import ConnectorSubject
from pathway.stdlib.ml.index import KNNIndex  # Note: ml.index, NOT indexing
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.llms import OpenAIChat

# from pathway.stdlib.ml.text_splitters import CharacterTextSplitter
import time
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "country": "us", 
        "pageSize": 20
    }
    return requests.get(url, params=params).json()
# print(fetch_news())

class NewsSchema(pw.Schema):
    article_id: str = pw.column_definition(primary_key=True)
    title: str
    content: str
    source: str
    published_at: str
    url: str


class NewsApiConnector(ConnectorSubject):
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.url = "https://newsapi.org/v2/top-headlines"

    def run(self):
        params = {'q': 'technology', 'apiKey': self.api_key, "language": "en", "country": "us", "pageSize": 20}
        while True:
            res = requests.get(self.url, params=params).json()
            if res.get("status") != "ok":
                print(f"News api error: {res}")
                time.sleep(60)
                continue

            articles = res.get('articles', [])
            print(f"Fetched {len(articles)} articles")
            for article in articles:
                url = article.get("url")
                if not url:
                    continue

                #Mapping api field to my schima field
                self.next(
                    article_id=article.get('url'), # Using URL as a unique ID
                    title=article.get('title') or "",
                    content=article.get('content') or article.get('description') or  "",
                    source=article.get('source', {}).get('name'),
                    published_at=article.get('publishedAt', ""),
                    url=article.get('url')
                )
            time.sleep(900)


news_connector = NewsApiConnector(api_key=NEWS_API_KEY)


news_table = pw.io.python.read(news_connector, schema=NewsSchema)


documents = news_table.select(
    doc_id=pw.this.article_id,
    text=pw.this.title + "\n\n" + pw.this.content,
    published_at=pw.this.published_at,
    source=pw.this.source,
    url=pw.this.url,
)


import pathway as pw
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

@pw.udf
def token_count_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
):
    if not text:
        return []

    tokens = enc.encode(text)
    chunks = []

    start = 0
    n = len(tokens)

    while start < n:
        end = start + chunk_size
        chunks.append(enc.decode(tokens[start:end]))
        start = max(end - chunk_overlap, 0)

    return chunks


chunks = documents.select(
    doc_id=pw.this.doc_id,
    chunks=token_count_split(
        pw.this.text,
        400,
        50,
    ),
    published_at=pw.this.published_at,
    source=pw.this.source,
    url=pw.this.url,
)

chunks = chunks.flatten(pw.this.chunks).select(
    doc_id=pw.this.doc_id,
    chunk=pw.this.chunks,
    published_at=pw.this.published_at,
    source=pw.this.source,
    url=pw.this.url,
)



import hashlib

def chunk_id(article_id, chunk_text):
    return hashlib.sha1(
        f"{article_id}:{chunk_text}".encode("utf-8")
    ).hexdigest()


import pathway as pw
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

@pw.udf
def embed_text(text: str):
    if not text:
        return []
    return model.encode(text).tolist()


import hashlib

@pw.udf
def chunk_id(doc_id: str, chunk: str) -> str:
    h = hashlib.sha1(f"{doc_id}:{chunk}".encode()).hexdigest()
    return h


embedded_chunks = chunks.select(
    chunk_id=pw.apply(chunk_id, pw.this.doc_id, pw.this.chunk),
    vector=embed_text(pw.this.chunk),
    text=pw.this.chunk,
    published_at=pw.this.published_at,
    source=pw.this.source,
    url=pw.this.url,
)






# embedder = OpenAIEmbedder()

# embedded_chunks = chunks.select(
#     chunk_id = pw.apply(chunk_id, pw.this.doc_id, pw.this.chunk),
#     vector=embedder(pw.this.chunk),
#     text=pw.this.chunk,
#     published_at=pw.this.published_at,
#     source=pw.this.source,
#     url=pw.this.url,
# )


index = KNNIndex(
    embedded_chunks.vector,
    embedded_chunks,
    n_dimensions=384 
)


class QuerySchema(pw.Schema):
    query: str

webserver = pw.io.http.PathwayWebserver(host="0.0.0.0", port=8090)
queries, response_writer = pw.io.http.rest_connector(
    webserver=webserver,
    route="/query",  # or your desired route
    schema=QuerySchema
)


#Embed query
query_vectors = queries.select(
    query_id=pw.this.id,
    query=pw.this.query,
    vector=embed_text(pw.this.query),
)

nearest = index.get_nearest_items(
    query_vectors.vector,
    k=5
)

retrieved = nearest.join(
    embedded_chunks,
    pw.left.id == pw.right.id
).select(
    text=pw.right.text,
    source=pw.right.source,
    url=pw.right.url,
    published_at=pw.right.published_at,
)

context_table = retrieved.reduce(
    texts=pw.reducers.tuple(pw.this.text),
    emit_on="append"
)

context = context_table.select(
    query_id=pw.this.id,
    context=pw.apply(
        lambda texts: "\n\n".join(texts[:3]),
        pw.this.texts
    )
)



prompt = context.join(
    queries,
    pw.left.query_id == pw.right.id,  # Use id
).select(
    prompt=pw.apply(
        lambda c, q: f"""
You are a helpful AI assistant.
Use ONLY the context below to answer the question.

Context:
{c}

Question:
{q}

Answer:
""".strip(),
        pw.left.context,
        pw.right.query
    )
)




# llm = OpenAIChat(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], temperature=0)

# answers = prompt.select(
#     answer=llm(pw.this.prompt)
# )

llm = OpenAIChat(
    model="llama3-70b-8192",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)
answers = prompt.select(
    answer=llm(pw.this.prompt)
)

response_writer(
    answers.select(
        answer=pw.this.answer
    )
)

pw.run()