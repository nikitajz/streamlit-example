import logging
import os
from typing import Dict, List, Union

import streamlit as st

import cohere
from qdrant_client.http import models
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_HOST"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

RESULT_LIMIT = 5
INDEX_NAME = "podcasts"
COHERE_SIZE_VECTOR = 1024
COHERE_MODEL = "small" # should match embedding size
QDRANT_DISTANCE = models.Distance.COSINE


st.set_page_config(layout="wide")


class SearchClient:
    def __init__(
        self,
        collection_name: str,
        qdrabt_api_key: str = QDRANT_API_KEY,
        qdrant_url: str = QDRANT_URL,
        cohere_api_key: str = COHERE_API_KEY,
        create_index: bool = False,
        embedding_dimension: int=COHERE_SIZE_VECTOR, 
        distance: models.Distance=QDRANT_DISTANCE
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrabt_api_key)
        self.collection_name = collection_name
        
        self.co_client = cohere.Client(api_key=cohere_api_key)
        if create_index:
            self._create_index(embedding_dimension, distance)

    # Qdrant requires data in float format
    def _float_vector(self, vector: List[float]):
        return list(map(float, vector))

    # Embedding using Cohere Embed model
    def _embed(self, texts: Union[str, List[str]], retry=0, timeout=10):
        try:
            if isinstance(texts, str):
                texts = [texts]
            emb = self.co_client.embed(texts=texts, model=COHERE_MODEL).embeddings[0]
            return emb
        except:
            logger.warning("Throttling Cohere API")
            time.sleep(timeout)
            return self._embed(texts, timeout=timeout+10)

    # Search using text query
    def search(self, query_text: str, limit: int = 3):
        query_vector = self._embed(query_text)

        return self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self._float_vector(query_vector),
            limit=limit,
        )

search_client = SearchClient(collection_name=INDEX_NAME)

    
def main():
    st.header("Find podcast episode")
    st.markdown(
        """
    This demo app allows to find relevant podcast episode using semantic search. Type search query your're interested.
    """
    )

    with st.form("Search podcasts"):

        query = st.text_area(
            # Instructions
            "Enter search phrase",
            # 'sample' variable that contains our keyphrases.
            placeholder="Try, for example: Career advice in Machine Learning",
            height=100,
            help="Try to describe your search at least in 4-6 words to make it more accurate",
            key="1",
        )

        submitted = st.form_submit_button(label="Search")

        if submitted:

            results = search_client.search(query, limit=RESULT_LIMIT)

            st.success("✅ Done!")

            st.markdown(
                f"""<style>
                .result {
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #f8f8f8;
                    border-radius: 10px;
                    box-shadow: 0 2px 2px rgba(0, 0, 0, 0.1);
                }
                .title {
                    font-size: 1.2rem;
                    font-weight: bold;
                    margin: 0 0 10px 0;
                }

                .pub_date {
                    font-size: 0.9rem;
                    color: #666;
                    margin: 0;
                }

                .score {
                    font-size: 0.9rem;
                    color: #666;
                    margin: 0;
                }
                .podcast {
                    font-size: 0.9rem;
                    font-weight: bold;
                    margin: 0 0 10px 0;
                }
                .text {
                    font-size: 1rem;
                    margin: 10px 0 0 0;
                    line-height: 1.5;
                }

                </style>""",
                unsafe_allow_html=True,
            )

            search_results_md = " ".join(
                    [
                        f"""
                            <div class="result">
                                <h5 class="title">{episode.payload['title']}</h5>
                                <p class="score">Relevance: {episode.score:.3f}</p>
                                <p class="podcast">{episode.payload['podcast']}</p>
                                <p class="pub_date">{episode.payload["pub_date"]}</p>
                                <p class="url"><a href="{episode.payload['url']}">{episode.payload['url']}</a></p>
                                <p class="text">{episode.payload['text']}</p>
                            </div>
                        """
                        for episode in results
                    ]
                )

            st.markdown(
                f'{search_results_md}',
                unsafe_allow_html=True,
            )

            with st.expander("See raw response"):
                st.write(results)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: " "%(message)s",
        force=True,
    )

    logger.setLevel(logging.INFO)

    main()
