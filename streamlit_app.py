import logging
import os

import streamlit as st
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()

QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_HOST"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

RESULT_LIMIT = 5
INDEX_NAME = "podcasts"

search_client = SearchClient(collection_name=INDEX_NAME)

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
        distance: Literal[models.Distance]=QDRANT_DISTANCE
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrabt_api_key)
        self.collection_name = collection_name
        
        self.co_client = cohere.Client(api_key=cohere_api_key)
        if create_index:
            self._create_index(embedding_dimension, distance)

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

            with open("src/search_results.css") as fp:
                search_style = fp.read()

            st.markdown(
                f"<style>{search_style}</style>",
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
