<br>

<div align="center">

## Financial Chatbot: RAG vs. Fine-Tuning

This project is a Streamlit-based financial chatbot designed to compare two different approaches to building an intelligent assistant: Retrieval-Augmented Generation (RAG) and Fine-Tuning. The chatbot answers questions based on a specific knowledge base of financial documents.

</div>

<br>


### Features

1. Two Operational Modes: The app can be toggled between RAG and a fine-tuned LLM.

2. RAG Pipeline: Utilizes a phi3:mini/gemma:2b model via Ollama to answer questions by retrieving and synthesizing information from a financial document corpus.

3. Fine-Tuning: Uses a fine-tuned distilgpt2 model for fast, specialized responses.

4. Chat History: Conversations are saved to a local SQLite database for persistence.

5. Performance Evaluation: Includes a dedicated script (evaluation.py) to benchmark the two models on inference speed and accuracy.

### Getting Started

    Follow these steps to set up and run the project locally.

    Prerequisites
    Python 3.10+
    Git

### Installation

1. Clone the Repository

        git clone https://github.com/shubhamlavaniya/Financial_ChatBot.git

        cd your-repo-name

2. Create a Virtual Environment

        python -m venv venv

        On Windows: venv\Scripts\activate

        On macOS/Linux: source venv/bin/activate

3. Install Dependencies

        pip install -r requirements.txt

4. Set up Ollama & Models

    . The RAG mode requires Ollama to be running locally.

    . Download and install Ollama from ollama.com.

    . Once installed, open a terminal and pull the required model:

        ollama pull gemma:2b/phi3:mini

5. Run the Data Pre Processing step

    . This script will create the necessary document chunks for the RAG pipeline.

        python src/preprocess.py ---> src/segmenter.py ---> src/chunker.py

6. Running the App

    . Run the Streamlit application from the project's root directory:

        streamlit run app.py

    . The app will open in your browser, and you can start a conversation in either RAG or         Fine-Tuning mode. The chat history will be saved to chat_history.db in the project root.

7. Running the Evaluation

    . To run the evaluation script and compare model performance, use the following command:

        python src/evaluation.py

    . This will generate an eval_results.csv file with a detailed comparison of the models.

### Technologies

1. Streamlit: For the web-based user interface.

2. LangChain: For building the RAG and fine-tuning pipelines.

3. Sentence-Transformers: For generating document embeddings.

4. Ollama: For running gemma:2b locally.

5. PyTorch & Transformers: For the fine-tuned distilgpt2 model.

6. SQLite3: For storing and managing chat history.

7. Pandas: For data handling and evaluation.