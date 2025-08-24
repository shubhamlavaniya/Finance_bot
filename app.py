# This file is part of the Financial RAG Chatbot project.
# It implements the main application logic for querying financial documents using Streamlit.

# Financial RAG + Fine-tuned Chatbot with Guardrails and Mode Switch


import streamlit as st
import time
from src.rag_core import get_rag_response, validate_query_simple
from src.ft_core import get_ft_response, load_ft_model_and_tokenizer, validate_query_simple
from src.db_handler import init_db, save_chat, load_chats, update_chat_title, check_and_reset_db
import uuid

# Initialize DB (creates the file on first run, reuses it on subsequent runs)
init_db()

# Add unique user ID for each user session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Initialize session state for the current conversation thread
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())
    st.session_state.thread_title = ""

# --- Sidebar ---
with st.sidebar:
    st.header("Start a New Chat")
    if st.button("New Chat", icon="✨"):
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.session_state.thread_title = ""
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.title("Settings")

    if "mode" not in st.session_state:
        st.session_state.mode = "RAG"

    st.radio(
        "Select Mode:",
        options=["RAG", "Fine-tuned"],
        index=0 if st.session_state.mode == "RAG" else 1,
        key="mode"
    )

    st.title("Recent")
    # Pass the user_id to load chats
    db_conversations = load_chats(user_id=st.session_state.user_id, limit=20)

    for conv in db_conversations:
        chat_title = conv["title"]

        col1, col2 = st.columns([0.7, 0.2])

        with col1:
            if st.button(f"**{chat_title}**", use_container_width=True, key=f"chat_button_{conv['thread_id']}"):
                st.session_state.messages = [
                    {"query": msg["query"], "answer": msg["answer"]}
                    for msg in conv["messages"]
                ]
                st.session_state.current_thread_id = conv["thread_id"]
                st.session_state.thread_title = conv["title"]
                st.rerun()

        with col2:
            with st.popover("⚙️", use_container_width=True):
                st.markdown("### Rename Chat")
                new_title = st.text_input(
                    "Enter a new name:",
                    value=chat_title,
                    key=f"rename_input_{conv['thread_id']}"
                )
                if st.button("Save Name", use_container_width=True, key=f"rename_button_{conv['thread_id']}"):
                    update_chat_title(st.session_state.user_id, conv["thread_id"], new_title)
                    st.success("Chat name updated!")
                    time.sleep(1)
                    st.rerun()

# --- Main Panel ---
st.title("The Accountant (RAG + Fine-tuned)")
st.write("Ask a financial question. Toggle between **RAG** and **Fine-tuned** modes from the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])

if prompt := st.chat_input("Enter your question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    validation_status = validate_query_simple(prompt)
    
    if validation_status in ["IRRELEVANT", "HARMFUL"]:
        response_text = f"Your query was flagged as **{validation_status}**. Please ask a relevant financial question."
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"query": prompt, "answer": response_text})
        st.stop()

    with st.spinner(f"Thinking with {st.session_state.mode}..."):
        start_time = time.time()

        if st.session_state.mode == "RAG":
            response_data = get_rag_response(prompt)
            
        elif st.session_state.mode == "Fine-tuned":
            with st.spinner("Loading financial expert..."):
                ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
            response_data = get_ft_response(prompt)

        response_time = round(time.time() - start_time, 2)
        answer = response_data["answer"]

    with st.chat_message("assistant"):
        if "VERIFIED" in str(response_data.get("verification", "")):
            st.success(f"**Answer (Verified):** {answer}")
        else:
            st.warning(f"**Answer (Warning: Potential Hallucination):** {answer}")
            st.info(f"Verification Status: {response_data.get('verification', 'N/A')}")

    st.markdown("---")
    st.markdown(f"**Method Used:** {response_data.get('method', 'N/A')}")
    st.markdown(f"**Response Time:** {response_time} seconds")

    if st.session_state.mode == "RAG":
        confidence_score = response_data.get("confidence", "N/A")
        if confidence_score != "N/A":
            st.markdown(f"**Retrieval Confidence:** {confidence_score:.2f}")
        else:
            st.markdown(f"**Retrieval Confidence:** {confidence_score}")

    st.session_state.messages.append(
        {"query": prompt, "answer": answer, "response_data": response_data}
    )

    chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

    save_chat(
        st.session_state.user_id,
        st.session_state.current_thread_id,
        chat_title,
        prompt,
        answer,
        st.session_state.mode,
        response_data,
        response_time
    )
    
    if not st.session_state.thread_title:
        st.session_state.thread_title = prompt

# import streamlit as st
# import time
# from src.rag_core import get_rag_response, validate_query_simple
# from src.ft_core import get_ft_response, load_ft_model_and_tokenizer, validate_query_simple
# from src.db_handler import init_db, save_chat, load_chats, update_chat_title
# import uuid

# # Initialize DB and run migration logic
# init_db()
# #migrate_schema()

# # # Add unique user ID for each user session
# # if "user_id" not in st.session_state:
# #     st.session_state.user_id = str(uuid.uuid4())

# st.set_page_config(page_title="Financial Chatbot", layout="wide")

# # Initialize session state for the current conversation thread
# if "current_thread_id" not in st.session_state:
#     st.session_state.current_thread_id = str(uuid.uuid4())
#     st.session_state.thread_title = ""

# # --- Sidebar ---
# with st.sidebar:
#     st.header("Start a New Chat")
#     if st.button("New Chat", icon="✨"):
#         st.session_state.current_thread_id = str(uuid.uuid4())
#         st.session_state.thread_title = ""
#         st.session_state.messages = []
#         st.rerun()

#     st.markdown("---")
#     st.title("Settings")

#     if "mode" not in st.session_state:
#         st.session_state.mode = "RAG"

#     st.radio(
#         "Select Mode:",
#         options=["RAG", "Fine-tuned"],
#         index=0 if st.session_state.mode == "RAG" else 1,
#         key="mode"
#     )

#     st.title("Recent")
#     # Pass the user_id to load chats
#     db_conversations = load_chats(limit=20)

#     for conv in db_conversations:
#         chat_title = conv["title"]

#         col1, col2 = st.columns([0.7, 0.2])

#         with col1:
#             if st.button(f"**{chat_title}**", use_container_width=True, key=f"chat_button_{conv['thread_id']}"):
#                 st.session_state.messages = [
#                     {"query": msg["query"], "answer": msg["answer"]}
#                     for msg in conv["messages"]
#                 ]
#                 st.session_state.current_thread_id = conv["thread_id"]
#                 st.session_state.thread_title = conv["title"]
#                 st.rerun()

#         with col2:
#             with st.popover("⚙️", use_container_width=True):
#                 st.markdown("### Rename Chat")
#                 new_title = st.text_input(
#                     "Enter a new name:",
#                     value=chat_title,
#                     key=f"rename_input_{conv['thread_id']}"
#                 )
#                 if st.button("Save Name", use_container_width=True, key=f"rename_button_{conv['thread_id']}"):
#                     # Pass the user_id to update the title
#                     update_chat_title(st.session_state.user_id, conv["thread_id"], new_title)
#                     st.success("Chat name updated!")
#                     time.sleep(1)
#                     st.rerun()

# # --- Main Panel ---
# st.title("The Accountant (RAG + Fine-tuned)")
# st.write("Ask a financial question. Toggle between **RAG** and **Fine-tuned** modes from the sidebar.")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message("user"):
#         st.markdown(message["query"])
#     with st.chat_message("assistant"):
#         st.markdown(message["answer"])

# if prompt := st.chat_input("Enter your question here..."):
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     validation_status = validate_query_simple(prompt)
    
#     if validation_status in ["IRRELEVANT", "HARMFUL"]:
#         response_text = f"Your query was flagged as **{validation_status}**. Please ask a relevant financial question."
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#         st.session_state.messages.append({"query": prompt, "answer": response_text})
#         st.stop()

#     with st.spinner(f"Thinking with {st.session_state.mode}..."):
#         start_time = time.time()

#         if st.session_state.mode == "RAG":
#             response_data = get_rag_response(prompt)
            
#         elif st.session_state.mode == "Fine-tuned":
#             with st.spinner("Loading financial expert..."):
#                 ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
#             response_data = get_ft_response(prompt)

#         response_time = round(time.time() - start_time, 2)
#         answer = response_data["answer"]

#     with st.chat_message("assistant"):
#         if "VERIFIED" in str(response_data.get("verification", "")):
#             st.success(f"**Answer (Verified):** {answer}")
#         else:
#             st.warning(f"**Answer (Warning: Potential Hallucination):** {answer}")
#             st.info(f"Verification Status: {response_data.get('verification', 'N/A')}")

#     st.markdown("---")
#     st.markdown(f"**Method Used:** {response_data.get('method', 'N/A')}")
#     st.markdown(f"**Response Time:** {response_time} seconds")

#     if st.session_state.mode == "RAG":
#         confidence_score = response_data.get("confidence", "N/A")
#         if confidence_score != "N/A":
#             st.markdown(f"**Retrieval Confidence:** {confidence_score:.2f}")
#         else:
#             st.markdown(f"**Retrieval Confidence:** {confidence_score}")

#     st.session_state.messages.append(
#         {"query": prompt, "answer": answer, "response_data": response_data}
#     )

#     chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

#     # Pass the user_id and thread_id to save the chat
#     save_chat(
#         st.session_state.user_id,
#         st.session_state.current_thread_id,
#         chat_title,
#         prompt,
#         answer,
#         st.session_state.mode,
#         response_data,
#         response_time
#     )
    
#     if not st.session_state.thread_title:
#         st.session_state.thread_title = prompt





# =================================Without Fine-Tuning, only for RAG mode========================================
# Uncomment the following lines if you want to run the RAG-only version without fine-tuning.
# This is useful for testing or if you want to focus solely on the RAG capabilities.
# import streamlit as st
# import time
# from src.rag_core import get_rag_response, validate_query, llm

# st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")

# # --- Initialize Session State ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # --- Left Sidebar for Chat History ---
# with st.sidebar:
#     st.title("Chat History")
#     if st.button("Clear History"):
#         st.session_state.messages = []
#         st.experimental_rerun()
    
#     for message in st.session_state.messages:
#         with st.expander(f"**You:** {message['query'][:50]}..."):
#             st.markdown(f"**Bot:** {message['answer']}")
#             st.markdown("---")
#             if "response_data" in message:
#                 response_data = message["response_data"]
#                 st.markdown(f"**Method Used:** {response_data['method']}")
#                 st.markdown(f"**Verification Status:** {response_data['verification']}")


# # --- Main Panel for Chat Interface ---
# st.title("Financial RAG Chatbot")
# st.write("Ask a question about financial filings.")

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message("user"):
#         st.markdown(message["query"])
#     with st.chat_message("assistant"):
#         st.markdown(message["answer"])

# # Accept user input
# if prompt := st.chat_input("Enter your question here..."):
#     # --- Display user message ---
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # --- Input-Side Guardrail ---
#     validation_status = validate_query(llm, prompt)
    
#     if validation_status in ["IRRELEVANT", "HARMFUL"]:
#         response_text = f"Your query was flagged as **{validation_status}**. Please ask a relevant financial question."
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#         st.session_state.messages.append({"query": prompt, "answer": response_text})
#         st.stop()
    
#     # --- Main RAG logic ---
#     with st.spinner("Thinking..."):
#         start_time = time.time()
        
#         response_data = get_rag_response(prompt)
        
#         end_time = time.time()
#         response_time = round(end_time - start_time, 2)
        
#         answer = response_data["answer"]
        
#         # --- Display assistant response ---
#         with st.chat_message("assistant"):
#             # Display based on verification status
#             if "VERIFIED" in response_data["verification"]:
#                 st.success(f"**Answer (Verified):** {answer}")
#             else:
#                 st.warning(f"**Answer (Warning: Potential Hallucination):** {answer}")
#                 st.info(f"Verification Status: {response_data['verification']}")
            
#             st.markdown("---")
            
#             # Display metadata
#             st.markdown(f"**Method Used:** {response_data['method']}")
#             st.markdown(f"**Response Time:** {response_time} seconds")
            
#             if response_data["method"] == "Hybrid RAG":
#                 st.markdown("**Source Documents:**")
#                 for i, doc in enumerate(response_data["source_docs"]):
#                     st.markdown(f"**Document {i+1}:**")
#                     st.markdown(f"**Section:** {doc.metadata.get('section', 'N/A')}")
#                     st.markdown(f"**Chunk ID:** {doc.metadata.get('chunk_id', 'N/A')}")
#                     st.markdown(f"**Content:** {doc.page_content[:200]}...")
#             else:
#                 st.markdown(f"**Source:** {response_data['source']}")
#                 st.markdown(f"**Confidence Score:** {response_data['confidence']}")

#     # Add messages to chat history
#     st.session_state.messages.append({"query": prompt, "answer": answer, "response_data": response_data})