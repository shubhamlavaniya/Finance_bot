# This file is part of the Financial RAG Chatbot project.
# It implements the main application logic for querying financial documents using Streamlit.

#openAI

import streamlit as st
import time
from src.rag_core import get_agentic_response
from src.ft_core import get_ft_response, load_ft_model_and_tokenizer
from src.db_handler import init_db, save_chat, load_chats, update_chat_title, load_latest_chat
from src.db_handler import migrate_schema
import uuid

# Initialize DB and migrations
init_db()
migrate_schema()

# Add unique user ID for each user session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

st.set_page_config(page_title="Financial Chatbot", layout="wide")

# Check if a chat thread exists. If not, try to load the latest one from the DB.
if "current_thread_id" not in st.session_state:
    latest_chat = load_latest_chat(user_id=st.session_state.user_id)
    if latest_chat:
        st.session_state.current_thread_id = latest_chat["thread_id"]
        st.session_state.thread_title = latest_chat["title"]
        # Load messages from the latest chat
        st.session_state.messages = [
            {"query": msg["query"], "answer": msg["answer"]}
            for msg in latest_chat["messages"]
        ]
    else:
        # If no previous chats exist, start a new, empty one
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.session_state.thread_title = ""
        st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Start a New Chat")
    if st.button("New Chat", icon="✨"):
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.session_state.thread_title = ""
        st.session_state.messages = []
        st.rerun()
    
    # Add clear chat button
    if st.button("Clear Current Chat", icon="🧹", use_container_width=True):
        st.session_state.messages = []
        st.sidebar.success("Current chat cleared!")
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
                    update_chat_title(
                        user_id=st.session_state.user_id,
                        thread_id=conv['thread_id'],
                        new_title=new_title
                    )
                    st.success("Chat name updated!")
                    time.sleep(1)
                    st.rerun()

# --- Main Panel ---
st.title("Lucius (Agentic RAG + Fine-tuned)")
st.write("Ask a financial(Apple filings 2022-2023) or AI related question. Toggle between **RAG** and **Fine-tuned** modes from the sidebar.")

# Display existing messages from the session state
for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])

if prompt := st.chat_input("Enter your question here..."):
    # Add user's message to the chat history and display it
    st.session_state.messages.append({"query": prompt, "answer": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Process Query Based on Selected Mode ---
    response_data = {}
    answer = ""
    response_time = 0
    with st.spinner(f"Thinking with {st.session_state.mode}..."):
        start_time = time.time()
        
        # Check the selected mode
        if st.session_state.mode == "RAG":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                answer = get_agentic_response(prompt)
                message_placeholder.markdown(answer)
                response_time = round(time.time() - start_time, 2)
                response_data = {
                    "method": "Agentic RAG", 
                    "verification": "Multi-step reasoning",
                    "confidence": "High"
                }
            
        elif st.session_state.mode == "Fine-tuned":
            with st.spinner("Loading financial expert..."):
                ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
            response_data = get_ft_response(prompt)
            answer = response_data["answer"]
            response_time = round(time.time() - start_time, 2)
            with st.chat_message("assistant"):
                st.markdown(answer)

    # Update the final message in the session state
    st.session_state.messages[-1]["answer"] = answer

    # Get the chat title (first question or saved title)
    chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

    # Save the chat with the new title
    save_chat(
        user_id=st.session_state.user_id,
        thread_id=st.session_state.current_thread_id,
        title=chat_title,
        query=prompt,
        answer=answer,
        mode=st.session_state.mode,
        response_data=response_data,
        response_time=response_time
    )

    # Update the session state title after saving the first message
    if not st.session_state.thread_title:
        st.session_state.thread_title = prompt
    
    st.rerun()








# Router based augumentation(not pure agentic RAG)

# import streamlit as st
# import time
# from src.rag_core import get_rag_response
# from src.ft_core import get_ft_response, load_ft_model_and_tokenizer
# from src.db_handler import init_db, save_chat, load_chats, update_chat_title, load_latest_chat
# from src.db_handler import migrate_schema
# import uuid

# # Initialize DB and migrations
# init_db()
# migrate_schema()

# # Add unique user ID for each user session
# if "user_id" not in st.session_state:
#     st.session_state.user_id = str(uuid.uuid4())

# st.set_page_config(page_title="Financial Chatbot", layout="wide")

# # Check if a chat thread exists. If not, try to load the latest one from the DB.
# if "current_thread_id" not in st.session_state:
#     latest_chat = load_latest_chat(user_id=st.session_state.user_id)
    
#     if latest_chat:
#         st.session_state.current_thread_id = latest_chat["thread_id"]
#         st.session_state.thread_title = latest_chat["title"]
#         # Load messages from the latest chat
#         st.session_state.messages = [
#             {"query": msg["query"], "answer": msg["answer"]}
#             for msg in latest_chat["messages"]
#         ]
#     else:
#         # If no previous chats exist, start a new, empty one
#         st.session_state.current_thread_id = str(uuid.uuid4())
#         st.session_state.thread_title = ""
#         st.session_state.messages = []

# # --- Sidebar ---
# with st.sidebar:
#     st.header("Start a New Chat")
#     if st.button("New Chat", icon="✨"):
#         st.session_state.current_thread_id = str(uuid.uuid4())
#         st.session_state.thread_title = ""
#         st.session_state.messages = []
#         st.rerun()
    
#     # Add clear chat button
#     if st.button("🧹 Clear Current Chat", icon="🧹", use_container_width=True):
#         st.session_state.messages = []
#         st.sidebar.success("Current chat cleared!")
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
    
#     db_conversations = load_chats(user_id=st.session_state.user_id, limit=20)

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
#                     update_chat_title(
#                         user_id=st.session_state.user_id,
#                         thread_id=conv['thread_id'],
#                         new_title=new_title
#                     )
#                     st.success("Chat name updated!")
#                     time.sleep(1)
#                     st.rerun()

# # --- Main Panel ---
# st.title("The Accountant (Agentic RAG + Fine-tuned)")
# st.write("Ask a financial(Apple filings 2022-2023) or AI related question. Toggle between **RAG** and **Fine-tuned** modes from the sidebar.")

# # Display existing messages from the session state
# for message in st.session_state.messages:
#     with st.chat_message("user"):
#         st.markdown(message["query"])
#     with st.chat_message("assistant"):
#         st.markdown(message["answer"])

# if prompt := st.chat_input("Enter your question here..."):
#     # Add user's message to the chat history and display it
#     st.session_state.messages.append({"query": prompt, "answer": ""})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # --- Process Query Based on Selected Mode ---
#     response_data = {}
#     answer = ""
#     response_time = 0
#     with st.spinner(f"Thinking with {st.session_state.mode}..."):
#         start_time = time.time()
        
#         # Check the selected mode
#         if st.session_state.mode == "RAG":
#             # The get_rag_response function now returns a generator that yields text chunks
#             # and then the final metadata dictionary.
#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 full_response = ""
#                 response_generator = get_rag_response(prompt)
                
#                 for chunk in response_generator:
#                     if isinstance(chunk, dict):
#                         response_data = chunk
#                     else:
#                         full_response += chunk
#                         message_placeholder.markdown(full_response + "▌")
                
#                 message_placeholder.markdown(full_response)
#                 answer = full_response
#                 response_time = round(time.time() - start_time, 2)
            
#         elif st.session_state.mode == "Fine-tuned":
#             with st.spinner("Loading financial expert..."):
#                 ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
#             response_data = get_ft_response(prompt)
#             answer = response_data["answer"]
#             response_time = round(time.time() - start_time, 2)
            
#             with st.chat_message("assistant"):
#                 st.markdown(answer)

#     # Update the final message in the session state
#     st.session_state.messages[-1]["answer"] = answer

#     # Get the chat title (first question or saved title)
#     chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

#     # Save the chat with the new title
#     save_chat(
#         user_id=st.session_state.user_id,
#         thread_id=st.session_state.current_thread_id,
#         title=chat_title,
#         query=prompt,
#         answer=answer,
#         mode=st.session_state.mode,
#         response_data=response_data,
#         response_time=response_time
#     )

#     # Update the session state title after saving the first message
#     if not st.session_state.thread_title:
#         st.session_state.thread_title = prompt
    
#     st.rerun()


# with warnings and verification----------------------------------------

# import streamlit as st
# import time
# from src.rag_core import get_rag_response, validate_query_simple
# from src.ft_core import get_ft_response, load_ft_model_and_tokenizer
# from src.db_handler import init_db, save_chat, load_chats, update_chat_title, load_latest_chat
# from src.db_handler import migrate_schema
# import uuid

# # =========================================================================
# # === UPDATED: STATE MANAGEMENT - INITIALIZE WITH LATEST CHAT FROM DB ===
# # =========================================================================

# # Initialize DB and migrations
# init_db()
# migrate_schema()

# # Add unique user ID for each user session
# if "user_id" not in st.session_state:
#     st.session_state.user_id = str(uuid.uuid4())

# st.set_page_config(page_title="Financial Chatbot", layout="wide")

# # Check if a chat thread exists. If not, try to load the latest one from the DB.
# if "current_thread_id" not in st.session_state:
#     latest_chat = load_latest_chat(user_id=st.session_state.user_id)
    
#     if latest_chat:
#         st.session_state.current_thread_id = latest_chat["thread_id"]
#         st.session_state.thread_title = latest_chat["title"]
#         # Load messages from the latest chat
#         st.session_state.messages = [
#             {"query": msg["query"], "answer": msg["answer"]}
#             for msg in latest_chat["messages"]
#         ]
#     else:
#         # If no previous chats exist, start a new, empty one
#         st.session_state.current_thread_id = str(uuid.uuid4())
#         st.session_state.thread_title = ""
#         st.session_state.messages = []

# # --- Sidebar ---
# with st.sidebar:
#     st.header("Start a New Chat")
#     if st.button("New Chat", icon="✨"):
#         st.session_state.current_thread_id = str(uuid.uuid4())
#         st.session_state.thread_title = ""
#         st.session_state.messages = []
#         st.rerun()
    
#     # Add clear chat button
#     if st.button("🧹 Clear Current Chat", icon="🧹", use_container_width=True):
#         st.session_state.messages = []
#         st.sidebar.success("Current chat cleared!")
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
    
#     # === UPDATED: Pass user_id parameter ===
#     db_conversations = load_chats(user_id=st.session_state.user_id, limit=20)
#     # =======================================

#     for conv in db_conversations:
#         chat_title = conv["title"]

#         # Use columns to put the chat button and popover side-by-side
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
#             # Use st.popover to create a compact, floating UI for renaming
#             with st.popover("⚙️", use_container_width=True):
#                 st.markdown("### Rename Chat")
#                 new_title = st.text_input(
#                     "Enter a new name:",
#                     value=chat_title,
#                     key=f"rename_input_{conv['thread_id']}"
#                 )
#                 if st.button("Save Name", use_container_width=True, key=f"rename_button_{conv['thread_id']}"):
#                     # === UPDATED: Pass user_id parameter ===
#                     update_chat_title(
#                         user_id=st.session_state.user_id,
#                         thread_id=conv['thread_id'],
#                         new_title=new_title
#                     )
#                     # ========================================
#                     st.success("Chat name updated!")
#                     time.sleep(1)
#                     st.rerun()

# # --- Main Panel ---
# st.title("The Accountant (RAG + Fine-tuned)")
# st.write("Ask a financial question. Toggle between **RAG** and **Fine-tuned** modes from the sidebar.")

# # Display existing messages from the session state
# for message in st.session_state.messages:
#     with st.chat_message("user"):
#         st.markdown(message["query"])
#     with st.chat_message("assistant"):
#         st.markdown(message["answer"])

# if prompt := st.chat_input("Enter your question here..."):
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # --- Input Validation (USING SIMPLE VALIDATION FOR BOTH MODES) ---
#     validation_status = validate_query_simple(prompt)
    
#     if validation_status in ["IRRELEVANT", "HARMFUL"]:
#         response_text = f"Your query was flagged as **{validation_status}**. Please ask a relevant financial question."
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#         st.session_state.messages.append({"query": prompt, "answer": response_text})
#         st.stop()

#     # --- Process Query Based on Selected Mode ---
#     response_data = {}
#     with st.spinner(f"Thinking with {st.session_state.mode}..."):
#         start_time = time.time()

#         if st.session_state.mode == "RAG":
#             try:
#                 # The generator yields text chunks and then the final response_data dict
#                 response_generator = get_rag_response(prompt)
                
#                 full_response = ""
#                 with st.chat_message("assistant"):
#                     with st.empty():
#                         for chunk in response_generator:
#                             if isinstance(chunk, dict):
#                                 response_data = chunk
#                             else:
#                                 full_response += chunk
#                                 st.markdown(full_response + "▌")
#                         st.markdown(full_response)
                
#                 answer = full_response
#                 response_time = round(time.time() - start_time, 2)
                
#             except Exception as e:
#                 error_message = f"RAG mode error: {str(e)}. Please try again."
#                 st.error(error_message)
#                 answer = error_message
#                 response_data = {"answer": answer, "method": "Error", "verification": "Error", "confidence": "N/A", "source": "N/A"}
#                 response_time = round(time.time() - start_time, 2)
#                 st.stop()
            
#         elif st.session_state.mode == "Fine-tuned":
#             with st.spinner("Loading financial expert..."):
#                 ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
#             response_data = get_ft_response(prompt)
#             answer = response_data["answer"]
#             response_time = round(time.time() - start_time, 2)

#     # --- Display Response ---
#     with st.chat_message("assistant"):
#         if "VERIFIED" in str(response_data.get("verification", "")):
#             st.success(f"**Answer (Verified):** {answer}")
#         else:
#             st.warning(f"**Answer (Warning: Potential Hallucination):** {answer}")
#             st.info(f"Verification Status: {response_data.get('verification', 'N/A')}")

#     st.markdown("---")
#     st.markdown(f"**Method Used:** {response_data.get('method', 'N/A')}")
#     st.markdown(f"**Response Time:** {response_time} seconds")

#     # Only display confidence if it exists for the RAG method
#     if st.session_state.mode == "RAG":
#         confidence_score = response_data.get("confidence", "N/A")
#         if confidence_score != "N/A":
#             st.markdown(f"**Retrieval Confidence:** {confidence_score:.2f}")
#         else:
#             st.markdown(f"**Retrieval Confidence:** {confidence_score}")

#     st.session_state.messages.append(
#         {"query": prompt, "answer": answer, "response_data": response_data}
#     )

#     # Get the chat title (first question or saved title)
#     chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

#     # Save the chat with the new title
#     save_chat(
#         user_id=st.session_state.user_id,
#         thread_id=st.session_state.current_thread_id,
#         title=chat_title,
#         query=prompt,
#         answer=answer,
#         mode=st.session_state.mode,
#         response_data=response_data,
#         response_time=response_time
#     )

#     # Update the session state title after saving the first message
#     if not st.session_state.thread_title:
#         st.session_state.thread_title = prompt

#---without streaming code------

# import streamlit as st
# import time
# from src.rag_core import get_rag_response, validate_query_simple
# from src.ft_core import get_ft_response, load_ft_model_and_tokenizer
# from src.db_handler import init_db, save_chat, load_chats, update_chat_title
# from src.db_handler import migrate_schema
# import uuid

# # Initialize DB and migrations
# init_db()
# migrate_schema()

# # Add unique user ID for each user session
# if "user_id" not in st.session_state:
#     st.session_state.user_id = str(uuid.uuid4())

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
    
#     # Add clear chat button
#     if st.button("🧹 Clear Current Chat", icon="🧹", use_container_width=True):
#         st.session_state.messages = []
#         st.sidebar.success("Current chat cleared!")
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
    
#     # === UPDATED: Pass user_id parameter ===
#     db_conversations = load_chats(user_id=st.session_state.user_id, limit=20)
#     # =======================================

#     for conv in db_conversations:
#         chat_title = conv["title"]

#         # Use columns to put the chat button and popover side-by-side
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
#             # Use st.popover to create a compact, floating UI for renaming
#             with st.popover("⚙️", use_container_width=True):
#                 st.markdown("### Rename Chat")
#                 new_title = st.text_input(
#                     "Enter a new name:",
#                     value=chat_title,
#                     key=f"rename_input_{conv['thread_id']}"
#                 )
#                 if st.button("Save Name", use_container_width=True, key=f"rename_button_{conv['thread_id']}"):
#                     # === UPDATED: Pass user_id parameter ===
#                     update_chat_title(
#                         user_id=st.session_state.user_id,
#                         thread_id=conv['thread_id'],
#                         new_title=new_title
#                     )
#                     # ========================================
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

#     # --- Input Validation (USING SIMPLE VALIDATION FOR BOTH MODES) ---
#     validation_status = validate_query_simple(prompt)
    
#     if validation_status in ["IRRELEVANT", "HARMFUL"]:
#         response_text = f"Your query was flagged as **{validation_status}**. Please ask a relevant financial question."
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#         st.session_state.messages.append({"query": prompt, "answer": response_text})
#         st.stop()

#     # --- Process Query Based on Selected Mode ---
#     with st.spinner(f"Thinking with {st.session_state.mode}..."):
#         start_time = time.time()

#         if st.session_state.mode == "RAG":
#             try:
#                 response_data = get_rag_response(prompt)
#                 answer = response_data["answer"]
#                 response_time = round(time.time() - start_time, 2)
                
#             except Exception as e:
#                 # Display the error but don't try to change state or call another model.
#                 error_message = f"RAG mode error: {str(e)}. Please try again or switch to Fine-tuned mode manually."
#                 st.error(error_message)
#             # Set a clear answer for display and saving
#                 answer = error_message
#                 response_data = {"answer": answer, "method": "Error"}
#                 response_time = round(time.time() - start_time, 2)
#             # Stop the script to prevent any further issues
#                 st.stop()
            
#         elif st.session_state.mode == "Fine-tuned":
#             # Load fine-tuned model ONLY when in fine-tuned mode
#             with st.spinner("Loading financial expert..."):
#                 ft_model, ft_tokenizer, ft_device = load_ft_model_and_tokenizer()
#             response_data = get_ft_response(prompt)
#             answer = response_data["answer"]
#             response_time = round(time.time() - start_time, 2)

#     # --- Display Response ---
#     with st.chat_message("assistant"):
#         if "VERIFIED" in str(response_data.get("verification", "")):
#             st.success(f"**Answer (Verified):** {answer}")
#         else:
#             st.warning(f"**Answer (Warning: Potential Hallucination):** {answer}")
#             st.info(f"Verification Status: {response_data.get('verification', 'N/A')}")

#     st.markdown("---")
#     st.markdown(f"**Method Used:** {response_data.get('method', 'N/A')}")
#     st.markdown(f"**Response Time:** {response_time} seconds")

#     # Only display confidence if it exists for the RAG method
#     if st.session_state.mode == "RAG":
#         confidence_score = response_data.get("confidence", "N/A")
#         if confidence_score != "N/A":
#             st.markdown(f"**Retrieval Confidence:** {confidence_score:.2f}")
#         else:
#             st.markdown(f"**Retrieval Confidence:** {confidence_score}")

#     st.session_state.messages.append(
#         {"query": prompt, "answer": answer, "response_data": response_data}
#     )

#     # Get the chat title (first question or saved title)
#     chat_title = st.session_state.thread_title if st.session_state.thread_title else prompt

#     # Save the chat with the new title
#     # === UPDATED: Pass user_id as first parameter ===
#     save_chat(
#         user_id=st.session_state.user_id,
#         thread_id=st.session_state.current_thread_id,
#         title=chat_title,
#         query=prompt,
#         answer=answer,
#         mode=st.session_state.mode,
#         response_data=response_data,
#         response_time=response_time
#     )
#     # ================================================

#     # Update the session state title after saving the first message
#     if not st.session_state.thread_title:
#         st.session_state.thread_title = prompt

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