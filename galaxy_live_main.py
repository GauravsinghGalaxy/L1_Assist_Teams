import os
import aiohttp
import random
from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
from botbuilder.schema._connector_client_enums import ActivityTypes
import pytz
from torch import device
import torch
from twilio.rest import Client
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer)
import datetime as dt 
import pyodbc
import openai
import json
import os
from typing import List, Optional, Dict
from guardrails import Guard
# from guardrails_hub.toxic_language import ToxicLanguage 
from sentence_transformers import SentenceTransformer
# from guardrails_ai import Guardrails
# from guardrails_ai.hub import ToxicLanguage, ProfanityFree
# from guardrails_hub.toxic_language import ToxicLanguage
from transformers import pipeline
from sentence_transformers import SentenceTransformer
classifier = pipeline("text-classification", model="unitary/toxic-bert")

from embeddings import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import PyPDF2
import rag_samba_continuous_function as rag
import pickle
from PIL import Image
from docx import Document
import uuid
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from datetime import datetime, timedelta
import fitz

link_url = "https://api.goapl.com"

# Azure AD App details
APP_ID = "f568f633-2a3f-43e9-99b9-6aad90474292"
APP_PASSWORD = "6HX8Q~mOa6Inw6NLRlaZTnI3DtyZ-1irJMLV7b8k"
TENANT_ID = "b76c67d2-fe4d-4a91-bab0-afbe00a44837"

VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None
embeddings = get_embeddings()

# FAISS configuration for optimal performance
FAISS_CONFIG = {
    "metric": "cosine"  # Distance metric for similarity search
}

# Create a directory for storing uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create a directory for storing temporary chunks
CHUNKS_DIR = "temp_chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Dictionary to track upload progress
upload_progress = {}
processing_store: Dict[str, Dict] = {}

# Chat history storage
chat_sessions: Dict[str, dict] = {}

model = SentenceTransformer('all-mpnet-base-v2') # Best model for general-purpose semantic matching

# Encode tags and user query using your NLP model
tags = ["tech_support", "product_support", "order_support", "payment_support", "account_support", "other"]
tag_embeddings = model.encode(tags)

SERVER="outsystems1.database.windows.net"
DATABASE="OUTSYSTEM_API"
UID="Galaxy"
PWD='OutSystems@123'
OPENAI_API_KEY="100cfa62-287e-4983-8986-010da6320a53"
TWILIO_ACCOUNT_SID="AC0593c6bc6f51e8c8bf203382b5ba2c58"
TWILIO_AUTH_TOKEN="f8431873a203913829c2766793af3bd4"
TWILIO_SERVICE_SID="VAa4f062c9ccf314355babbad3997c77ff"

client = openai.OpenAI(
    api_key= OPENAI_API_KEY,
    base_url="https://api.sambanova.ai/v1",
)


# guard = Guard().use(
#     ToxicLanguage(threshold=0.8)  # Old initialization style
# )
try:
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={UID};"
        f"PWD={PWD}"
    )
    cursor = conn.cursor()
    
    print("Database connection successful")
except pyodbc.Error as db_err:
    print(f"Database connection error: {db_err}")

adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

recent_activity_ids = set()

def check_text_content(text):
    try:
        validation_result = Guard.validate(text)
        return {
            'is_valid': validation_result.validation_passed,
            'message': "Your message contains content that violates our community guidelines. Please ensure your message is respectful and appropriate before trying again."
        }
    except Exception as e:
        return {
            'is_valid': False,
            'message': "We encountered an issue processing your message. Please try again with different wording."
        }

def extract_section(pdf, start_page, end_page, title):
    section_text = ""
    for page_num in range(
        start_page - 1, end_page
    ):  # Page numbers are 0-indexed in PyMuPDF
        page = pdf.load_page(page_num)
        section_text += page.get_text()

    return section_text

def store_messages(uuid_id, session_id, message, remote_phone_number, sent_by):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={{outsystems1.database.windows.net}};"
        f"DATABASE={{OUTSYSTEM_API}};"
        f"UID={{Galaxy}};"
        f"PWD={{OutSystems@123}}"
    )
    cursor = conn.cursor()

    id = uuid_id
    uuid = uuid_id
    session_key = session_id
    message_text = message
    media_url = "NULL"
    media_type = "NULL"
    media_mime_type = "NULL"
    remote_phone_number = remote_phone_number
    _2chat_link = "NULL"
    channel_phone_number = "+919322261280"
    sent_by = sent_by
            
    # Check if id already exists in the database
    cursor.execute("SELECT COUNT(*) FROM WhatsAppMsgs WHERE id = ?", (id,))
    result = cursor.fetchone()

    ist_timezone = pytz.timezone("Asia/Kolkata")
    current_datetime = dt.datetime.now(ist_timezone)

    if result[0] == 0:
        # If id does not exist, insert the new record
        cursor.execute(
            """
            INSERT INTO WhatsAppMsgs 
            (id, uuid, session_key, message_text, media_url, media_type, media_mime_type, created_at, remote_phone_number, _2chat_link, channel_phone_number, sent_by, Issue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                id,
                uuid,
                session_key,
                message_text,
                media_url,
                media_type,
                media_mime_type,
                current_datetime,
                remote_phone_number,
                _2chat_link,
                channel_phone_number,
                sent_by,
                "NULL"
            ),
        )
        conn.commit()

def get_all_data(from_number: str):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        
        return data.get(from_number, None)
            
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_stage(from_number: str):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        
        return data.get(from_number, {})
            
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def set_stage(stage: str, phone_number: str, com_name: str = '0', mo_name: str = '0', user_name: str = '0', pdf_file: str = '0', vector_file: str = '0', conversation_history: list = [], chunks_file: str = '0', last_uuid: list = [], solution_type: str = '0', rag_no: int = 0, last_time: str = '0', session_key: str = '0', use_which_gpu: str = '0'):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    if phone_number not in data:
        data[phone_number] = {}

    # Update only if the new value is not the default '0'
    if stage != '0':
        data[phone_number]["stage"] = stage
    if com_name != '0':
        data[phone_number]["com_name"] = com_name
    if mo_name != '0':
        data[phone_number]["mo_name"] = mo_name
    if user_name != '0':
        data[phone_number]["user_name"] = user_name
    if pdf_file != '0':
        data[phone_number]["pdf_file"] = pdf_file
    if vector_file != '0':
        data[phone_number]["vector_file"] = vector_file
    if conversation_history:
        data[phone_number]["conversation_history"] = conversation_history
    if chunks_file != '0':
        data[phone_number]["chunks_file"] = chunks_file
    if last_uuid:
        data[phone_number]["last_uuid"] = last_uuid
    if solution_type != '0':
        data[phone_number]["solution_type"] = solution_type
    if rag_no != 0:
        data[phone_number]["rag_no"] = rag_no
    if last_time != 0:
        data[phone_number]["last_time"] = last_time
    if session_key != '0':
        data[phone_number]["session_key"] = session_key
    if use_which_gpu != '0':
        data[phone_number]["use_which_gpu"] = use_which_gpu
    with open("user_data.json", "w") as file:
        json.dump(data, file, indent=4)

    return "Stage set successfully"

def clear_stage(phone_number: str):
    # Clear user_data.json
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        # Remove the entry for this phone number if it exists
        if phone_number in data:
            del data[phone_number]
        with open("user_data.json", "w") as file:
            json.dump(data, file, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Clear user_interactions.json
    try:
        with open("user_interactions.json", "r") as file:
            interactions = json.load(file)
        # Remove the entry for this phone number
        interactions = [i for i in interactions if i['phone_number'] != phone_number]
        with open("user_interactions.json", "w") as file:
            json.dump(interactions, file, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return "Stage cleared successfully"

def get_best_matching_tag(user_query):
    # Fetch all distinct tags and strip extra whitespace
    cursor.execute("""
        SELECT DISTINCT LTRIM(RTRIM(value)) AS tag 
        FROM decision_tree 
        CROSS APPLY STRING_SPLIT(tags_list, ',')
    """)
    rows = cursor.fetchall()
    tags = [row[0] for row in rows]
    
    if not tags:
        return None, None, None, None
    
    # Encode tags and user query using your NLP model
    tag_embeddings = model.encode(tags)
    query_embedding = model.encode([user_query])
    
    # Compute cosine similarity between the query and each tag
    similarities = cosine_similarity(query_embedding, tag_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx] * 100

    if best_match_score < 80:
        return None, None, None, None

    best_tag = tags[best_match_idx]
    
    # Retrieve dt_id for the row that contains the best matching tag
    cursor.execute("""
        SELECT dt_id 
        FROM decision_tree 
        WHERE EXISTS (
            SELECT 1 
            FROM STRING_SPLIT(tags_list, ',')
            WHERE LTRIM(RTRIM(value)) = ?
        )
    """, (best_tag,))
    result = cursor.fetchone()
    if not result:
        return None, None, None, None
    dt_id = result[0]
    
    # Retrieve the question text for the dt_id
    cursor.execute("""
        SELECT question_text 
        FROM decision_tree 
        WHERE type_id = 'Issue' AND dt_id = ?
    """, (dt_id,))
    dt_data = cursor.fetchone()
    if not dt_data or not dt_data[0]:
        return None, None, None, None
    question_text = dt_data[0]

    # Retrieve the action_id for the dt_id
    cursor.execute("""
        SELECT action_id 
        FROM decision_tree 
        WHERE type_id = 'Issue' AND dt_id = ?
    """, (dt_id,))
    action_data = cursor.fetchone()
    if not action_data or not action_data[0]:
        return None, None, None, None
    action = action_data[0]
    
    return best_tag, dt_id, question_text, action

def store_user_interaction(phone_number: str, stage: str = '0', solution_number: int = 0, result: dict = None, issue: str = None, dt_id: int = None, action: str = None, yes_id: str = None, user_name: str = None):
    # Convert result to serializable format if it's a Row object
    if result and hasattr(result, '_mapping'):
        result = dict(result._mapping)
    elif result and isinstance(result, pyodbc.Row):
        result = {key: value for key, value in zip([column[0] for column in cursor.description], result)}
    
    interaction = {
        "phone_number": phone_number,
        "stage": stage,
        "issue": str(issue) if issue else None,  # Convert to string in case it's a Row
        "dt_id": int(dt_id) if dt_id else None,  # Convert to int in case it's a Row
        "solution_number": solution_number,
        "timestamp": str(dt.datetime.now()),
        "user_name": user_name,
        "result": result,
        "action": str(action) if action else None,  # Convert to string in case it's a Row
        "yes_id": str(yes_id) if yes_id else None,  # Convert to string in case it's a Row
    }
    
    try:
        with open('user_interactions.json', 'r') as file:
            interactions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        interactions = []
    
    # Update or append interaction
    updated = False
    for i, existing in enumerate(interactions):
        if existing['phone_number'] == phone_number:
            interactions[i] = interaction
            updated = True
            break
    
    if not updated:
        interactions.append(interaction)
    
    # Save updated interactions
    with open('user_interactions.json', 'w') as file:
        json.dump(interactions, file, indent=4)

def get_user_interaction(phone_number: str) -> dict:
    """Get stored interaction details for a user"""
    try:
        with open('user_interactions.json', 'r') as file:
            interactions = json.load(file)
            for interaction in interactions:
                if interaction['phone_number'] == phone_number:
                    return interaction
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def encodings_process(pdf_file: str, phone_number: str, com_name: str, mo_name: str, username: str):
    all_chunks = []
    pdf_file = os.path.join("PDFs", pdf_file)
    current_chunks = rag.get_chunks(pdf_file)
    all_chunks.extend(current_chunks)
    chunks = all_chunks

    request = requests.get("http://43.252.194.228:6001/encode", json={"context": chunks})
    context_encodings = request.json()["encodings"]

    # Save chunks to a file
    chunks_filename = f"encodings/chunks_{phone_number}.pkl"
    with open(chunks_filename, 'wb') as f:
        pickle.dump(chunks, f)

    # Save encodings to a file
    encodings_filename = f"encodings/encodings_{phone_number}.npy"
    np.save(encodings_filename, context_encodings)

    # Update vector_file in database
    cursor.execute("""
        UPDATE l1_tree
        SET vector_file = ?, chunks_file = ?
        WHERE phone_number = ?
    """, (encodings_filename, chunks_filename, phone_number))
    conn.commit()
    vector_file = encodings_filename
    set_stage("tech_support", phone_number, com_name, mo_name, username, pdf_file=pdf_file, vector_file=vector_file, chunks_file=chunks_filename)
    result = "Great! I'll use specialized support for your model. What seems to be the problem?"
    # Use the phone number as the key
    key = phone_number
    if key in processing_store:
        processing_store[key]["result"] = result
    else:
        print(f"Key {key} not found in processing_store")

def data_store(issue: str, remote_phone: str, uuid_id: str, session_id: str):
    # Fetch conversation history from database
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={{outsystems1.database.windows.net}};"
        f"DATABASE={{OUTSYSTEM_API}};"
        f"UID={{Galaxy}};"
        f"PWD={{OutSystems@123}}"
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT CAST(message_text AS NVARCHAR(MAX)) as message_text,
               CAST(response AS NVARCHAR(MAX)) as response,
               sent_by
        FROM l1_chat_history
        WHERE session_key = ?
        ORDER BY created_at ASC
    """, (session_id,))
   
    chat_records = cursor.fetchall()
   
    # Format conversation history with specific spacing
    formatted_history = ""
    for msg_text, response, sent_by in chat_records:
        if sent_by == "user" and msg_text:
            formatted_history += f"User : \r\n{msg_text}{' ' * 15}\r\n"
        if sent_by == "bot" and response:
            formatted_history += f"{' ' * 15}Bot : {response}\r\n"
   
    ist_timezone = pytz.timezone("Asia/Kolkata")
    current_datetime = dt.datetime.now(ist_timezone)
    cursor.execute(
        """
        INSERT INTO WhatsAppMsgs 
        (id, uuid, session_key, message_text, media_url, media_type, media_mime_type, created_at, remote_phone_number, _2chat_link, channel_phone_number, sent_by, Issue)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            uuid_id,
            uuid_id,
            session_id,
            formatted_history,
            "NULL",
            "NULL",
            "NULL",
            current_datetime,
            remote_phone,
            "NULL",
            "+919322261280",
            "NULL",
            issue,
        ),
    )
    conn.commit()
    conn.close()
   
    return "Done"

def check_query_type(message: str, phone_number: str, current_last_uuid: list):
    """Background task to determine query type and store result"""
    result, dt_id, question_text, action = get_best_matching_tag(message)
    
    # Remove '+91' prefix for processing store key
    key = phone_number 
    
    if result is not None:
        solution_type = "DT"
        if key in processing_store:
            processing_store[key]["result"] = {
                "type": "DT",
                "question_text": question_text,
                "dt_id": dt_id,
                "action": action,
                "result": result
            }
    else:
        solution_type = "RAG"
        if key in processing_store:
            processing_store[key]["result"] = {
                "type": "RAG"
            }
    
    current_last_uuid.append(str(uuid))
    set_stage(stage="tech_support", phone_number=phone_number, 
             solution_type=solution_type, last_uuid=current_last_uuid)

def get_file_loader(file_path: str, file_type: str):
    """Get appropriate loader based on file type"""
    if file_type == "application/pdf":
        # Return a simple function to extract text from PDF
        return lambda: extract_text_from_pdf(file_path)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Return a simple function to extract text from DOCX
        return lambda: extract_text_from_docx(file_path)
    elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        # Return a simple function to extract text from Excel
        return lambda: extract_text_from_excel(file_path)
    elif file_type.startswith("image/"):
        # Return a simple function to extract text from image
        return lambda: extract_text_from_image(file_path)

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Extract text from PDF using PyPDF2, fallback to OCR if text content is too small"""
    chunks = []
    try:
        # First try normal PDF text extraction
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # If text content is too small, try OCR
                if len(text.strip()) < 100:
                    print(f"Text content too small on page {page_num + 1}, falling back to OCR")
                    # Convert PDF page to image
                    images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
                    if images:
                        # Save image temporarily
                        temp_image_path = f"temp_page_{page_num}.png"
                        images[0].save(temp_image_path, 'PNG')
                        
                        # Perform OCR
                        try:
                            text = pytesseract.image_to_string(Image.open(temp_image_path))
                        except Exception as e:
                            print(f"OCR error on page {page_num + 1}: {str(e)}")
                        finally:
                            # Clean up temporary image
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                
                # Split text into chunks with metadata
                text_chunks = split_text_into_chunks(text)
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_metadata = {
                        "source": file_path,
                        "page": page_num + 1,
                        "chunk": chunk_idx + 1,
                        "total_chunks": len(text_chunks),
                        "extraction_method": "ocr" if len(text.strip()) < 100 else "direct",
                        "timestamp": datetime.now().isoformat()
                    }
                    chunks.append({
                        "page_content": chunk_text,
                        "metadata": chunk_metadata
                    })
        
        return chunks
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")

def extract_text_from_excel(file_path: str) -> str:
    """Extract text from Excel using pandas"""
    try:
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        print(f"Error extracting text from Excel: {str(e)}")

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using PIL (placeholder - would need OCR for actual text extraction)"""
    try:
        # This is a placeholder - in a real implementation, you would use an OCR library
        # For now, we'll just return a message indicating the image was processed
        return f"Image processed: {os.path.basename(file_path)}"
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap and proper sentence boundaries"""
    chunks = []
    sentences = text.split('. ')
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(''.join(current_chunk))
            # Start new chunk with overlap
            overlap_size = 0
            overlap_chunk = []
            for prev_sentence in reversed(current_chunk):
                if overlap_size + len(prev_sentence) <= chunk_overlap:
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_size += len(prev_sentence)
                else:
                    break
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def process_document(file_path: str, file_type: str):
    """Process document and return chunks with metadata"""
    if file_type == "application/pdf":
        # Get chunks with metadata from PDF
        chunks = extract_text_from_pdf(file_path)
        
        # If no chunks were found, try alternative method
        if not chunks:
            pdf_document = fitz.open(file_path)
            toc = pdf_document.get_toc()
            if toc:
                for i in range(len(toc)):
                    title = toc[i][1]
                    start_page = toc[i][2]
                    end_page = (
                        toc[i + 1][2] - 1 if i + 1 < len(toc) else pdf_document.page_count
                    )
                    
                    # Extract text for the section
                    section_text = extract_section(pdf_document, start_page, end_page, title)
                    
                    if section_text:
                        # Split section into chunks
                        text_chunks = split_text_into_chunks(section_text)
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            chunk_metadata = {
                                "source": file_path,
                                "section": title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "chunk": chunk_idx + 1,
                                "total_chunks": len(text_chunks),
                                "extraction_method": "toc",
                                "timestamp": datetime.now().isoformat()
                            }
                            chunks.append({
                                "page_content": chunk_text,
                                "metadata": chunk_metadata
                            })
            else:
                # Process page by page if no TOC
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text("text")
                    
                    if page_text:
                        text_chunks = split_text_into_chunks(page_text)
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            chunk_metadata = {
                                "source": file_path,
                                "page": page_num + 1,
                                "chunk": chunk_idx + 1,
                                "total_chunks": len(text_chunks),
                                "extraction_method": "page",
                                "timestamp": datetime.now().isoformat()
                            }
                            chunks.append({
                                "page_content": chunk_text,
                                "metadata": chunk_metadata
                            })
        
        return chunks
    else:
        # For non-PDF files, use the original loader
        loader = get_file_loader(file_path, file_type)
        text = loader()
        chunks = split_text_into_chunks(text)
        return [{"page_content": chunk, "metadata": {"source": file_path}} for chunk in chunks]

def should_clear_history(message: str) -> bool:
    """Check if the message indicates the conversation should end"""
    end_phrases = ["thank you", "thanks", "bye", "goodbye"]
    return any(phrase in message.lower() for phrase in end_phrases)

def is_session_expired(session: dict) -> bool:
    """Check if the session has expired (10 minutes of inactivity)"""
    if 'last_activity' not in session:
        return True
    last_activity = datetime.fromisoformat(session['last_activity'])
    return datetime.now() - last_activity > timedelta(minutes=10)

def query_llm(prompt: str, context: str, chat_history: List[dict]) -> str:
    """Query the local LLM API with chat history"""
    # Format chat history
    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for msg in chat_history[-3:]:  # Include last 3 messages for context
            history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
    
    message = f"""Based on the following context and chat history, please provide a detailed solution for the status error.
    
    {history_text}
    
    Context: {context}
    
    Please provide:
    1. Root cause analysis
    2. Step-by-step solution (one solution at a time)
    3. Prevention measures
    4. Additional recommendations
    
    Status Error: {prompt}"""

    client = openai.OpenAI(
            api_key="100cfa62-287e-4983-8986-010da6320a53",
            base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",
        messages=[{"role":"system","content":str(history_text)},{"role":"user","content":message}],
        temperature=0.1,
        top_p=0.1
    )

    response = response.choices[0].message.content
    
    return response

def load_vector_store():
    """Load the vector store from disk"""
    global vector_store
    try:
        embeddings_path = os.path.join(VECTOR_STORE_PATH, "embeddings.npy")
        texts_path = os.path.join(VECTOR_STORE_PATH, "texts.json")
        metadatas_path = os.path.join(VECTOR_STORE_PATH, "metadatas.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(texts_path) and os.path.exists(metadatas_path):
            embeddings_array = np.load(embeddings_path)
            
            with open(texts_path, "r") as f:
                texts = json.load(f)
                
            with open(metadatas_path, "r") as f:
                metadatas = json.load(f)
                
            vector_store = {
                "embeddings": embeddings_array.tolist(),
                "texts": texts,
                "metadatas": metadatas
            }
            
            print("Vector store loaded successfully")
            return True
        return False
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return False

def save_vector_store():
    """Save the vector store to disk"""
    if vector_store is not None:
        try:
            # Save embeddings, texts, and metadatas separately
            np.save(os.path.join(VECTOR_STORE_PATH, "embeddings.npy"), np.array(vector_store["embeddings"]))
            
            with open(os.path.join(VECTOR_STORE_PATH, "texts.json"), "w") as f:
                json.dump(vector_store["texts"], f)
                
            with open(os.path.join(VECTOR_STORE_PATH, "metadatas.json"), "w") as f:
                json.dump(vector_store["metadatas"], f)
                
            print("Vector store saved successfully")
            return True
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            return False
    return False

# Get Microsoft Graph Access Token
async def get_graph_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'client_credentials',
        'client_id': "d7b17808-debd-4c15-af60-9b2a91fbbb16",
        'client_secret': "SlE8Q~vjipRJf1OkRnBwQW3idnNgEZsG_pSztdat",
        'scope': 'https://graph.microsoft.com/.default'
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=headers) as resp:
            result = await resp.json()
            return result.get("access_token")

async def messages(req):
    body = await req.json()
    activity = Activity().deserialize(body)

    async def on_turn(turn_context: TurnContext):
        activity = turn_context.activity

        print(f"\n== Incoming Activity ==")
        from_user = activity.from_property
        user_id = from_user.id  # Unique ID (per channel like Teams)
        user_name = from_user.name  # Display name (e.g., "Sagar Agicha")
        aad_id = from_user.aad_object_id
        email = "Not found"

        # Get user email from Microsoft Graph if AAD ID is available
        if aad_id:
            access_token = await get_graph_token()
            headers = {"Authorization": f"Bearer {access_token}"}
            graph_url = f"https://graph.microsoft.com/v1.0/users/{aad_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(graph_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        email = data.get("mail") or data.get("userPrincipalName")
                        phone_number = data.get("mobilePhone")
                        phone_number = phone_number
                        job_title = data.get("jobTitle")

        print(f"User ID: {user_id}")
        print(f"User Name: {user_name}")
        print(f"Phone Number: {phone_number}")
        print(f"Job Title: {job_title}")
        print(f"Email: {email}")
        print(f"Type: {activity.type}")
        print(f"Text: {activity.text}")
        print(f"From: {activity.from_property.name}")
        print(f"Conversation ID: {activity.conversation.id}")
        print(f"Activity ID: {activity.id}\n")

        if activity.id in recent_activity_ids:
            print(f"[Deduplicated] Activity {activity.id} already processed.\n")
            return
        recent_activity_ids.add(activity.id)
        if len(recent_activity_ids) > 10:
            recent_activity_ids.clear()
        
        if activity.attachments:
            for attachment in activity.attachments:
                content_type = attachment.content_type
                if content_type == 'text/html':
                    continue
                content_url = attachment.content_url
                name = attachment.name

                print(f"Attachment received: {name} ({content_type})")
                print(f"URL: {content_url}")

        if activity.type == ActivityTypes.message:
            # Use email from Graph API if available, otherwise use the one from activity
            user_validation = check_text_content(activity.text)
            if user_validation['is_valid']:
                if activity.text.lower() == "use_local":
                    use_which_gpu = '1'
                    with open("use_which_gpu.txt", "w") as file:
                        file.write(str(use_which_gpu))
                    await turn_context.send_activity(f"Using local GPU")
                    return
    
                elif activity.text.lower() == "use_cloud":
                    use_which_gpu = '0'
                    with open("use_which_gpu.txt", "w") as file:
                        file.write(str(use_which_gpu))
                    await turn_context.send_activity(f"Using cloud GPU")
                    return

                else:
                    stage_data = get_stage(phone_number)
                    current_stage = stage_data.get('stage', '')
                    rag_no = stage_data.get('rag_no', 0)
                    solution_type = stage_data.get('solution_type', "0")
                        
                    if get_stage(phone_number) == {}:
                        cursor.execute("""
                            SELECT user_name
                            FROM l1_tree 
                            WHERE phone_number = ?
                        """, (phone_number,))
                        
                        result = cursor.fetchone()

                        if result:
                            cursor.execute("""
                                SELECT user_name, com_name, mo_name
                                FROM l1_tree 
                                WHERE phone_number = ?
                            """, (phone_number,))
                            
                            result = cursor.fetchone()
                            if result:
                                username, com_name, mo_name = result
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = activity.conversation.id
                                session_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        activity.text,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = activity.conversation.id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"Welcome {username}\nCan you please confirm your this {mo_name} is your Model Name?",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                set_stage(stage = "data_found", phone_number = phone_number, com_name = com_name, mo_name = mo_name, user_name = username, session_key=session_key)    
                                await turn_context.send_activity(f"Welcome {username}\nCan you please confirm your this {mo_name} is your Model Name?")
                                return
                            
                            else:
                                set_stage("no_data", phone_number=phone_number)
                                await turn_context.send_activity(f"No user data found do you enter a new model name?")
                                return
                            
                        else:
                            store_messages(uuid_id = activity.conversation.id, session_id = session_key, message = activity.text, remote_phone_number=phone_number, sent_by = "user")
                    
                    elif get_stage(phone_number)['stage'] == "data_found":
                        user_response = activity.text.lower()
                        yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect"]
                        no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect"]
                        client = openai.OpenAI(
                            api_key=OPENAI_API_KEY,
                            base_url="https://api.sambanova.ai/v1",
                        )

                        response = client.chat.completions.create(
                            model="Meta-Llama-3.1-8B-Instruct",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant give the response in 'Yes' or 'No' or 'Not related'"},
                                {"role": "user", "content": f"Based on the following text: '{user_response}', please respond with either 'Yes', 'No', or 'Not related' in just 1 word only and no extra explantion. Here are some examples: 'that's right' - 'Yes', 'correct' - 'Yes', 'wrong' - 'No', 'incorrect' - 'No', 'haa' - 'Yes', 'nahi' - 'No', 'naa' - 'No', 'y' - 'Yes', 'n' - 'No', 'i m having problem' - 'Not related', 'i cant login to my outlook' - 'Not related', 'absolutely' - 'Yes', 'never' - 'No', 'definitely' - 'Yes', 'no way' - 'No', 'help me with my email' - 'Not related', 'I forgot my password' - 'Not related'."}
                            ],
                            temperature=0.1,
                            top_p=0.1
                        )

                        print(response)
                        response = response.choices[0].message.content
                        
                        session_key = get_stage(phone_number).get("session_key", "")
                        
                        # Direct string matching instead of embeddings
                        user_response = user_response.strip().lower()
                        
                        # Check if response contains any yes variations
                        max_similarity = 1.0 if any(yes_word in user_response for yes_word in yes_variations) else 0.0
                        no_max_similarity = 1.0 if any(no_word in user_response for no_word in no_variations) else 0.0
                        
                        if response.lower() == "yes":
                            phone_number = phone_number 
                            cursor.execute("""
                                SELECT user_name, com_name, mo_name
                                FROM l1_tree 
                                WHERE phone_number = ?
                            """, (phone_number,))

                            result = cursor.fetchone()
                            if result:
                                username, com_name, mo_name = result
                            default_pdf_path = "/home/sagar/Master_pdfs/pdfs/"
                            default_encode_path = "/home/sagar/Master_pdfs/encodings/"
                            default_chunks_path = "/home/sagar/Master_pdfs/chunks/"
                            unique_laptop = {'Lenovo L14':'lenovo_l14.pdf', 'Lenovo Thinkbook 14':'Not Found', 'Lenovo Thinkpad E14 Gen5':'lenovo_e14.pdf', 'L470' : 'lenovo_e14.pdf', 
                                             'Latitude 3420':'dell_latitude_3420.pdf', 'K 14':'lenovo_k14.pdf', 'Lenovo X1 Yoga 6th Gen':'lenovo_X1_Yoga_Gen_6.pdf', 
                                             'DELL Latitude 7440':'Not Found', 'Lenovo V14':'lenove_v14.pdf', 'MicroSoft Surface Laptop Go 3':'microsoft_surface_go_3.pdf',
                                             'Yoga Duet 7-13ITL6':'Not Found', 'Dell Latitude 7420':'dell_latitude_7420.pdf', 'Latitude 3420':'dell_latitude_3420.pdf'}
                            
                            if mo_name in unique_laptop:
                                pdf_file = default_pdf_path + unique_laptop[mo_name]
                                encodings_filename = default_encode_path + f"{unique_laptop[mo_name].split('.')[0]}.npy"
                                chunks_filename = default_chunks_path + f"{unique_laptop[mo_name].split('.')[0]}.pkl"

                            vector_file = encodings_filename
                            set_stage("tech_support", phone_number, com_name, mo_name, username, pdf_file=pdf_file, vector_file=vector_file, chunks_file=chunks_filename)
                            result = "Great! I'll use specialized support for your model. What seems to be the problem?"
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            #session_key = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    "Great! I'll use specialized support for your model. What seems to be the problem?",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            await turn_context.send_activity(f"Great! I'll use specialized support for your model. What seems to be the problem?")
                            return

                        elif response.lower() == "no":
                            set_stage("no_data", phone_number)
                            await turn_context.send_activity(f"Please let me know your model name")
                            return

                        else:
                            set_stage("data_found", phone_number)
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            #session_key = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    "Please Say Yes or No",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            await turn_context.send_activity(f"Please Say Yes or No")
                            return 

                    elif get_stage(phone_number)['stage'] == "tech_support":
                        stage_data = get_all_data(phone_number)
                        with open("use_which_gpu.txt", "r") as file:
                            use_which_gpu = file.read()
                        pdf_file = stage_data.get('pdf_file')
                        encodings_file = stage_data.get('vector_file')
                        chunks_file = stage_data.get('chunks_file')
                        conversation_history = stage_data.get('conversation_history', [])
                        solution_type = stage_data.get('solution_type', "0")
                        vector_file = stage_data.get('vector_file')
                        current_last_uuid = get_stage(phone_number).get("last_uuid", [])
                        rag_no = stage_data.get('rag_no', 0)
                        user_response = activity.text.lower()
                        yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect", "haa"]
                        no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect", "nahi", "na"]
                        session_key = get_stage(phone_number).get("session_key", "")

                        cursor.execute("""
                            SELECT assets_serial_number
                            FROM l1_tree 
                            WHERE phone_number = ?
                        """, (phone_number,))
                        
                        result = cursor.fetchone()
                        assets_serial_number = result[0]
                        
                        # Direct string matching instead of embeddings
                        user_response = user_response.strip().lower()
                        
                        if solution_type == "shutdown":
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            #ession_key = str(uuid.uuid4())
                            session_key = get_stage(phone_number).get("session_key", "")
                            issue = "Ram Upgrade"
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"I think you need to change or upgrade your ram. \nWould you like to book an agent?",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            set_stage(stage="tech_support", solution_type="shutdown1", phone_number=phone_number)
                            await turn_context.send_activity(f"I think you need to change or upgrade your ram. \nWould you like to book an agent?")
                            return

                        elif solution_type == "shutdown1":
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            #ession_key = str(uuid.uuid4())
                            session_key = get_stage(phone_number).get("session_key", "")
                            issue = "Ram Upgrade"
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            clear_stage(phone_number)
                            data_store(issue, phone_number, activity.conversation.id, session_key)
                            await turn_context.send_activity(f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                            return 
                        
                        if "password" in user_response:
                            if "amaha" in user_response:
                                solution_type = "password"
                                set_stage(stage="tech_support", solution_type="password", phone_number=phone_number)

                        def contains_whole_word(text, word):
                            return f' {word} ' in f' {text.lower()} '

                        # Check for yes variations
                        max_similarity = 1.0 if any(contains_whole_word(user_response, yes_word) for yes_word in yes_variations) else 0.0
                        # Check for no variations
                        no_max_similarity = 1.0 if any(contains_whole_word(user_response, no_word) for no_word in no_variations) else 0.0

                        client = openai.OpenAI(
                            api_key=OPENAI_API_KEY,
                            base_url="https://api.sambanova.ai/v1",
                        )

                        if len(user_response) > 8:
                            response = user_response
                        else:
                            response = client.chat.completions.create(
                                model="Meta-Llama-3.1-8B-Instruct",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant give the response in 'Yes' or 'No' or 'Not related'"},
                                    {"role": "user", "content": f"Based on the following text: '{user_response}', please respond with either 'Yes', 'No', or 'Not related' in just 1 word only and no extra explantion. Here are some examples: 'that's right' - 'Yes', 'correct' - 'Yes', 'wrong' - 'No', 'incorrect' - 'No', 'haa' - 'Yes', 'nahi' - 'No', 'naa' - 'No', 'y' - 'Yes', 'n' - 'No', 'i m having problem' - 'Not related', 'i cant login to my outlook' - 'Not related', 'absolutely' - 'Yes', 'never' - 'No', 'definitely' - 'Yes', 'no way' - 'No', 'help me with my email' - 'Not related', 'I forgot my password' - 'Not related'."}
                                ],
                                temperature=0.1,
                                top_p=0.1
                            )

                            print(response)
                            response = response.choices[0].message.content
                        
                        session_key = get_stage(phone_number).get("session_key", "")

                        if solution_type == "password":
                            set_stage(stage="tech_support", solution_type="password1", phone_number=phone_number)
                            otp_code = str(random.randint(100000, 999999))
                            file = open("otp.txt", "w")
                            file.write(otp_code)
                            file.close()
                            account_sid = TWILIO_ACCOUNT_SID
                            auth_token = TWILIO_AUTH_TOKEN
                            client = Client(account_sid, auth_token)
                            message = client.messages.create(
                                from_='+19473005715', 
                                body=f"Your OTP is {otp_code}",
                                to=phone_number
                            )
                            print(message.sid)
                            await turn_context.send_activity(f"An OTP has been sent to your registered number to confirm your identity. Please enter that OTP.")
                            return

                        elif solution_type == "password1":
                            with open("otp.txt", "r") as file:
                                otp_code = file.read()

                            if activity.text == otp_code:
                                set_stage(stage="tech_support", solution_type="password2", phone_number=phone_number)
                                await turn_context.send_activity(f"OTP is verified \nPlease enter the new password")
                                return
                            else:
                                await turn_context.send_activity(f"Invalid OTP \nPlease enter the otp sent to your phone number")
                                return
                            
                        elif solution_type == "password2":
                            current_password = "Sagar1010"
                            # background_tasks.add_task(
                            #     navigate_to_login,
                            #     phone_number=phone_number,
                            #     current_password=current_password,
                            #     new_password=activity.text
                            # )
                            # clear_stage(phone_number)
                            set_stage(stage="tech_support", solution_type="password3", phone_number=phone_number)
                            await turn_context.send_activity(f"Your password has been changed successfully. \nPlease try after 2 minutes and confirm if you can log in successfully.")
                            return 
                        
                        elif solution_type == "password3":
                            clear_stage(phone_number)
                            await turn_context.send_activity(f"Thank You for contacting us.")
                            return

                        if rag_no >= 4:
                            client = openai.OpenAI(
                                api_key=OPENAI_API_KEY,
                                base_url="https://api.sambanova.ai/v1",
                            )

                            if len(user_response) > 8:
                                response = user_response
                            else:
                                response = client.chat.completions.create(
                                    model="Meta-Llama-3.1-8B-Instruct",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant give the response in 'Yes' or 'No' or 'Not related'"},
                                        {"role": "user", "content": f"Based on the following text: '{user_response}', please respond with either 'Yes', 'No', or 'Not related' in just 1 word only and no extra explantion. Here are some examples: 'that's right' - 'Yes', 'correct' - 'Yes', 'wrong' - 'No', 'incorrect' - 'No', 'haa' - 'Yes', 'nahi' - 'No', 'naa' - 'No', 'y' - 'Yes', 'n' - 'No', 'i m having problem' - 'Not related', 'i cant login to my outlook' - 'Not related', 'absolutely' - 'Yes', 'never' - 'No', 'definitely' - 'Yes', 'no way' - 'No', 'help me with my email' - 'Not related', 'I forgot my password' - 'Not related'."}
                                    ],
                                    temperature=0.1,
                                    top_p=0.1
                                )

                                print(response)
                                response = response.choices[0].message.content

                            if response.lower() == "no":
                                session_key = get_stage(phone_number).get("session_key", "")
                                set_stage(stage="live_agent", phone_number=phone_number)
                                try:
                                    current_last_uuid.append(str(uuid))
                                except:
                                    print('error')
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = activity.conversation.id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        activity.text,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = activity.conversation.id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        "Do you want to connect with a live agent?",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                await turn_context.send_activity(f"Do you want to connect with a live agent?")
                                return
                            
                        elif solution_type == "0":
                            result, dt_id, question_text, action = get_best_matching_tag(activity.text)

                            if result is not None:
                                solution_type = "DT"
                                current_stage = "start_solution"
                                store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=question_text, dt_id=dt_id, action=action)
                                set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = activity.conversation.id
                                #session_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        activity.conversation.id,
                                        session_key,
                                        activity.text,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = activity.conversation.id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"{question_text} \nCan you confirm this is related to your issue?",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()

                                current_last_uuid.append(str(uuid))
                                set_stage(stage="tech_support", phone_number=phone_number, solution_type=solution_type, last_uuid=current_last_uuid)
                                await turn_context.send_activity(f"{question_text} \nCan you confirm this is related to your issue?")
                                return 
                            
                            else:
                                solution_type = "RAG"
                                current_last_uuid.append(str(uuid))
                                set_stage(stage="tech_support", phone_number=phone_number, solution_type=solution_type, last_uuid=current_last_uuid)
                    
                        if solution_type == "RAG":
                            if chunks_file != '0':
                                with open(chunks_file, 'rb') as f:
                                    chunks = pickle.load(f)
                                context_encodings = np.load(encodings_file)
                                
                                # Get user's laptop model info
                                stage_data = get_all_data(phone_number)
                                com_name = stage_data.get('com_name', '')
                                mo_name = stage_data.get('mo_name', '')
                                user_name = stage_data.get('user_name', '')
                                laptop_info = f"{mo_name} Serial Number: {assets_serial_number}"

                                # Use LLM to determine if this is a satisfaction response or step confirmation
                                client = openai.OpenAI(
                                    api_key=OPENAI_API_KEY,
                                    base_url="https://api.sambanova.ai/v1",
                                )

                                # Get the last assistant message to check context
                                last_assistant_msg = None
                                for msg in reversed(conversation_history):
                                    if msg.get("role") == "assistant":
                                        last_assistant_msg = msg.get("content", "")
                                        break

                                # If rag_no >= 4, analyze if user wants to connect to agent or is satisfied
                                if rag_no >= 4:
                                    agent_analysis_prompt = f"""Based on the following conversation context, determine if the user wants to:
                                                                1. Connect to a live agent (because solutions didn't work)
                                                                2. End the conversation (because they are satisfied with the solution)
                                                                3. Continue troubleshooting (because they want to try more solutions)

                                                                Last assistant message: {last_assistant_msg}
                                                                User response: {user_response}

                                                                Respond with exactly one word:
                                                                - "connect_agent" if user wants to connect to a live agent
                                                                - "satisfied" if user is satisfied and wants to end conversation
                                                                - "continue" if user wants to continue troubleshooting

                                                                Example responses:
                                                                - "yes" -> "connect_agent"
                                                                - "yes i want to talk to someone" -> "connect_agent"
                                                                - "yes connect me to an agent" -> "connect_agent"
                                                                - "yes that fixed it" -> "satisfied"
                                                                - "yes it's working now" -> "satisfied"
                                                                - "its working" -> "satisfied"
                                                                - "no let's try something else" -> "continue"
                                                                - "i want to try another solution" -> "continue"
                                                                """

                                    response_type = client.chat.completions.create(
                                        model="Meta-Llama-3.1-8B-Instruct",
                                        messages=[{"role": "system", "content": agent_analysis_prompt}],
                                        temperature=0.1,
                                        top_p=0.1
                                    ).choices[0].message.content.strip().lower()

                                    if response_type == "connect_agent":
                                        # User wants to connect to agent
                                        clear_stage(phone_number)
                                        result = get_user_interaction(phone_number)
                                        try:
                                            issue = result.get('issue', None)
                                        except:
                                            issue = 'None'
                                        session_key = get_stage(phone_number).get("session_key", "")
                                        data_store(issue, phone_number, str(uuid.uuid4()), session_key)
                                        await turn_context.send_activity(f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                        return
                                    
                                    elif response_type == "satisfied":
                                        # User is satisfied with the solution
                                        clear_stage(phone_number)
                                        await turn_context.send_activity(f"Thank you for contacting us.")
                                        return

                                # Add system message with laptop context
                                conversation_history.append({"role": "system", "content": f"""You are a helpful technical support assistant for {laptop_info} laptops. 
                                                                Your responses should follow these rules:

                                                                1. Provide ONE solution at a time, clearly labeled as "Step numbers [1, 2, n]: [solution]"
                                                                2. Keep responses clear and concise
                                                                3. Use professional yet friendly tone
                                                                4. Include relevant context from the manual when applicable
                                                                5. After providing a solution, ask "Please try this step and let me know if you need any clarification"
                                                                6. If the solution worked, ask if they need help with anything else
                                                                7. If the solution didn't work, provide the next step
                                                                8. After 3 attempts, if user agrees to connect with agent, respond with exactly "connect agent = true"
                                                                9. Otherwise, continue providing solutions one at a time

                                                                Example conversation flow:
                                                                User: "I can't connect to the printer"
                                                                Assistant: "Step 1: Let's check your network connection first. Open Windows Settings and verify you're connected to the correct network. Please try this step and let me know if you need any clarification."
                                                                User: "I'm connected to the right network"
                                                                Assistant: "Step 2: Great! Now let's try adding the printer. Open Windows Settings > Printers & Scanners > Add a printer. Please try this step and let me know if you need any clarification."
                                                                User: "I can see the printer but can't connect"
                                                                Assistant: "Step n: Let's try resetting the printer connection. First, remove the existing printer from your system. Please try this step and let me know if you need any clarification."

                                                                If the user's response is unclear, ask for clarification.
                                                                If they're still having issues after multiple attempts, suggest connecting with a live agent."""})
                                                                
                                conversation_history.append({"role": "user", "content": user_response})
                                
                                retrieved_context = rag.retrieve_context(user_response, chunks, context_encodings)
                                conversation_history.append({"role": "system", "content": f"Context from {laptop_info} manual:\n{retrieved_context}"})

                                if use_which_gpu == '1':
                                    request = requests.post("http://192.168.200.67:30501/process_text", json={"prompt" : str(conversation_history), "text" : user_response})
                                    response = request.json()
                                    response = response['generated_text']
                                else:
                                    client = openai.OpenAI(
                                        api_key="100cfa62-287e-4983-8986-010da6320a53",
                                        base_url="https://api.sambanova.ai/v1",
                                    )

                                    response = client.chat.completions.create(
                                        model="Meta-Llama-3.1-8B-Instruct",
                                        messages=[{"role":"system","content":str(conversation_history)},{"role":"user","content":user_response}],
                                        temperature=0.1,
                                        top_p=0.1
                                    )

                                    response = response.choices[0].message.content

                                conversation_history.append({"role": "assistant", "content": response})
                                rag_no += 1
                                
                                # Check if LLM wants to connect to agent
                                if response.strip().lower() == "connect agent = true":
                                    # Get the issue from user interaction
                                    result = get_user_interaction(phone_number)
                                    issue = result.get('issue', None)
                                    session_key = get_stage(phone_number).get("session_key", "")
                                    # Store the agent connection message
                                    # Clear stage and store data
                                    clear_stage(phone_number)
                                    data_store(issue, phone_number, str(uuid.uuid4()), session_key)
                                    await turn_context.send_activity(f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                    return
                                
                                if rag_no >= 4:
                                    response = response + "\n\nI notice we've tried a few solutions. Would you like to connect with a live agent for more specialized help?"
                                
                                # Update the stage while preserving all existing user data
                                set_stage(
                                    stage="tech_support",
                                    phone_number=phone_number,
                                    com_name=com_name,
                                    mo_name=mo_name,
                                    user_name=user_name,
                                    pdf_file=pdf_file,
                                    vector_file=vector_file,
                                    conversation_history=conversation_history,
                                    chunks_file=chunks_file,
                                    solution_type="RAG",
                                    rag_no=rag_no,
                                    session_key=stage_data.get('session_key', '0')
                                )

                                # Store the user message in chat history
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        activity.conversation.id,
                                        session_key,
                                        activity.text,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        activity.conversation.id,
                                        session_key,
                                        response,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()

                                await turn_context.send_activity(response)
                                return

                        elif solution_type == "DT":
                            if get_user_interaction(phone_number)["stage"] == "start_solution":
                                if response.lower() == "yes":
                                    result = get_user_interaction(phone_number)
                                    issue = result.get('issue', None)
                                    dt_id = result.get('dt_id', None)
                                    action = result.get('action', None)
                                    yes_id = result.get('yes_id', None)
                                    if issue and dt_id and action:
                                        cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                                        question_text = cursor.fetchone()

                                        cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                                        link_id = cursor.fetchone()

                                        cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (action, dt_id))
                                        no_id = cursor.fetchone()

                                        cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (action, dt_id))
                                        yes_id = cursor.fetchone()

                                        action = no_id[0]
                                        yes_id = yes_id[0]

                                        current_stage = "ongoing_solution"
                                        store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                                        if link_id[0] == "0":
                                            current_last_uuid.append(str(uuid))
                                            ist_timezone = pytz.timezone("Asia/Kolkata")
                                            current_datetime = dt.datetime.now(ist_timezone)
                                            
                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    activity.text,
                                                    "",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "user",
                                                ),
                                            )
                                            conn.commit()

                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    "",
                                                    f"{question_text[0]}",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "bot",
                                                ),
                                            )
                                            conn.commit()
                                            set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                            await turn_context.send_activity(question_text[0])
                                            return

                                        else:
                                            video_name = link_id[0]
                                            ist_timezone = pytz.timezone("Asia/Kolkata")
                                            current_datetime = dt.datetime.now(ist_timezone)
                                            
                                            uuid_id = activity.conversation.id
                                            #ession_key = str(uuid.uuid4())
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    activity.text,
                                                    "",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "user",
                                                ),
                                            )
                                            conn.commit()

                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    "",
                                                    f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "bot",
                                                ),
                                            )
                                            conn.commit()
                                            store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                            set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                            await turn_context.send_activity(question_text[0] + "\n" + f"{link_url}/videos/{video_name}")
                                            return
                                        
                                else:
                                    solution_type = "RAG"
                                    current_last_uuid.append(str(uuid))
                                    set_stage(stage="tech_support", phone_number=phone_number, solution_type=solution_type, last_uuid=current_last_uuid)
                                    current_stage = "location_requested"
                                    store_user_interaction(phone_number, current_stage, 0)
                                    current_last_uuid.append(str(uuid))
                                    set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                    await turn_context.send_activity("Please describe the issue again in more simple words.")
                                    return

                            elif get_user_interaction(phone_number)["stage"] == "ongoing_solution":
                                result = get_user_interaction(phone_number)
                                issue = result.get('issue', None)
                                dt_id = result.get('dt_id', None)
                                action = result.get('action', None)
                                yes_id = result.get('yes_id', None)
                                no_id = result.get('no_id', None)

                                if yes_id == 'change_password':
                                    set_stage(stage="tech_support", solution_type="password1", phone_number=phone_number)
                                    otp_code = str(random.randint(100000, 999999))
                                    file = open("otp.txt", "w")
                                    file.write(otp_code)
                                    file.close()
                                    account_sid = TWILIO_ACCOUNT_SID
                                    auth_token = TWILIO_AUTH_TOKEN
                                    client = Client(account_sid, auth_token)
                                    message = client.messages.create(
                                        from_='+19473005715', 
                                        body=f"Your OTP is {otp_code}",
                                        to=phone_number
                                    )
                                    print(message.sid)
                                    await turn_context.send_activity("An OTP has been sent to your registered number to confirm your identity. \nPlease enter that OTP")
                                    return
                                
                                elif response.lower() == "yes":
                                    if yes_id == "handover":
                                        current_stage = "live_agent"
                                        store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action)
                                        current_last_uuid.append(str(uuid))
                                        ist_timezone = pytz.timezone("Asia/Kolkata")
                                        current_datetime = dt.datetime.now(ist_timezone)
                                        
                                        uuid_id = activity.conversation.id
                                        #ession_key = str(uuid.uuid4())
                                        cursor.execute(
                                            """
                                            INSERT INTO l1_chat_history 
                                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                            (
                                                uuid_id,
                                                session_key,
                                                activity.text,
                                                "",
                                                phone_number,
                                                "Teams",
                                                str(current_datetime),
                                                "user",
                                            ),
                                        )
                                        conn.commit()

                                        uuid_id = activity.conversation.id
                                        cursor.execute(
                                            """
                                            INSERT INTO l1_chat_history 
                                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                            (
                                                uuid_id,
                                                session_key,
                                                "",
                                                f"Sorry It seems I cant help you\n Do you want to connect to an Live Agent?",
                                                phone_number,
                                                "Teams",
                                                str(current_datetime),
                                                "bot",
                                            ),
                                        )
                                        conn.commit()
                                        set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                        await turn_context.send_activity("Sorry It seems I cant help you\n Do you want to connect to an Live Agent?")
                                        return
                                    
                                    elif issue and dt_id and action:
                                        cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                        question_text = cursor.fetchone()

                                        cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                        link_id = cursor.fetchone()

                                        cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (yes_id, dt_id))
                                        no_id = cursor.fetchone()
                                        print("no_id = ", no_id)

                                        cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (yes_id, dt_id))
                                        yes_id = cursor.fetchone()
                                        print("yes_id = ", yes_id)

                                        if yes_id:  
                                            yes_id = yes_id[0]
                                        if no_id:
                                            no_id = no_id[0]

                                        if link_id[0] == "0":
                                            current_last_uuid.append(str(uuid))
                                            current_stage = "ongoing_solution"
                                            store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                            ist_timezone = pytz.timezone("Asia/Kolkata")
                                            current_datetime = dt.datetime.now(ist_timezone)
                                            
                                            uuid_id = activity.conversation.id
                                            #ession_key = str(uuid.uuid4())
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    activity.text,
                                                    "",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "user",
                                                ),
                                            )
                                            conn.commit()

                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    "",
                                                    f"{question_text[0]}",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "bot",
                                                ),
                                            )
                                            conn.commit()
                                            store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                            set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                            await turn_context.send_activity(question_text[0])

                                        else:
                                            video_name = link_id[0]
                                            ist_timezone = pytz.timezone("Asia/Kolkata")
                                            current_datetime = dt.datetime.now(ist_timezone)
                                            
                                            uuid_id = activity.conversation.id
                                            #ession_key = str(uuid.uuid4())
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    activity.text,
                                                    "",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "user",
                                                ),
                                            )
                                            conn.commit()

                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    "",
                                                    f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "bot",
                                                ),
                                            )
                                            conn.commit()
                                            store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                            set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                            await turn_context.send_activity(question_text[0] + "\n" + f"{link_url}/videos/{video_name}")
                                            return
                                        
                                elif response.lower() == "yes":
                                    result = get_user_interaction(phone_number)
                                    yes_id = result.get('yes_id', None)
                                    if yes_id != "solved":
                                        if yes_id == "handover":
                                            current_stage = "live_agent"
                                            store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                                            current_last_uuid.append(str(uuid))
                                            ist_timezone = pytz.timezone("Asia/Kolkata")
                                            current_datetime = dt.datetime.now(ist_timezone)
                                            
                                            uuid_id = activity.conversation.id
                                            #ession_key = str(uuid.uuid4())
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    activity.text,
                                                    "",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "user",
                                                ),
                                            )
                                            conn.commit()

                                            uuid_id = activity.conversation.id
                                            cursor.execute(
                                                """
                                                INSERT INTO l1_chat_history 
                                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """,
                                                (
                                                    uuid_id,
                                                    session_key,
                                                    "",
                                                    f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                                    phone_number,
                                                    "Teams",
                                                    str(current_datetime),
                                                    "bot",
                                                ),
                                            )
                                            conn.commit()
                                            clear_stage(phone_number)
                                            data_store(issue, phone_number, activity.conversation.id, session_key)
                                            #set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                            await turn_context.send_activity("Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                            return
                                        
                                        elif issue and dt_id and action:
                                            cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                            question_text = cursor.fetchone()

                                            cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                            link_id = cursor.fetchone()

                                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (yes_id, dt_id))
                                            no_id = cursor.fetchone()
                                            print("no_id = ", no_id)

                                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (yes_id, dt_id))
                                            yes_id = cursor.fetchone()
                                            print("yes_id = ", yes_id)

                                            if yes_id:  
                                                yes_id = yes_id[0]
                                            if no_id:
                                                no_id = no_id[0]

                                            if link_id[0] == "0":
                                                current_last_uuid.append(str(uuid))
                                                current_stage = "ongoing_solution"
                                                store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                                current_datetime = dt.datetime.now(ist_timezone)
                                                
                                                uuid_id = activity.conversation.id
                                                #ession_key = str(uuid.uuid4())
                                                cursor.execute(
                                                    """
                                                    INSERT INTO l1_chat_history 
                                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """,
                                                    (
                                                        uuid_id,
                                                        session_key,
                                                        activity.text,
                                                        "",
                                                        phone_number,
                                                        "Teams",
                                                        str(current_datetime),
                                                        "user",
                                                    ),
                                                )
                                                conn.commit()

                                                uuid_id = activity.conversation.id
                                                cursor.execute(
                                                    """
                                                    INSERT INTO l1_chat_history 
                                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """,
                                                    (
                                                        uuid_id,
                                                        session_key,
                                                        "",
                                                        f"{question_text[0]}",
                                                        phone_number,
                                                        "Teams",
                                                        str(current_datetime),
                                                        "bot",
                                                    ),
                                                )
                                                conn.commit()
                                                store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                                set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                                await turn_context.send_activity(question_text[0])
                                                return

                                            else:
                                                video_name = link_id[0]
                                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                                current_datetime = dt.datetime.now(ist_timezone)
                                                
                                                uuid_id = activity.conversation.id
                                                #ession_key = str(uuid.uuid4())
                                                cursor.execute(
                                                    """
                                                    INSERT INTO l1_chat_history 
                                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """,
                                                    (
                                                        uuid_id,
                                                        session_key,
                                                        activity.text,
                                                        "",
                                                        phone_number,
                                                        "Teams",
                                                        str(current_datetime),
                                                        "user",
                                                    ),
                                                )
                                                conn.commit()

                                                uuid_id = activity.conversation.id
                                                cursor.execute(
                                                    """
                                                    INSERT INTO l1_chat_history 
                                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """,
                                                    (
                                                        uuid_id,
                                                        session_key,
                                                        "",
                                                        f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                                        phone_number,
                                                        "Teams",
                                                        str(current_datetime),
                                                        "bot",
                                                    ),
                                                )
                                                conn.commit()
                                                store_user_interaction(phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                                set_stage(stage="tech_support", phone_number=phone_number, last_uuid=current_last_uuid)
                                                await turn_context.send_activity(question_text[0] + "\n" + f"{link_url}/videos/{video_name}")
                                                return

                                    else:
                                        current_last_uuid.append(str(uuid))
                                        ist_timezone = pytz.timezone("Asia/Kolkata")
                                        current_datetime = dt.datetime.now(ist_timezone)
                                        
                                        uuid_id = activity.conversation.id
                                        #ession_key = str(uuid.uuid4())
                                        cursor.execute(
                                            """
                                            INSERT INTO l1_chat_history 
                                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                            (
                                                uuid_id,
                                                session_key,
                                                activity.text,
                                                "",
                                                phone_number,
                                                "Teams",
                                                str(current_datetime),
                                                "user",
                                            ),
                                        )
                                        conn.commit()

                                        uuid_id = activity.conversation.id
                                        cursor.execute(
                                            """
                                            INSERT INTO l1_chat_history 
                                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                            (
                                                uuid_id,
                                                session_key,
                                                "",
                                                f"Thank you for contacting us.",
                                                phone_number,
                                                "Teams",
                                                str(current_datetime),
                                                "bot",
                                            ),
                                        )
                                        conn.commit()
                                        #set_stage(stage="start", phone_number=phone_number, last_uuid=current_last_uuid)
                                        clear_stage(phone_number)
                                        await turn_context.send_activity("Thank you for contacting us.")
                                        return

                            elif get_user_interaction(phone_number)["stage"] == "live_agent":
                                if response.lower() == "yes":
                                    result = get_user_interaction(phone_number)
                                    issue = result.get('issue', None)
                                    ist_timezone = pytz.timezone("Asia/Kolkata")
                                    current_datetime = dt.datetime.now(ist_timezone)
                                    
                                    uuid_id = activity.conversation.id
                                    #ession_key = str(uuid.uuid4())
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            activity.text,
                                            "",
                                            phone_number,
                                            "Teams",
                                            str(current_datetime),
                                            "user",
                                        ),
                                    )
                                    conn.commit()

                                    uuid_id = activity.conversation.id
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            "",
                                            f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                            phone_number,
                                            "Teams",
                                            str(current_datetime),
                                            "bot",
                                        ),
                                    )
                                    conn.commit()
                                    clear_stage(phone_number)
                                    data_store(issue, phone_number, activity.conversation.id, session_key)
                                    await turn_context.send_activity("Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                    return 
                                
                                else:
                                    ist_timezone = pytz.timezone("Asia/Kolkata")
                                    current_datetime = dt.datetime.now(ist_timezone)
                                    
                                    uuid_id = activity.conversation.id
                                    #ession_key = str(uuid.uuid4())
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            activity.text,
                                            "",
                                            phone_number,
                                            "Teams",
                                            str(current_datetime),
                                            "user",
                                        ),
                                    )
                                    conn.commit()

                                    uuid_id = activity.conversation.id
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            "",
                                            f"Thank you for contacting us.",
                                            phone_number,
                                            "Teams",
                                            str(current_datetime),
                                            "bot",
                                        ),
                                    )
                                    conn.commit()
                                    clear_stage(phone_number)
                                    await turn_context.send_activity("Thank you for contacting us.")
                                    return

                    elif get_stage(phone_number)["stage"] == "live_agent":
                        user_response = activity.text.lower()
                        try:
                            result = get_user_interaction(phone_number)
                            issue = result.get('issue', None)
                        except:
                            issue = None
                        yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect", "haa"]
                        no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect", "nahi", "na"]
                        session_key = get_stage(phone_number).get("session_key", "")

                        # Direct string matching instead of embeddings
                        user_response = user_response.strip().lower()
                        
                        # Check if response contains any yes variations
                        max_similarity = 1.0 if any(yes_word in user_response for yes_word in yes_variations) else 0.0
                        no_max_similarity = 1.0 if any(no_word in user_response for no_word in no_variations) else 0.0
                        if max_similarity > 0.7:
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            #ession_key = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            clear_stage(phone_number)
                            data_store(issue, phone_number, activity.conversation.id, session_key)
                            await turn_context.send_activity("Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                            return
                        
                        else:
                            stage_data = get_all_data(phone_number)
                            with open("use_which_gpu.txt", "r") as file:
                                use_which_gpu = file.read()
                            pdf_file = stage_data.get('pdf_file')
                            encodings_file = stage_data.get('vector_file')
                            chunks_file = stage_data.get('chunks_file')
                            conversation_history = stage_data.get('conversation_history', [])
                            solution_type = stage_data.get('solution_type', "0")
                            vector_file = stage_data.get('vector_file')
                            current_last_uuid = get_stage(phone_number).get("last_uuid", [])
                            rag_no = stage_data.get('rag_no', 0)
                            user_response = activity.text.lower()
                            yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect", "haa"]
                            no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect", "nahi", "na"]
                            session_key = get_stage(phone_number).get("session_key", "")
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    activity.text,
                                    "",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = activity.conversation.id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"Thank you for contacting us.",
                                    phone_number,
                                    "Teams",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            set_stage(stage="tech_support", phone_number=phone_number, last_uuid=activity.conversation.id)

                            if chunks_file != '0':
                                with open(chunks_file, 'rb') as f:
                                    chunks = pickle.load(f)
                                context_encodings = np.load(encodings_file)
                                
                                # Get user's laptop model info
                                stage_data = get_all_data(phone_number)
                                com_name = stage_data.get('com_name', '')
                                mo_name = stage_data.get('mo_name', '')
                                user_name = stage_data.get('user_name', '')
                                laptop_info = f"{com_name} {mo_name}"

                                # Use LLM to determine if this is a satisfaction response or step confirmation
                                client = openai.OpenAI(
                                    api_key=OPENAI_API_KEY,
                                    base_url="https://api.sambanova.ai/v1",
                                )

                                # Get the last assistant message to check context
                                last_assistant_msg = None
                                for msg in reversed(conversation_history):
                                    if msg.get("role") == "assistant":
                                        last_assistant_msg = msg.get("content", "")
                                        break

                                # If rag_no >= 4, analyze if user wants to connect to agent or is satisfied
                                if rag_no >= 4:
                                    agent_analysis_prompt = f"""Based on the following conversation context, determine if the user wants to:
                                                                1. Connect to a live agent (because solutions didn't work)
                                                                2. End the conversation (because they are satisfied with the solution)
                                                                3. Continue troubleshooting (because they want to try more solutions)

                                                                Last assistant message: {last_assistant_msg}
                                                                User response: {user_response}

                                                                Respond with exactly one word:
                                                                - "connect_agent" if user wants to connect to a live agent
                                                                - "satisfied" if user is satisfied and wants to end conversation
                                                                - "continue" if user wants to continue troubleshooting

                                                                Example responses:
                                                                - "yes i want to talk to someone" -> "connect_agent"
                                                                - "yes connect me to an agent" -> "connect_agent"
                                                                - "yes that fixed it" -> "satisfied"
                                                                - "yes it's working now" -> "satisfied"
                                                                - "no let's try something else" -> "continue"
                                                                - "i want to try another solution" -> "continue"
                                                                """

                                    response_type = client.chat.completions.create(
                                        model="Meta-Llama-3.1-8B-Instruct",
                                        messages=[{"role": "system", "content": agent_analysis_prompt}],
                                        temperature=0.1,
                                        top_p=0.1
                                    ).choices[0].message.content.strip().lower()

                                    if response_type == "connect_agent":
                                        # User wants to connect to agent
                                        clear_stage(phone_number)
                                        result = get_user_interaction(phone_number)
                                        issue = result.get('issue', None)
                                        session_key = get_stage(phone_number).get("session_key", "")
                                        data_store(issue, phone_number, str(uuid.uuid4()), session_key)
                                        await turn_context.send_activity(f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                        return
                                    
                                    elif response_type == "satisfied":
                                        # User is satisfied with the solution
                                        clear_stage(phone_number)
                                        await turn_context.send_activity(f"Thank you for contacting us.")
                                        return

                                # Add system message with laptop context
                                conversation_history.append({"role": "system", "content": f"""You are a helpful technical support assistant for {laptop_info} laptops. 
                                                                Your responses should follow these rules:

                                                                1. Provide ONE solution at a time, clearly labeled as "Step numbers [1, 2, n]: [solution]"
                                                                2. Keep responses clear and concise
                                                                3. Use professional yet friendly tone
                                                                4. Include relevant context from the manual when applicable
                                                                5. After providing a solution, ask "Please try this step and let me know if you need any clarification"
                                                                6. If the solution worked, ask if they need help with anything else
                                                                7. If the solution didn't work, provide the next step
                                                                8. After 3 attempts, if user agrees to connect with agent, respond with exactly "connect agent = true"
                                                                9. Otherwise, continue providing solutions one at a time

                                                                Example conversation flow:
                                                                User: "I can't connect to the printer"
                                                                Assistant: "Step 1: Let's check your network connection first. Open Windows Settings and verify you're connected to the correct network. Please try this step and let me know if you need any clarification."
                                                                User: "I'm connected to the right network"
                                                                Assistant: "Step 2: Great! Now let's try adding the printer. Open Windows Settings > Printers & Scanners > Add a printer. Please try this step and let me know if you need any clarification."
                                                                User: "I can see the printer but can't connect"
                                                                Assistant: "Step n: Let's try resetting the printer connection. First, remove the existing printer from your system. Please try this step and let me know if you need any clarification."

                                                                If the user's response is unclear, ask for clarification.
                                                                If they're still having issues after multiple attempts, suggest connecting with a live agent."""})
                                                                
                                conversation_history.append({"role": "user", "content": user_response})
                                
                                retrieved_context = rag.retrieve_context(user_response, chunks, context_encodings)
                                conversation_history.append({"role": "system", "content": f"Context from {laptop_info} manual:\n{retrieved_context}"})

                                if use_which_gpu == '1':
                                    request = requests.post("http://192.168.200.67:30501/process_text", json={"prompt" : str(conversation_history), "text" : user_response})
                                    response = request['generated_text']
                                else:
                                    client = openai.OpenAI(
                                        api_key="100cfa62-287e-4983-8986-010da6320a53",
                                        base_url="https://api.sambanova.ai/v1",
                                    )

                                    response = client.chat.completions.create(
                                        model="Meta-Llama-3.1-8B-Instruct",
                                        messages=[{"role":"system","content":str(conversation_history)},{"role":"user","content":user_response}],
                                        temperature=0.1,
                                        top_p=0.1
                                    )

                                    response = response.choices[0].message.content

                                conversation_history.append({"role": "assistant", "content": response})
                                rag_no += 1
                                
                                # Check if LLM wants to connect to agent
                                if response.strip().lower() == "connect agent = true":
                                    # Get the issue from user interaction
                                    result = get_user_interaction(phone_number)
                                    issue = result.get('issue', None)
                                    session_key = get_stage(phone_number).get("session_key", "")
                                    # Store the agent connection message
                                    # Clear stage and store data
                                    clear_stage(phone_number)
                                    data_store(issue, phone_number, str(uuid.uuid4()), session_key)
                                    await turn_context.send_activity(f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --")
                                    return
                                
                                if rag_no >= 4:
                                    await turn_context.send_activity(response + "\n\nI notice we've tried a few solutions. Would you like to connect with a live agent for more specialized help?")
                                    return
                                
                                # Update the stage while preserving all existing user data
                                set_stage(
                                    stage="tech_support",
                                    phone_number=phone_number,
                                    com_name=com_name,
                                    mo_name=mo_name,
                                    user_name=user_name,
                                    pdf_file=pdf_file,
                                    vector_file=vector_file,
                                    conversation_history=conversation_history,
                                    chunks_file=chunks_file,
                                    solution_type="RAG",
                                    rag_no=rag_no,
                                    session_key=stage_data.get('session_key', '0')
                                )

                                # Store the user message in chat history
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        activity.conversation.id,
                                        session_key,
                                        activity.text,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        activity.conversation.id,
                                        session_key,
                                        response,
                                        "",
                                        phone_number,
                                        "Teams",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                
                                await turn_context.send_activity(response)
                                return

    auth_header = req.headers.get("Authorization", "")

    try:
        await adapter.process_activity(activity, auth_header, on_turn)
        return web.Response(status=200, text="OK")
    except Exception as e:
        print(f"[Error] {e}")
        return web.Response(status=500, text=str(e))

app = web.Application()
app.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(app, port=6600)