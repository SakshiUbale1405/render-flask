from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
import docx
import pytesseract
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from googletrans import Translator
from werkzeug.utils import secure_filename
from PIL import Image
from langdetect import detect
from yt_dlp import YoutubeDL
import re
import json
import requests
from whisper import load_model 
import torch
from typing import Optional
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Initialize models (load once at startup)
WHISPER_MODEL = load_model("medium").to("cuda" if torch.cuda.is_available() else "cpu")
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')  # Fixed model name
qa_model = pipeline('question-answering', model='deepset/xlm-roberta-large-squad2')
translator = Translator()

# Helper function to check allowed image files
def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Text extraction functions
def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        return False
    
# Text extraction functions
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text if text else "No readable text found in PDF."

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_image(image_path, lang='eng'):
    if not check_tesseract():
        raise Exception("Tesseract OCR is not installed or not in PATH")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Could not read image file")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
        
        return text if text.strip() else "No readable text found in image."
    except Exception as e:
        raise Exception(f"Image processing error: {str(e)}")


def extract_transcript_yt_dlp(video_url):
    """ Extracts a YouTube transcript using yt-dlp. """
    try:
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,  # Allow auto-generated if manual is missing
            'subtitleslangs': ['en'],
            'outtmpl': '%(id)s.%(ext)s',
            'dumpjson': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get('subtitles', {}).get('en') or info.get('automatic_captions', {}).get('en')
            if subtitles:
                transcript_url = subtitles[-1]['url']
                return transcript_url
            else:
                raise ValueError("No transcript found.")
    except Exception as e:
        raise ValueError(f"Error fetching transcript: {e}")

def preprocess_transcript(transcript_url):
    """Fetch and clean YouTube transcript (remove timestamps and empty lines)."""
    try:
        response = requests.get(transcript_url)
        lines = response.text.splitlines()

        clean_lines = []
        for line in lines:
            if re.match(r'^\d{2}:\d{2}:\d{2}\.\d+', line):
                continue  # Skip timestamps
            if line.strip() == '' or '-->' in line:
                continue  # Skip empty lines and cue lines
            clean_lines.append(line.strip())

        return ' '.join(clean_lines).strip()

    except Exception as e:
        raise ValueError(f"Error processing transcript: {e}")
    
def transcribe_with_whisper(youtube_url: str) -> str:
    """Transcribe audio from YouTube video using Whisper (supports 100+ languages)"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': 'temp_audio',
            'quiet': True
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            audio_path = "temp_audio.wav"
        
        # Transcribe with forced English output
        result = WHISPER_MODEL.transcribe(
            audio_path,
            task="translate",  # Convert to English
            fp16=torch.cuda.is_available()
        )
        os.remove(audio_path)
        return result["text"]
    except Exception as e:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")

def get_english_transcript(url: str) -> tuple[str, Optional[str]]:
    """
    Get English transcript by:
    1. Trying official captions first
    2. Falling back to Whisper transcription
    Returns (transcript_text, detected_language)
    """
    try:
        # Try official captions first
        try:
            transcript_url = extract_transcript_yt_dlp(url)
            raw_text = preprocess_transcript(transcript_url)
            return raw_text, "en"  # Assume English if we get captions
        except Exception as caption_error:
            # Fallback to Whisper
            return transcribe_with_whisper(url), None  # Language detected by Whisper
    
    except Exception as e:
        raise RuntimeError(f"Failed to get transcript: {str(e)}")
    
def summarize_youtube_transcript(text: str, format_type: str = "paragraph") -> str:
    """Clean and summarize transcript text"""
    # Remove common artifacts
    cleaned_text = re.sub(r'(\d{2}:)?\d{2}:\d{2}(\.\d+)?\s*', '', text)  # Timestamps
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)  # [Music] etc.
    cleaned_text = ' '.join(cleaned_text.split()).strip()  # Extra whitespace
    
    # Summarize in chunks
    max_chunk = 1000
    chunks = [cleaned_text[i:i+max_chunk] for i in range(0, len(cleaned_text), max_chunk)]
    
    summary_parts = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summary_parts.append(result[0]['summary_text'])
    
    full_summary = " ".join(summary_parts)
    
    if format_type == "bullet":
        sentences = [s.strip() for s in full_summary.split('. ') if s.strip() and len(s) > 10]
        return "- " + "\n- ".join(sentences)
    return full_summary

def enhanced_summarize(text, max_length=150, min_length=50, format_type="paragraph"):
    max_chunk = 1000  # Keep chunks manageable for the model
    text_chunks = []
    
    # Split text into sentences first
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Create chunks without breaking sentences
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk:
            current_chunk += sentence + ". "
        else:
            text_chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        text_chunks.append(current_chunk)
    
    summary = []
    for chunk in text_chunks:
        try:
            result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summary.append(result[0]['summary_text'])
        except Exception as e:
            print(f"Summarization error: {e}")
    
    full_summary = " ".join(summary).strip()
    
    if format_type == "bullet":
        # Split into sentences and create proper bullet points
        bullet_points = [s.strip() for s in full_summary.split('.') if s.strip()]
        # Format each point properly
        bullet_summary = "\n- " + "\n- ".join(bullet_points)
        return bullet_summary
    else:
        # For paragraphs, ensure proper spacing between sentences
        return full_summary.replace(". ", ".\n\n")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pdf')
def pdf():
    return render_template('pdf.html')

@app.route('/youtube')
def youtube():
    return render_template('youtube.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/summarize', methods=['POST'])
def handle_summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        ext = filename.split('.')[-1].lower()
        if ext == 'pdf':
            text = extract_text_from_pdf(filepath)
        elif ext == 'docx':
            text = extract_text_from_docx(filepath)
        elif ext == 'txt':
            text = extract_text_from_txt(filepath)
        elif ext in ['jpg', 'jpeg', 'png']:
            text = extract_text_from_image(filepath)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # More reliable language detection using Google Translate
        detected = translator.detect(text)
        source_lang = detected.lang
        target_lang = request.form.get('language', 'en')

        # Translate before summarizing
        if source_lang != target_lang:
            text = translator.translate(text, src=source_lang, dest=target_lang).text


        format_type = request.form.get('format', 'paragraph')
        summary = enhanced_summarize(text, format_type=format_type)


        
        # Safety check - ensure summary is in target language
        post_summary_lang = translator.detect(summary).lang
        if post_summary_lang != target_lang:
            summary = translator.translate(summary, dest=target_lang).text
            
        os.remove(filepath)
        
        return jsonify({
            'summary': summary,
            'detected_language': source_lang,
            'output_language': target_lang
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/summarize-youtube', methods=['POST'])
def handle_youtube():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'YouTube URL required'}), 400

    try:
        # Get English transcript (auto-converted if needed)
        transcript, detected_lang = get_english_transcript(data['url'])
        
        # Generate summary
        summary = summarize_youtube_transcript(transcript, data.get('format', 'paragraph'))
        
        return jsonify({
            'summary': summary,
            'detected_language': detected_lang or 'auto'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize-image', methods=['POST'])
def summarize_image():
    try:
        print("request.files keys:", request.files.keys())
        print("request.form keys:", request.form.keys())
        print("request.content_type:", request.content_type)

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        format_type = request.form.get('format', 'paragraph')
        target_lang = request.form.get('language', 'en')

        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # OCR
        text = extract_text_from_image(filepath)
        source_lang = detect(text)

        # Translate to English if needed
        if source_lang != 'en':
            text = translator.translate(text, src=source_lang, dest='en').text

        # Summarize
        summary = enhanced_summarize(text, format_type=format_type)

        # Translate to final target language if needed
        if target_lang != 'en':
            summary = translator.translate(summary, dest=target_lang).text

        os.remove(filepath)

        return jsonify({'summary': summary})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def handle_question():
    data = request.get_json()
    if not data or 'question' not in data or 'context' not in data:
        return jsonify({'error': 'Question and context required'}), 400
    
    try:
        response = qa_model({
            'question': data['question'],
            'context': data['context']
        })
        return jsonify({'answer': response['answer']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
