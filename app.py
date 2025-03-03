from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from difflib import get_close_matches

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_audio_from_video(video_file, audio_file):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

def split_audio(audio_file, chunk_length_ms):
    audio = AudioSegment.from_wav(audio_file)
    chunks = []
    overlap = 3000
    start = 0
    while start < len(audio):
        end = start + chunk_length_ms
        chunks.append(audio[start:end])
        start += chunk_length_ms - overlap
    return chunks

def transcribe_audio_chunk(chunk, recognizer, language='en-US'):
    with chunk.export(format="wav") as source:
        audio = sr.AudioFile(source)
        with audio as audio_source:
            audio_data = recognizer.record(audio_source)
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                return ""

def milliseconds_to_hms(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def search_word_in_transcript(transcript, search_words, chunk_start_time, chunk_length_ms, fuzzy_match, cutoff):
    timestamps = {}
    word_times = {}
    words = transcript.lower().split()
    search_words = [word.lower() for word in search_words]

    for i, word in enumerate(words):
        if word in search_words:
            word_time = chunk_start_time + (i / len(words)) * chunk_length_ms
            word_time_hms = milliseconds_to_hms(word_time)
            if word in timestamps:
                timestamps[word].append(word_time_hms)
            else:
                timestamps[word] = [word_time_hms]

            if word not in word_times:
                word_times[word] = []
            word_times[word].append(word_time)
        elif fuzzy_match:
            similar_words = get_close_matches(word, search_words, n=1, cutoff=cutoff)
            if similar_words:
                closest_word = similar_words[0]
                word_time = chunk_start_time + (i / len(words)) * chunk_length_ms
                word_time_hms = milliseconds_to_hms(word_time)
                if closest_word in timestamps:
                    timestamps[closest_word].append(word_time_hms)
                else:
                    timestamps[closest_word] = [word_time_hms]

                if closest_word not in word_times:
                    word_times[closest_word] = []
                word_times[closest_word].append(word_time)

    concatenated_words = []
    concatenated_times = []
    sorted_words = sorted(word_times.keys(), key=lambda w: min(word_times[w]) if word_times[w] else float('inf'))

    i = 0
    while i < len(sorted_words) - 1:
        current_word = sorted_words[i]
        current_times = word_times[current_word]
        j = i + 1
        concatenated = current_word
        min_time = min(current_times)

        while j < len(sorted_words):
            next_word = sorted_words[j]
            next_times = word_times[next_word]

            if abs(min(next_times) - min(current_times)) <= 1000: 
                concatenated += " " + next_word
                j += 1
                current_times = next_times
                min_time = min(min_time, min(current_times))
            else:
                break

        if concatenated != current_word:
            concatenated_words.append(concatenated)
            concatenated_times.append(milliseconds_to_hms(min_time))

        i = j

    return timestamps, concatenated_words, concatenated_times

def transliterate_search_words(words):
    transliterated_words = []
    for word in words:
        try:
            hindi_script = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            transliterated_words.append(hindi_script)
        except Exception as e:
            transliterated_words.append(word)
    return transliterated_words

# Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return render_template('error.html', message="No video file uploaded")
    
    video = request.files['video']
    if video.filename == '':
        return render_template('error.html', message="No video selected for uploading")

    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        language = request.form.get('language', 'en')
        search_string = request.form.get('search_string', '')
        fuzzy_match = request.form.get('fuzzy_match', 'no') == 'yes'
        cutoff = float(request.form.get('cutoff', 0.8))

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        extract_audio_from_video(video_path, audio_path)

        recognizer = sr.Recognizer()
        chunks = split_audio(audio_path, 15000)

        search_words = search_string.split()
        if language == 'hi':
            search_words = transliterate_search_words(search_words)

        timestamps = {}
        chunk_transcriptions = [] 
        for i, chunk in enumerate(chunks):
            chunk_start_time = i * (15000 - 3000)
            text = transcribe_audio_chunk(chunk, recognizer, language='hi-IN' if language == 'hi' else 'en-US')
            
            chunk_transcriptions.append({'chunk': i + 1, 'transcription': text})
            word_timestamps, concatenated_words, concatenated_times = search_word_in_transcript(
                text, search_words, chunk_start_time, 15000, fuzzy_match, cutoff
            )

            for word, times in word_timestamps.items():
                if word in timestamps:
                    timestamps[word].extend(times)
                else:
                    timestamps[word] = times

            for concatenated, time in zip(concatenated_words, concatenated_times):
                if concatenated not in timestamps:
                    timestamps[concatenated] = [time]

        # Render results on the webpage
        return render_template(
            'results.html',
            transcriptions=chunk_transcriptions,
            timestamps=timestamps,
            search_words=search_words
        )
    else:
        return render_template('error.html', message="Allowed video formats are mp4, avi, mov, mkv")
    
if __name__ == '__main__':
    app.run(debug=True)