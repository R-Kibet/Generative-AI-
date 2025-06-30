from openai import OpenAI
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify,render_template

load_dotenv()

#initialize open ai client

client = OpenAI(api_key = os.getenv("OPEN_AI_KEY"))  # Fixed variable name

#initialize flask
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/", methods= ["GET", "POST"])
def main():
    
    if request.method == "GET":
        # Render upload form
        return render_template("index.html")
    
    if request.method  == "POST":
        lang = request.form['language', "English"]
        file = request.files["file"]
        
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        if file:
            fn = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fn )
            file.save(file_path)
            
            try:
                with open("static/name.mp3", "rb") as audio: # rb -> is a read binary file

                    translate = client.audio.translation.create(
                        model = "whisper-1",
                        file = audio
                    )
                    
                    original_text = translate.text
                    
                    # If target language is not English, translate the text
                    if lang.lower() != 'english':
                        chat_response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "system", 
                                    "content": f"You are a professional translator. Translate the following text to {lang}. Only provide the translation, no additional comments."
                                },
                                {
                                    "role": "user", 
                                    "content": original_text
                                }
                            ],
                            max_tokens=1000,
                            temperature=0.3
                        )
                        
                        final_translation = chat_response.choices[0].message.content
                    else:
                        final_translation = original_text
                        
                    # Clean up uploaded file
                    os.remove(file_path)
                    
                    return jsonify({
                        "original_text": original_text,
                        "translated_text": final_translation,
                        "target_language": lang
                    })
                    
            except Exception as e:
                # Clean up file if error occurs
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({"error": f"Translation failed: {str(e)}"}), 500
                    
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, port=8080)
                