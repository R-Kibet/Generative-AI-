from openai import OpenAI
import os
import requests
from datetime import datetime
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/generate", methods=['POST'])
def generate():
    try:
        #get the data from the front end
        data = request.get_json()
        prompt = data.get('prompt', "").strip()
        
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        # get additional parameteres
        size = data.get("size", "1024x1024")
        quality = data.get("quality", "standard")
        style = data.get('style', 'vivid')
        
        print(f"generating image for prompt: {prompt}")
        
        # generate image using DALL-E 3
        res = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1 
        )
        
        # Get the image URL
        image_url = res.data[0].url
        
        # Download and save image locally
        image_res = requests.get(image_url)
        
        if image_res.status_code == 200:
            
            # create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_image_{timestamp}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # save image 
            with open(filepath, 'wb') as f:
                f.write(image_res.content)
                
            
            # success response
            return jsonify({
                "success": True,
                "image_url": image_url,
                "local_path": f"/static/generated_images/{filename}",
                "prompt": prompt,
                "revised_prompt": res.data[0].revised_prompt if hasattr(res.data[0], "revised_prompt") else prompt
            })
        else:
            return jsonify({"eror": "failed to download generated image"}) 
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": f"Image generation failed: {str(e)}"})

#generating multiple images
@app.route("/generate_multiple", methods=['POST'])
def generate_multiple():
    try:
        data = request.get_json()
        prompt = data.get("prompt", '').strip()
        count = min(int(data.get("count", 1)), 5) # limit to 5 images
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        size = data.get("size", "1024x1024")
        quality = data.get("quality", "standard")
        
        print(f"Generating {count} images for prompt: {prompt}")
        
        images = []
        
        # Generate multiple images 
        # DALL-E 3 only supports generation n=1 so we loop
        
        for i in range (count):
            res = client.images.generate(
                model = "dall-e-3",
                prompt = prompt,
                size =  size,
                quality = quality,
                n = 1
                
            )
            
            image_url = res.data[0].url
            
            # Download and save each image
            image_res = requests.get(image_url)
            if image_res.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}_{i+1}.png"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            
            with open(filepath, "wb") as f:
                f.write(image_res.content)
                
            
            images.append({
                "image_url": image_url,
                "local_path": f"/static/generated_images/{filename}",
                "revised_prompt": res.data[0].revised_prompt if hasattr(res.data[0], 'revised_prompt') else prompt
            })
            
        return jsonify({
            "success": True,
            "images": images,
            "prompt": prompt,
            "count": len(images) 
        })
        
    except Exception as e:
        print(f"Error generating multiple images: {str(e)}")
        return jsonify({"error": f"multiple images failed: {str(e)}"})   
    
                    
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, port=8080)
                