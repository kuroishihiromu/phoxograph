from flask import Flask, render_template, request,redirect, url_for
import os
import subprocess
from model import sounds
from model import caption

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def top():
    return render_template('top.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/top')
def title():
    return render_template('top.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/sound')
def sound():
    return render_template('sound.html')


@app.route('/result')
def result():
    return render_template('result.html', text=text)

@app.route('/text', methods=['POST'])
def text():
    global text
    if request.method == 'POST':
        text = request.form['text']
        print(text, flush=True)
        return render_template('result.html', text=text, filepaths=filepaths, filepaths_len=filepaths_len)


count = 0
filepaths = []
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path =os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # subprocess.run(['python', script_path, file_path])
            image = caption.input_image(file_path)
            filepaths.append(file_path)
            #filepathsの長さを変数に保存
            global filepaths_len
            filepaths_len = len(filepaths)
            caption_result = caption.predict(image)
            print(caption_result, flush=True)
            soundfile = sounds.predict(caption_result)
            global count
            count += 1
            if count >= 3:
                return render_template('sound.html', condition=True, count=count, soundfile=soundfile, caption_result=caption_result)
            else:
                return render_template('sound.html', count=count, soundfile=soundfile, caption_result=caption_result)


if __name__ == "__main__":
    app.run(debug=True)
