from flask import Flask, render_template, request
from chat import chat

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return str(chat(user_text))

if __name__ == "__main__":
    app.run()
