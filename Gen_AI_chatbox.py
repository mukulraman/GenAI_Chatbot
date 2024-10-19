from flask import Flask , request,jsonify,render_template
from Gen_AI import get_llm
import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the data folder
path = os.path.join(current_directory, 'data')

chatbot=get_llm(path)

# Create Flask app
app=Flask(__name__)

# Creating route for API
@app.route("/")
def home():
    return render_template("index.html")

# Route for prediction end point
@app.route('/get_answer',methods=["POST"])

def get_answer():
    question=request.form['question']
    answer=chatbot.invoke({'question':question,'chat_history':[]})
    return (jsonify(answer))

# Run the app if executed directly
if __name__ == "__main__":
    app.run(port=5001)