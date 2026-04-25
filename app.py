# app.py

from flask import Flask, request, jsonify, session
from ai_engine.conversation_manager import ConversationManager
from utils.preprocessing import SymptomEncoder
import torch
from utils.symptom_graph import SymptomGraph

import sys
import os

sys.path.append(os.path.abspath("models"))

import joblib
from models.classifier import DiseaseClassifier


clf = joblib.load("saved_models/classifier.pkl")


graph = SymptomGraph("data/Training.csv")

app = Flask(__name__)
app.secret_key = "super_secret_key"


# 🧠 Load once (global)
encoder = SymptomEncoder("data/Training.csv")

# Load trained DQN
class DQNWrapper:
    def __init__(self):
        from models.dqn import DQN
        self.model = DQN(132, 132)  # adjust size
        self.model.load_state_dict(torch.load("saved_models/dqn_model.pth"))
        self.model.eval()

    def __call__(self, state):
        return self.model(state)


dqn_model = DQNWrapper()


# 🔥 IMPORTANT: initialize question bank
from ai_engine import question_strategy
question_strategy.QUESTION_BANK = encoder.get_all_symptoms()

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/start", methods=["GET"])
# def start():

#     cm = ConversationManager(dqn_model, encoder)
#     question = cm.start()

#     session["chat_state"] = cm.get_state()

#     return jsonify({"response": question})


# @app.route("/chat", methods=["POST"])
# def chat():

#     user_input = request.json["message"]

#     # 🔥 Restore state
#     state = session.get("chat_state")

#     cm = ConversationManager(dqn_model, encoder, state)

#     response = cm.update(user_input)

#     # 🔥 Save back
#     session["chat_state"] = cm.get_state()

#     return jsonify({"response": response})

@app.route("/start")
def start():
    cm = ConversationManager(dqn_model, encoder, graph, clf)
    session["chat_state"] = cm.get_state()
    return jsonify({"response": cm.start()})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json["message"]
        state = session.get("chat_state")
        cm = ConversationManager(dqn_model, encoder, graph, state)
        response = cm.update(user_input)
        session["chat_state"] = cm.get_state()
        return jsonify({"response": response})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"response": "Something went wrong. Please try again."})
    

@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    return jsonify({"message": "Session reset"})


if __name__ == "__main__":
    app.run(debug=True)
