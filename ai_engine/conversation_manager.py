from ai_engine.question_strategy import get_next_question, fallback_question
# from ai_engine.reasoning_engine import generate_diagnosis
from utils.bert_nlp import extract_symptoms
from ai_engine.reasoning_engine import ReasoningEngine

engine = ReasoningEngine()


class ConversationManager:

    def __init__(self, dqn_model, encoder, graph, classifier, state=None):

        self.dqn_model = dqn_model
        self.encoder = encoder
        self.graph = graph
        self.classifier = classifier

        if state:
            self.symptoms = state.get("symptoms", [])
            self.asked_questions = state.get("asked_questions", [])
        else:
            self.symptoms = []
            self.asked_questions = []

        self.max_q = 10

    def get_state(self):
        return {
            "symptoms": self.symptoms,
            "asked_questions": self.asked_questions
        }

    def start(self):
        return "Hi 👋 Tell me what symptoms you're experiencing."

    def should_stop(self):
        return (
            len(self.asked_questions) >= self.max_q or
            len(self.symptoms) >= 4
        )

    def update(self, user_input):

        # 🧠 Extract symptoms
        new_symptoms = extract_symptoms(
            user_input,
            self.encoder.get_all_symptoms()
        )

        if not new_symptoms:
            return "I didn't fully understand. Could you describe your symptoms more clearly?"

        self.symptoms = list(set(self.symptoms + new_symptoms))

        # =========================
        # 🧠 STOP → DIAGNOSIS
        # =========================
        if self.should_stop():

            state_vec = self.encoder.encode(self.symptoms)

            top_preds, probs = self.classifier.predict_top_k(state_vec, 3)

            top_disease = top_preds[0]
            confidence = round(probs[0], 2)

            explanation = engine.explain(
                top_disease,
                self.symptoms,
                confidence
            )

            return {
                "type": "result",
                "data": explanation
            }


        # =========================
        # ⚡ RL decides WHAT to ask
        # =========================
        rl_question = get_next_question(
            self.symptoms,
            self.dqn_model,
            self.encoder,
            self.graph
        )

        # 🔥 extract symptom from RL question
        symptom = (
            rl_question
            .replace("Do you have ", "")
            .replace("Are you experiencing ", "")
            .replace("Have you noticed any ", "")
            .replace("Any signs of ", "")
            .replace("Just checking — do you feel ", "")
            .replace("?", "")
            .strip()
        )


        # =========================
        # 🤖 Gemini improves HOW to ask
        # =========================
        gemini_question = engine.generate_question(
            self.symptoms,
            self.asked_questions,
            symptom
        )

        # fallback if Gemini fails
        question = gemini_question if gemini_question else rl_question

        # prevent repetition
        if question in self.asked_questions:
            question = fallback_question(self.symptoms)

        self.asked_questions.append(question)

        print("Symptoms:", self.symptoms)
        print("Next question:", question)

        return question
    