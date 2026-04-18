# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# import concurrent.futures
# import hashlib
# import time

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-1.5-flash")

# cache = {}
# last_call_time = 0


# def get_cache_key(prompt):
#     return hashlib.md5(prompt.encode()).hexdigest()


# def safe_generate(prompt, timeout=8):
#     """Gemini call with cache + timeout"""

#     key = get_cache_key(prompt)

#     # ✅ CACHE HIT
#     if key in cache:
#         print("⚡ Cache hit")
#         return cache[key]

#     def task():
#         return model.generate_content(prompt).text.strip()

#     try:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(task)
#             result = future.result(timeout=timeout)

#             cache[key] = result
#             return result

#     except Exception as e:
#         print("Gemini error:", e)
#         return None


# def rate_limited_generate(prompt, delay=10):
#     global last_call_time

#     now = time.time()

#     if now - last_call_time < delay:
#         wait = delay - (now - last_call_time)
#         print(f"⏳ Waiting {wait:.1f}s to avoid rate limit...")
#         time.sleep(wait)

#     result = safe_generate(prompt)

#     last_call_time = time.time()

#     return result


# def generate_diagnosis(symptoms):

#     prompt = f"""
#     You are a professional medical assistant.

#     Patient symptoms: {symptoms}

#     Provide:
#     1. Most likely condition
#     2. Causes
#     3. Precautions
#     4. Should they see a doctor?

#     Keep it simple and human.
#     """

#     result = rate_limited_generate(prompt)

#     if not result:
#         return fallback_diagnosis(symptoms)

#     return result


# # 🔥 SMART FALLBACK (NO GEMINI)
# def fallback_diagnosis(symptoms):

#     if "cough" in symptoms and "high_fever" in symptoms:
#         return "This may indicate a flu or viral infection. Stay hydrated and monitor symptoms."

#     if "cough" in symptoms:
#         return "This may be a common cold. Rest and drink fluids."

#     if "high_fever" in symptoms:
#         return "This could be a viral infection. Monitor temperature and rest."

#     return "Symptoms are unclear. Please consult a doctor for proper diagnosis."






import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


class ReasoningEngine:

    # =========================
    # SAFE GEMINI CALL
    # =========================
    def safe_generate(self, prompt):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("⚠️ Gemini error:", e)
            return None

    # =========================
    # QUESTION GENERATION (HYBRID)
    # =========================
    def generate_question(self, symptoms, asked, target_symptom):

        prompt = f"""
You are a smart medical assistant.

Patient symptoms so far: {symptoms}
Already asked: {asked}

Ask a natural, human-friendly question to check for:
👉 {target_symptom}

Rules:
- Ask ONLY one question
- Do NOT repeat
- Keep it conversational
"""

        response = self.safe_generate(prompt)

        if not response:
            return f"Are you experiencing {target_symptom.replace('_', ' ')}?"

        return response

    # =========================
    # EXPLANATION (FINAL DIAGNOSIS)
    # =========================
    def explain(self, disease, symptoms, probs):

        prompt = f"""
Patient symptoms: {symptoms}

Predicted disease: {disease}
Confidence: {probs}

Explain clearly:
1. Why this disease
2. Causes
3. Precautions
4. Should user consult doctor?
"""

        response = self.safe_generate(prompt)

        if not response:
            return f"{disease} is likely based on symptoms. Please consult a doctor."

        return response
    