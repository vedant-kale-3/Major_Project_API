from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import uvicorn
import os


# --- 1️⃣ Initialize FastAPI App ---
app = FastAPI(
    title="Motivational Dialogue Generator API",
    description="Generates motivational dialogue based on a user's task and situation.",
    version="1.0.0"
)

# --- 2️⃣ Define Input Model ---
class TaskRequest(BaseModel):
    Task: str
    Situation: str  # "first_time", "notification", etc.

# --- 3️⃣ Load or Train Model (same as your original logic) ---

MOTIVATIONAL_CORPUS = [
    "stop scrolling start doing",
    "a wise mind finishes tasks before excuses",
    "you can’t brag about tasks unfinished",
    "small wins today make epic stories tomorrow",
    "naps feel better after work is done",
    "you handle tasks i’ll handle the wisdom",
    "even heroes need discipline champ",
    "mess today stress tomorrow—your call",
    "work fast rest longer",
    "a half-done task is still undone",
    "delay is the thief of peace kid",
    "one push now a leap tomorrow",
    "finish tasks like a pro",
    "you deserve celebration not excuses",
    "that task is glaring at you right now",
    "tasks first netflix later",
    "you can’t coach excuses only effort",
    "finish it now drama is for later",
    "if laziness burned calories you’d be shredded",
    "even my grandma works faster and she naps twice a day",
]

# --- Hyperparameters ---
SEQUENCE_LENGTH = 3
EPOCHS = 20  # reduced for API startup
LATENT_DIM = 64
GENERATED_WORD_COUNT = 6

tokenizer = Tokenizer()
tokenizer.fit_on_texts(MOTIVATIONAL_CORPUS)
total_words = len(tokenizer.word_index) + 1

# Prepare data
input_sequences = []
for line in MOTIVATIONAL_CORPUS:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(SEQUENCE_LENGTH, len(token_list)):
        n_gram_sequence = token_list[i-SEQUENCE_LENGTH:i+1]
        input_sequences.append(n_gram_sequence)

input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]
y_int = input_sequences[:, -1]
y = to_categorical(y_int, num_classes=total_words)

# Train model once (you can load saved weights later)
model = Sequential()
model.add(Embedding(total_words, LATENT_DIM, input_length=SEQUENCE_LENGTH))
model.add(LSTM(LATENT_DIM))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=EPOCHS, verbose=0)

# --- 4️⃣ Text Generation Function ---
def generate_motivational_text(seed_text, next_words_count=GENERATED_WORD_COUNT):
    current_text = seed_text
    for _ in range(next_words_count):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=SEQUENCE_LENGTH, padding='pre'
        )
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_word_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        if output_word:
            current_text += " " + output_word
        else:
            break

    return current_text


# --- 5️⃣ Core Task Handling Logic ---
TASK_KEYWORDS = {
    'gym': 'Fitness/Gym',
    'workout': 'Fitness/Gym',
    'exercise': 'Fitness/Gym',
    'study': 'Academics/Learning',
    'math': 'Academics/Learning',
    'report': 'Work/Project',
    'work': 'Work/Project',
    'clean': 'Household/Chores',
}

CATEGORY_SEEDS = {
    'Fitness/Gym': 'hustle beats talent',
    'Academics/Learning': 'a wise mind finishes',
    'Work/Project': 'finish tasks like',
    'Household/Chores': "don't watch the",
}

DEADLINE_MISSED_SEED = 'procrastination is a luxury'
TASK_COMPLETED_RESPONSE = "🎉 HURRAY! I knew you'd do it! A done task is sweeter than dessert."


def handle_user_input(task: str, situation: str):
    normalized_input = task.lower().strip()

    # Task completed
    if situation.lower() == "done":
        return TASK_COMPLETED_RESPONSE

    # Missed deadline
    if situation.lower() == "notification":
        dialogue = generate_motivational_text(DEADLINE_MISSED_SEED)
        return f"🔥 Time’s up! {dialogue}"

    # First-time task (motivate)
    detected_category = None
    for keyword, category in TASK_KEYWORDS.items():
        if keyword in normalized_input:
            detected_category = category
            break

    if detected_category:
        seed = CATEGORY_SEEDS.get(detected_category, 'start where you')
        dialogue = generate_motivational_text(seed)
        return f"🤖 LET’S GO! {dialogue}"

    return "I’m not sure what that task is, but you’ve got this!"


# --- 6️⃣ API Endpoint ---
@app.post("/generate-dialogue/")
def generate_dialogue(req: TaskRequest):
    """
    Generates a motivational dialogue based on the given task and situation.
    """
    response_text = handle_user_input(req.Task, req.Situation)
    return {"Dialogue": response_text}




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
