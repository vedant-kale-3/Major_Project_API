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

# --- 3️⃣ Load or Train Model (same as your original logic)  ---

MOTIVATIONAL_CORPUS = [
    "Son, finish this **load of laundry** before I start my lecture—trust me, it’s scarier than deadlines.",
    "I made snacks. You **clean the bathroom sink**. Fair deal?",
    "Every time you delay a task, somewhere a mom sighs. Don’t make me sigh again—start that **errand run**.",
    "If you can scroll for hours, you can surely work for fifteen minutes, na? Time to **go for a walk**!",
    "Honey, **organize the pantry** now. Future-you won’t be rummaging for salt while the pasta burns.",
    "Did you **take your multivitamin**? It’s the easiest A+ you’ll get today. Go on, right now.",
    "That **sink full of dishes** is breeding small sorrows. Let’s tackle it together. Five minutes, promise!",
    "Your brain needs quiet time. **Drink a cup of herbal tea** and let the day's chaos settle.",
    "I'm giving you a time-out to **read a book for fun**. You deserve the break, but you have to take it!",
    "You can't eat takeout forever. Let's **plan healthy meals for the week**. Mama's not cooking for one!",
    "**Sweep the kitchen floor**, darling. Small chores are like stretching for the day. Necessary!",
    "That **yoga session** is your self-care prescription. Don't skip your medicine, okay?",
    "**Sort through those old clothes** for donation. Letting go of clutter is good for the soul, sweetie.",
    "Have you **drank enough water** today? Go fill that big jug right now, no arguments.",
    "Let’s **clean the bathroom mirror**. You need to see that beautiful face clearly!",
    "Your **to-do list** is short. Finish the shortest task first. It's a momentum builder, dear.",
    "I see **go to bed by 10 PM** on your list. Lights out! Self-care is non-negotiable, sweetie.",
    "That **grocery shopping** is an adventure waiting to happen. Put on some music and let's conquer the aisles.",
    "**Mop the floors** now. The clean smell will make your whole day better. Trust your mother on this one.",
    "Are you sure you **ate a balanced meal**? Show me your plate! Fuel your body properly, love.",
    "Come on, **call the doctor’s office** and **schedule that eye exam**. It’s 3 minutes of effort for 365 days of clear vision.",
    "Let's **make the bed**. A tidy bed is a tidy life. It’s the first win of the day.",
    "The most important work you'll do today is **take a few deep breaths**. Go on, right now.",
    "**Replace the air filter** in your room. Breathe fresh air, think fresh thoughts.",
    "That **gardening task** is calling you. Go get some dirt on your hands. It’s grounding, my love.",
    "I made **breakfast smoothies for the week**. All you have to do is pour it. No excuses for skipping breakfast!",
    "Your **desk is too dusty**. Wipe it down. Respect your workspace, and it will respect your work.",
    "**Walk the dog** is on your list. Fresh air for both of you! Double duty, double reward.",
    "You need to **practice gratitude journaling**. Write down three things you are thankful for. It brightens the spirit.",
    "That **broken chair repair** is a safety hazard. Let's find the screwdriver and fix it right now.",
    "**Floss your teeth**! It’s the small things, honey. Self-care protects future-you.",
    "Let's **clean out the fridge**. Throw out the old to make room for the new. It’s a life metaphor!",
    "**Call the dry cleaner**. Don't let your good clothes sit in the bag. Presentation matters, dear.",
    "Time for **20 minutes of light stretching**. Undo the knots that the day has tied in you.",
    "Your **organize the spice rack** task is waiting. A tidy rack means a tasty dinner. Go!",
    "**Take a few minutes to meditate**. Clear the cache in that busy brain of yours, darling.",
    "The **pet grooming** is waiting. Your furry friend deserves your attention, and you need a break.",
    "**Wash the windows** in the living room. Let the light in! It changes the whole energy of the house.",
    "You said you'd **read a poem**. Find a beautiful one and let it nourish your soul for a moment.",
    "That **budget check** is essential. Spend 15 minutes reviewing your expenses. Responsibility is sweet.",
    "**Schedule a dentist appointment**. The hardest part is calling. I'll dial the number for you if you promise to talk!",
    "Let's **dust the ceiling fans**. No one wants to breathe old dust. High chores, high reward!",
    "**Do some basic squats or push-ups**. Just 10 of each. A little movement is better than none.",
    "**Clean out your handbag/backpack**. Carrying unnecessary weight is tiring, emotionally and physically.",
    "You need to **sort the recycling**. Do your part for the planet, my sweet. Every piece matters.",
    "**Get the car washed**. Treat your car nicely, and it will treat you nicely on the road.",
    "Have you **put on sunscreen**? Even indoors, honey. Protect that beautiful skin!",
    "Let's **repair that small tear in your shirt**. Don't let a small flaw become permanent.",
    "Time to **light a candle** or use an essential oil diffuser. Create a pleasant atmosphere for yourself.",
    "That **weekly grocery list** is calling. Let's make it together. Efficiency in the kitchen starts now.",
    "**Take a relaxing bath**. You’ve earned it, my dear. Soak away the stress.",
    "**Organize the entryway shoes**. A cluttered entrance means cluttered energy for the house.",
    "Did you **apply hand cream**? Soft hands are happy hands. Go moisturize!",
    "The **vacuuming** can wait no longer. Put on your favorite music and dance with the machine!",
    "**Check the tire pressure** on your car. Safety is always worth the effort, my child.",
    "Let's **clean the oven**. It’s the dreaded task, but the satisfaction after is immense.",
    "You put **listen to a calming podcast**. Go for it! Feed your mind with peace.",
    "**Water the plants**. They need care, just like you do. Nurture the life around you.",
    "That **return an item to the store** errand is so close to being done. Go clear it from your mind.",
    "**Wipe down the kitchen cabinets**. Little messes become sticky messes. Prevent the stickiness!",
    "**Review your calendar for the next week**. Knowing what’s coming reduces anxiety.",
    "Time to **clean out the utensil drawer**. Everything should have its proper place.",
    "**Take a quiet moment by the window**. Observe the world. This is self-care, too.",
    "**Fix the leaky faucet**. Stop the drip! Conservation and quiet are important.",
    "The **mail sorting** needs to happen. Separate the junk from the joy immediately.",
    "**Put away the clean laundry**. Don’t let it live in the basket, my dear. It deserves a drawer.",
    "You need to **clean your makeup brushes**. Hygiene is paramount, love.",
    "Let’s **wipe down the light switches and doorknobs**. Clean the things you touch most often.",
    "That **change the sheets** chore is a necessity. Fresh bedding for fresh sleep!",
    "**Prepare your clothes for tomorrow**. Lay them out. Simple preparation makes for a smooth morning.",
    "**Find a comfortable chair** and sit with a hot drink for 10 minutes. Just sit.",
    "The **dog walking** is waiting. Go get your steps in, both of you!",
    "**Clear the browser tabs** on your computer. Mental clutter often hides in the digital space.",
    "You put **do some breathing exercises**. Three minutes. Inhale peace, exhale worry.",
    "That **pet feeding schedule check** is important. Consistency is key for their well-being.",
    "Let's **organize the utility closet**. Tidy supplies are happy supplies.",
    "**Get a haircut**. A fresh look is a fresh start, my dear.",
    "The **cable bill payment** is due. Let's pay it now so you don't have to worry later.",
    "**Listen to music that makes you happy**. Turn it up! Joy is contagious.",
    "Time to **clean the windows in the bedroom**. Wake up to a bright world.",
    "**Take a few minutes to stretch your hands and wrists**. Protect the tools that serve your mind.",
    "That **organize the first aid kit** task is crucial. Preparedness is motherly wisdom.",
    "You need to **make your lunch for tomorrow**. Don't rely on fast food, please.",
    "Let’s **find a new herbal tea blend** to try. Keep exploring little comforts.",
    "**Sweep the patio/balcony**. Cleanliness extends beyond the walls, darling.",
    "The **library book return** is pending. Clear your debts and move on!",
    "**Do a quick 5-minute tidy of the living room**. Small efforts, big impact.",
    "You put **write a letter to yourself**. A moment of self-reflection. Go do it.",
    "That **check the mail** errand is simple. Go see what the day brought you.",
    "**Wipe down the washing machine**. Even the helpers need cleaning sometimes.",
    "Time to **schedule a friend date**. Social connection is good for the heart.",
    "**Clear the digital downloads folder**. Get rid of the digital junk, sweetie.",
    "You need to **sit in the sunshine** for 10 minutes. Natural light is therapy.",
    "Let's **clean the kitchen counter**. Always finish the day with a clean counter, my love.",
    "**Review your self-care routine**. What's missing? Add one new activity.",
    "The **small recycling collection** is full. Take it out. Close that loop, my darling.",
    "**Organize your reusable shopping bags**. Preparedness is key for smooth errands.",
    "You deserve a **hot cup of cocoa/warm milk**. Make it yourself, and enjoy the moment.",
    "Final chore: **check all the locks** before bed. Safety is peace of mind."
]

# --- Hyperparameters ---
SEQUENCE_LENGTH = 5
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

DEADLINE_MISSED_SEED = ' Beta, you still haven’t done it? Come on, just finish it and make me proud! '
TASK_COMPLETED_RESPONSE = " Task complete! You deserve a big warm hug! "


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
        return f"{dialogue}"

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
