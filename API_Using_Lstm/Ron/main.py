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
    "Kiddo, Rome wasn’t built in a day… but your **task can be done in an hour**. Go build your small city.",
    "Discipline today, stories tomorrow. **Finish that chore** and then brag about the smooth process.",
    "Even my **back pain doesn’t procrastinate**. Why should your mind? Get up and start!",
    "Hope you slept well; otherwise, your pending tasks will surely wake you up. Start with **organizing those keys**.",
    "The universe rewards efficiency. **Buy those stamps** now, and close that chapter of admin.",
    "A life well-lived is a life well-maintained. **Take the car for that oil change** before it decides to teach you a lesson.",
    "You can't achieve clarity in chaos. **Organize your email inbox**; clean your mental windshield.",
    "**Calling the internet service provider** is a necessary battle. Get it over with and return to peace.",
    "Wisdom is anticipating needs. **Get that prescription refilled** now, not when you run out at midnight.",
    "**Sorting through mail** is sorting through life's small paperwork. Do it now, or drown in paper later.",
    "A true master finds time for maintenance. **Backup your computer files**; prevention is the highest form of power.",
    "**Calling a utility company** is a test of patience. Pass the test and move on to better things.",
    "**Organizing my digital photos** is curating your own history. Start the archive process now.",
    "If you have time to worry about the task, you have time to start it. **Schedule that car service**.",
    "**Fixing the squeaky cabinet door** is a small victory that eliminates a daily annoyance. Chase the quiet.",
    "**Finding a good restaurant for dinner** is a reward. Earn it by completing a small administrative task first.",
    "**Replace the air filter** in your living space. The quality of your breath affects the quality of your thought.",
    "**Change a flat tire** is a life skill. Go conquer that small mechanical beast now.",
    "That one nagging task is an anchor. **Organize your keys and keychains** and let your mind sail free.",
    "**Reviewing insurance policies** is not fun, but it's essential wisdom. Dedicate 15 minutes to clarity.",
    "**Clearing out the junk drawer** is a symbolic act. Clear the clutter, clear the path.",
    "A clean tool is a powerful tool. **Clean your camera lenses** or your spectacles now.",
    "**Finding a notary public** is a moment of necessary bureaucracy. Get it done before the clock runs out.",
    "**Organize your toolbox**. Every tool in its place. Apply that principle to your life.",
    "**Set up two-factor authentication** on your accounts. Protect your kingdom, young one.",
    "**Call your bank** about that small issue. Ignoring small leaks leads to shipwrecks.",
    "**Shredding old documents** is releasing the past. Free yourself from that paper burden.",
    "**Organize your cloud storage**. Digital clutter is still clutter. Delete the unnecessary files.",
    "**Take out the recycling**. Small acts of discipline compound into a grand life.",
    "**Wipe down all surfaces** of your workstation. A clean environment fosters clean ideas.",
    "**Find a pet sitter for your next trip**. Pre-production is everything, in life and in film.",
    "**Refill the printer cartridge**. Don’t wait for an emergency. Prepare for the moment.",
    "**Organize the linen closet**. Fold those sheets like they hold the secrets of the universe.",
    "**Research new investment options**. Wisdom isn't just about the present, it's about the horizon.",
    "That **quick shopping errand** is a simple loop. Close the loop and feel the satisfaction of completion.",
    "**Tidy up the wires behind your desk**. The unseen chaos affects the seen productivity.",
    "**Donate those unused items**. Clinging to the past blocks the flow of new opportunity.",
    "**Check the expiration dates** in your fridge. Remove what is no longer serving you.",
    "**Clean your car interior**. Your chariot should reflect the order in your mind.",
    "**Find a tailor for that clothing repair**. Small imperfections should be corrected quickly.",
    "**Look into a new internet plan**. Optimization is a continuous process.",
    "**Organize your bookshelf**. Knowledge deserves a beautiful home. Do the curation now.",
    "The world is run by people who complete tasks. **Finish that online form**.",
    "**Reset the passwords** on two old accounts. Security is never a finished product.",
    "**Write down your next week’s priorities**. A map is useless if you don't commit to drawing it.",
    "**Clean the dust from your keyboard**. Respect the tools that serve your mind.",
    "**Schedule a plumber for that drip**. Small problems become large expenses. Act now.",
    "**Find a place to sell old electronics**. Convert clutter into currency.",
    "**Organize your physical wallet/purse**. Know exactly where your resources lie.",
    "**Update your contact list**. Maintain the integrity of your network.",
    "**Research a new recipe for dinner**. Even sustenance deserves a moment of planning.",
    "**Mend that sock**. Don't discard what can be easily repaired. Value preservation.",
    "**Sort through your spare change**. Collect the fragments and find the whole.",
    "**Review your monthly statements**. Know where your energy and money are flowing.",
    "**Create a list of long-term goals**. Define the North Star that guides your daily tasks.",
    "**Clean out the dryer vent**. Safety first, my student. Tend to the hidden dangers.",
    "**Test your smoke detector batteries**. Protect your sacred space.",
    "**Clear the browser history**. A digital cleanse is good for the soul and the machine.",
    "**Research a new hobby to explore**. A wise person always cultivates new fields.",
    "**Check the oil and fluids in your car**. Dependability is a virtue.",
    "**Organize your filing cabinet**. Paperwork is the archaeology of life. Keep it sorted.",
    "**Review your will or estate plan**. The wise prepare for all seasons of life.",
    "**Wipe down the inside of your microwave**. Cleanliness is a form of respect for your possessions.",
    "**Find a new podcast for intellectual growth**. Feed your mind with good material.",
    "**Cancel one unused subscription**. Eliminate the small, constant drains on your resources.",
    "**Organize your spice rack**. Flavor is in the order. Culinary wisdom, young one.",
    "**Replace a burned-out lightbulb**. Illuminate the darkness immediately.",
    "**Check your credit score**. Know your standing in the world's economy.",
    "**Plan the route for your next trip**. The journey is in the planning.",
    "**Refill the salt shakers and other pantry staples**. Always maintain the basics.",
    "**Find a new book to read for pleasure**. Reading is the only true form of time travel.",
    "**Write down three things you are grateful for**. Gratitude is the foundation of wisdom.",
    "**Review your warranty documents**. Know the limits of your possessions.",
    "**Take five minutes to do nothing**. Stillness is essential for finding the answers.",
    "**Clean out the gutters**. Attend to the external channels of your home.",
    "**Find a local farmers market**. Support the community that feeds you.",
    "**Test the water pressure in your home**. Efficiency in all things, even plumbing.",
    "**Sort through your collection of plastic bags**. Practical organization, practical life.",
    "**Write a note to yourself about a lesson learned**. Self-reflection is the highest study.",
    "**Research the history of a common object**. Find the wisdom in the mundane.",
    "**Dust the hard-to-reach places**. True cleanliness is in the hidden details.",
    "**Check the tire pressure on your bicycle**. Even small machines need attention.",
    "**Organize your spare buttons/sewing kit**. Preparedness is the mentor's constant lesson.",
    "**Find a new coffee bean blend**. Elevate your simple daily rituals.",
    "**Look up the meaning of a word you often misuse**. Precision in language is precision in thought.",
    "**Draft a to-do list for tomorrow**. Sleep comes easy when the next day is pre-planned.",
    "**Check the settings on your phone notifications**. Silence the unnecessary noise.",
    "**Tidy up the entrance to your home**. The gateway should be welcoming and ordered.",
    "**Research a historical event you've always misunderstood**. Correct the errors in your knowledge base.",
    "**Make a backup of your phone photos**. Cherish your memories, but protect them digitally.",
    "**Clean the windows in one room**. See the world with fresh clarity.",
    "**Read about a different culture's customs**. Expand your universal awareness.",
    "**Finalize your gift-buying list for an upcoming occasion**. Planning prevents panic.",
    "**Wipe down the kitchen cabinets**. Attend to the surfaces that collect dust and oil.",
    "**Create a digital contact card for yourself**. Make networking effortless for others.",
    "**Check the air quality forecast for tomorrow**. Plan your activities with awareness.",
    "**Organize your keys and keychains**. A final reminder: clarity starts small."
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
