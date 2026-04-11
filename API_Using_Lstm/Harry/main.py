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
import re        # ✅ FIX 1: Added for whole-word keyword matching
import random    # ✅ FIX 4: Added for corpus fallback



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

# --- 3️⃣ Load or Train Model (same as your original logic)  Motivational_corpus---

MOTIVATIONAL_CORPUS = [
    "Finish that **client email** and you've got a perfect excuse for a coffee montage. Edit quick, reward big.",
    "Let’s render today’s work on the **monthly report** in real life before it renders in your head all night.",
    "Work is like loving someone, it's incomplete without efforts. So, start your **follow-up on that lead**!",
    "Time for 15 minutes of **stretching**. Gotta keep the gear (your body) flexible for the next shoot.",
    "That **new song on the piano** is waiting to be tracked. Get the first verse down, bro.",
    "I know the **client folder setup** is tedious. Let’s do a 'power hour' of admin, then we're free agents.",
    "Go for that **walk in nature**. It’s like hitting the reset button on your white balance. Essential for creativity.",
    "Your **creative brainstorm session** starts now. Write down 10 wild ideas. Quantity over quality for the first take!",
    "**Plan the agenda** for the team meeting. A good script makes the whole production run smoothly.",
    "You put **prepare breakfast smoothies**. That's future-you giving present-you a high-five. Make the high-five happen!",
    "Let's tackle three problems on that **budget review**. Three quick wins, and we can call it a day on finance.",
    "That **photography practice** needs a quick shoot. Grab your phone, snap one killer frame. Done.",
    "**Write that internal memo** now. Don't let a small piece of communication turn into an overdue project.",
    "**Eat a balanced meal** is on the list. Fuel the body like you fuel the camera. Quality energy only.",
    "Just 10 minutes on your **website design mock-up**. It's your online reel—keep it sharp.",
    "That **business trip booking** is admin B-roll. Get it filed and out of your mental hard drive.",
    "Time to learn that **basic photo editing**. Open the software, try one filter. Level up your skill set.",
    "**Renewing software licenses** is boring, but downtime costs money. Get that paperwork handled immediately.",
    "That **vlog concept outline** is genius. Let's map out the first 60 seconds of the video—the hook is everything.",
    "You need that **cup of herbal tea**. It's the perfect cut to a moment of peace. Don't skip the quiet scene.",
    "The most important work on your **freelance pipeline** is starting. Open the list, pick the easiest task, and crush it.",
    "**Write a poem** is self-care. Just three lines of whatever. Let the creative flow run.",
    "Let’s knock out three steps on **onboarding the new team member**. Smooth transition, happy team. Good management is good editing.",
    "**Prepare a healthy lunch** for the week. Meal prep is the ultimate life hack—future proof your hunger!",
    "You said you’d **start that blog**. Let's define the first blog post title and three key talking points. Get the headline in.",
    "The **next drawing for your portfolio** needs a simple pencil sketch. Just the rough shape. The detail can wait.",
    "That **client meeting follow-up** needs to be fast. Draft the summary email in 5 minutes. Strike while the iron is hot.",
    "You need to **schedule an eye exam**. Can’t balance life if you can’t see the frame clearly, bro!",
    "Let’s make the first five tracks of that **new playlist**. Good music is essential for deep work sessions.",
    "That **social media content plan** needs to be solid. Plan the next three posts. Pre-production is everything.",
    "**Reviewing budget for the next quarter** is the director's responsibility. Let’s look at the numbers and set the scene.",
    "**Do some light stretching** in the morning. Loosen up the hinges. You're not 16 anymore, man.",
    "Your **short story draft** needs an ending. It doesn't have to be perfect, just finished. Ship it!",
    "Time to create a template for the **project expense tracking**. Set up the system once, benefit forever.",
    "You’re supposed to **practice your dance routine**. Put the music on and drill the chorus. Get the rhythm back.",
    "Let’s tackle the **new database setup**. Once the structure is there, everything else is just data entry.",
    "The **knitting project** looks cozy! Cast on the first ten stitches. You need a hands-off hobby.",
    "That **client check-in call** needs to happen. Let’s make it a 5-minute call, clear the air, and move on.",
    "**Take a multivitamin** is the easiest win today. Don't forget your supplements, you gotta optimize the hardware!",
    "Your **short film storyboard** is your blueprint. Sketch out the next ten frames. Get the visual flow down.",
    "I know the **internal memo writing** is dull. Just get the core message across. Keep it brief, no fluff.",
    "**Plan healthy meals for the week** is the cheat code for wellness. Prep once, eat well five times.",
    "Let's focus on the first three steps of your **new journaling habit**. Just a quick log. It's metadata for your mind.",
    "You need to **organize your music files**. Clean up the asset library. Tidy files make fast edits.",
    "That **pitch deck revision** needs fresh eyes. Let's look at one slide and make the visual pop.",
    "Time to do a quick 15 minutes of **basic instrument practice**. Keep the creative muscles warm.",
    "Let’s find three options for the **company event organization**. Give the client choices, not problems.",
    "**Go to bed by 10 PM** is the ultimate power move. Maximize your rest and render time!",
    "That **personal logo design** needs five thumbnail sketches. It's all about finding the core brand identity.",
    "The **task prioritization session** is a must. Let’s arrange the list by urgency—what has the fastest deadline?",
    "**Do some meditation** right now. 5 minutes. Clear the cache and reboot the system.",
    "Your **digital art sketch** is waiting. Block out the main colors. Get the scene set.",
    "That **client contract review** is high-value work. Let’s do 30 minutes of deep focus, then a reward.",
    "I saw you put **practice calligraphy**. That's a fun one. Do two warm-up pages and call it productive.",
    "Let's look at the first three items on your **email inbox organization**. Process the high-priority stuff first.",
    "**Write three thank-you notes**. Good karma and good networking. Send out the appreciation.",
    "Let's find one killer image for your **mood board for design project**. Find the core inspiration.",
    "Time for 20 minutes of **cardio**. Run like you're trying to meet a final deadline!",
    "That **old portfolio piece polish** only needs one small tweak. Adjust the typography, make the file size smaller.",
    "Let’s tackle that **cold email campaign**. Write the first three personalized subject lines. Get the click!",
    "**Drink plenty of water** is the simplest fix for low energy. Hydrate the crew, man.",
    "That **lyric writing session** needs a hook. Listen to a great song, then write down one great line of your own.",
    "You need to **check your security updates** on your computer. Don't risk a corrupt file, or your whole life.",
    "The **sewing project** looks relaxing. Cut the fabric for one piece. It's tactile therapy.",
    "That **supplier negotiation email** is sensitive. Write the draft, walk away for 10 minutes, then review and send.",
    "**Prep the healthy snacks**. Keep the fuel handy for the long editing sessions.",
    "Your **photo album organization** is a trip down memory lane. Sort five photos into the right folder. Archive complete!",
    "Let’s spend 10 minutes looking for **video inspiration**. Fill the visual library.",
    "I see **try watercolor**. Let’s load the brush and paint one single wash of color. Simple start.",
    "That **project retrospective** needs two key takeaways. What did you learn? Final review time.",
    "Time to read one chapter of that **non-fiction book**. Intellectual protein for the mind.",
    "Your **fantasy map design** is waiting! Name the four major regions. World-building win!",
    "That **expense report submission** is a drain. Let’s knock out three receipts right now. Finish the paperwork!",
    "You need to **stretch your neck and back**. Don't let the desk life turn you into a statue!",
    "Let’s master the first **card trick**. Practical magic for networking events, bro.",
    "That **invoice follow-up** is priority one. Money first. Send the email and set a reminder for tomorrow.",
    "**Walk the dog** is on the list. Get out there. Fresh air is the best light source.",
    "I know you wanted to **try composing a short piece**. Start with eight bars. That's a solid foundation.",
    "That **new software research** is essential. Spend 15 minutes reading one review. Upgrade your toolkit.",
    "Let’s focus on the first three segments of your **podcast pitch outline**. Get the audio flow down.",
    "Your **pitch deck creation** needs a narrative. What’s the story? Outline the emotional beats.",
    "**Make a doctor's appointment** is adulting. Do the admin now, feel better later.",
    "That **creative writing prompt** is just a suggestion. Don't follow it exactly. Just start the scene.",
    "Time to clean up the **digital desktop**. Clear space, clear mind. Delete the junk files!",
    "The **online course registration** deadline is soon. Let’s register for one module. Get the booking confirmed.",
    "You need to **check your posture**. Sit up straight, man! Good frame composition matters.",
    "Let’s find the perfect title for your **photo series**. A good title sets the mood for the whole gallery!",
    "That **quick client revision** is nagging you. Do it now, send it back, and close the loop.",
    "**Practice your instrument for a recital**. Don't let your performance get stale! Run through the hardest part once.",
    "Your **tax document sorting** is future proofing. File three documents now. Reduce future stress.",
    "Time for 10 minutes of **sunlight**. Get some natural light exposure. Essential vitamin D.",
    "I see you put **work on a craft project**. Start gluing the first piece. Get into the tactile zone.",
    "That **team training session prep** needs one engaging activity. Plan a quick icebreaker. Make it fun.",
    "You need to **practice your three-minute pitch**. Record yourself on your phone. Review the footage.",
    "The most important part of your **creative task** is enjoying it. Let’s start the task you find most fun today!",
    "That **new equipment research** is fun. Find two models you like. Don't buy, just research. Smart shopper.",
    "**Cook a new healthy recipe**. Treat it like an experiment. Good food, good mood.",
    "Let’s make sure all your **devices are backed up**. Never lose your files, man. Always double archive."
]

# --- Hyperparameters ---
SEQUENCE_LENGTH = 5
EPOCHS = 100  
LATENT_DIM = 128
GENERATED_WORD_COUNT = 10 

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
model.fit(X, y, epochs=EPOCHS, verbose=1)

# --- 4️⃣ Text Generation Function ---


# NEW helper function — temperature sampling replaces np.argmax
# Place this ABOVE generate_motivational_text()
def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions - np.max(predictions))  # numerically stable
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)


def generate_motivational_text(seed_text, next_words_count=GENERATED_WORD_COUNT):
    current_text = seed_text
    for _ in range(next_words_count):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=SEQUENCE_LENGTH, padding='pre'
        )
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Was np.argmax(predicted_probs) — now uses temperature sampling
        predicted_word_index = sample_with_temperature(predicted_probs, temperature=0.8)

        # Was a manual for-loop — now uses cleaner built-in dict lookup
        output_word = tokenizer.index_word.get(predicted_word_index, "")

        if output_word:
            current_text += " " + output_word
        else:
            break

    # Repetition guard — if >40% words are duplicates, fallback to corpus
    words = current_text.split()
    if len(set(words)) < len(words) * 0.6:
        return random.choice(MOTIVATIONAL_CORPUS)

    return current_text



# --- 5️⃣ Core Task Handling Logic ---
TASK_KEYWORDS = {
    'gym': 'Fitness/Gym',
    'workout': 'Fitness/Gym',
    'exercise': 'Fitness/Gym',
    'study': 'Academics/Learning',
    'homework': 'Academics/Learning',    
    'assignment': 'Academics/Learning',  
    'math': 'Academics/Learning',
    'report': 'Work/Project',
    'work': 'Work/Project',
    'project': 'Work/Project',           
    'clean': 'Household/Chores',
}

CATEGORY_SEEDS = {
    'Fitness/Gym': 'hustle beats talent',
    'Academics/Learning': 'a wise mind finishes',   
    'Work/Project': 'finish tasks like',
    'Household/Chores': "don't watch the",
}

DEADLINE_MISSED_SEED = "let's go finish"   
TASK_COMPLETED_RESPONSE = "Mission complete — your strength echoes in the halls of Questify!"


def handle_user_input(task: str, situation: str):
    normalized_input = task.lower().strip()
    normalized_situation = situation.lower().strip()   # ✅ FIX 2: Normalize situation too

    # Task completed — no change needed
    if normalized_situation == "done":
        return TASK_COMPLETED_RESPONSE

    # Missed deadline
    if normalized_situation == "notification":
        dialogue = generate_motivational_text(DEADLINE_MISSED_SEED)
        return f"🔥 Time's up! {dialogue}"

    # ✅ FIX 2: Explicitly handle "first time" / "first_time" situation
    if normalized_situation in ["first time", "first_time", "new"]:
        for keyword, category in TASK_KEYWORDS.items():
            # ✅ FIX 1: \b...\b ensures whole-word match — "work" won't match "homework"
            if re.search(r'\b' + keyword + r'\b', normalized_input):
                seed = CATEGORY_SEEDS.get(category, 'start where you are')
                dialogue = generate_motivational_text(seed)
                return f"✨ First time? Here's your push: {dialogue}"

        # ✅ FIX 2: Generic motivational fallback for unrecognised tasks on first time
        return "Every expert was once a beginner. Take the first step — you've got this! 💪"

    # Fallback for any other situation (covers unrecognized situations too)
    for keyword, category in TASK_KEYWORDS.items():
        if re.search(r'\b' + keyword + r'\b', normalized_input):
            seed = CATEGORY_SEEDS.get(category, 'start where you are')
            return generate_motivational_text(seed)

    return "I'm not sure what that task is, but you've got this!"



# --- 6️⃣ API Endpoint ---
@app.post("/generate-dialogue/")
def generate_dialogue(req: TaskRequest):
    """
    Generates a motivational dialogue based on the given task and situation.
    """
    response_text = handle_user_input(req.Task, req.Situation)
    return {"Dialogue": response_text}




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="127.0.0.1", port=port)
