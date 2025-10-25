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
    "Your **term paper outline** is giving me FOMO. Let's crush that skeleton structure right now before Netflix calls!",
    "We agreed on 30 minutes of **Spanish verbs**! Get that brain muscle moving; the Duolingo owl is judging us.",
    "Just open the document for your **research paper**. The first sentence is the hardest. I'll time you for a 15-minute 'vomit draft.'",
    "**Quiz review** + a high-energy playlist = boss level unlocked. Shall we start the mission?",
    "Think of this **midterm study session** as a challenge. We pass, and we get immediate weekend permission. Deal?",
    "**Researching coding languages** sounds like future-you getting a job. Let's make future-you happy today! What's the first topic?",
    "That **history reading list** isn't going to read itself. Let's do a 'power 30'—just one chapter, and then we chill.",
    "Your brain needs its veggies! 🥗 Let's tackle the hard part of that **lab report** first. Everything else is dessert.",
    "I know that **mandatory tutorial video** is long, but I'll watch it with you on mute. Accountability partners, activate!",
    "The **calculus problem set** is scary, but think of the relief once you submit it. Let's fight one problem at a time.",
    "You're so close to the 'aha!' moment on that **project proposal**. What's one keyword we can Google right now?",
    "Just 15 minutes of focus on **practice questions** now = guilt-free scrolling later. Deal?",
    "That **documentary on marine life** sounds cool, but it's on your *to-do*. Let's watch the first 10 minutes, and the momentum will take over.",
    "I'll make the coffee, you bring the brainpower for that **essay intro**. Two hours, maximum.",
    "Let's find one solid **source for your term paper**. Just one citation! That's a win for the day.",
    "The most productive you is waiting for you to finish this **presentation slide deck**. Let's make them proud.",
    "Five minutes of suffering on the **financial literacy book** now = zero future money stress. Ready to open the page?",
    "Don't let the **grammar revision** become a midnight task! Let’s tick off three points now and forget about it.",
    "Let's play 'Beat the Timer' on your **physics homework**. 45 minutes, no distractions. You got this!",
    "That **programming bug** is stressing you out. Take a 5-minute break, then let's look at the code together. Fresh eyes win.",
    "Future you is begging you to start on that **reading list**. Let’s do the first chapter—just the first one. Sound doable?",
    "Remember that feeling when you finally hit 'submit'? Let’s chase that **submission high** for the literature analysis!",
    "Your **statistics equations** are your final boss battle. We take them down, and you get XP (extra chill points) for the rest of the week.",
    "I know **researching internships** is daunting. Let's apply for ONE today. Just one click!",
    "That **online course module** is short. Let's knock it out before lunch. Lunch tastes better after a win!",
    "Let's turn your **note-taking** into a competition. Who can make the prettiest, most organized notes in the next hour?",
    "I'm setting a **20-minute focus block** for your thesis reading. No phone, no distractions. Let's sprint!",
    "Is your **study group agenda** ready? A plan is half the work. Let's finalize the topics.",
    "I saw you put **practice calligraphy** on the list. That's a fun one! Let's do two warm-up pages and call it productive.",
    "Let's **review your flashcards** out loud for 10 minutes. Verbalizing is a game-changer for memorization.",
    "The secret to your **final review** is starting early. What’s the easiest topic to review right now? Pick your win.",
    "You put **read up on current events**. Let's check three reliable news sources together. Quick and relevant!",
    "**Practice mental math exercises** is on your list. 5 minutes of brain training! It's better than scrolling TikTok.",
    "Don't let that **essay topic selection** paralyze you. Choose two options and we'll debate them for 10 minutes. Decision made!",
    "That **logic puzzle** you’re staring at? It's time to break it. You've got the smarts; just need the start.",
    "Your **thesis literature review** only needs five more summaries. Let’s do two now. Two down, three to go!",
    "This **practice instrument for a recital** is waiting. Don't let your fingers get rusty! 20 minutes of scales and a song?",
    "Time to tackle that **grammar exercise**. Think of it as a workout for your writing skills. Ready for a quick set?",
    "What if we treat the **report writing** like a conversation? Just write what you'd say out loud. Perfection later!",
    "That **art history term definition list** is a quick win. Knock out the first ten before your next class.",
    "Let's do a 'study date' for your **pre-med reading**. I'll bring the tea, you bring the focus.",
    "The **video editing tutorial** is your gateway to cool skills. Open it now. First click is the hardest!",
    "I see you put **write a term paper outline**. Let's bullet point the intro, body, and conclusion. Done in 15!",
    "That **practice interview** is essential. Let’s draft three answers to common questions. Get it out of your head!",
    "You only need to finish one section of the **group project analysis**. Let's power through that one section together.",
    "I know the **foreign language dialogue prep** is tough. Let's record ourselves speaking it, then fix the awkward bits.",
    "That **annotated bibliography** is tedious. Let’s make a game: two summaries, then a chocolate reward!",
    "The **next chapter in your textbook** is calling. Answer the call! Just read the headings and the first paragraph.",
    "Time to **organize your notes** from last week. A tidy mind starts with tidy notebooks! Let's file them electronically.",
    "Let's use the **Pomodoro Technique** for your **exam review**. 25 minutes of pure focus, then a dance break!",
    "That **new song on the piano** is waiting for your fingers! Even 15 minutes of practice counts as a win today.",
    "You put **write a poem** on the list. Just write *three lines* of whatever comes to mind. That’s all a draft is!",
    "Your brain needs a break, and your heart needs this **new journaling habit**. Let's write for five minutes about your day.",
    "Time to **plan the next drawing for your portfolio**. Pick the theme and one color palette. That’s a huge step!",
    "I see **create a new playlist** is on the list! Music is crucial. Let's make the first five songs and set the vibe.",
    "That **practice photography** task is screaming your name. Go outside and take one amazing photo of something small!",
    "That **craft project** is looking at you sadly. Let’s find the supplies right now; that’s the hardest part!",
    "Don’t let 'perfectionism' kill your **short story draft**. Just get the ending down, no matter how messy.",
    "Let’s spend 20 minutes on your **website design mock-up**. You're building your future, one pixel at a time.",
    "Your **creative brainstorm session** is scheduled. Let's just write down ten ridiculous ideas. Quantity over quality!",
    "That **paint-by-number kit** is the perfect mental break. Let's fill in ten squares; low-stress productivity!",
    "I know you want to **start that blog**. Let’s decide on the first blog post title and three talking points.",
    "The world needs to see your **vlog concept outline**! Let's map out the first 60 seconds of the video.",
    "You've been meaning to learn **basic photo editing**. Open the app now and just play with one filter. Experiment!",
    "I see **read a book for fun** is on your list. You earned it. Go read one chapter and let your brain relax.",
    "Let's find three pieces of furniture for your **interior design mood board**. Just three images! Instant progress.",
    "That **new instrument practice** is key. Focus on one tough chord for 10 minutes. Conquer it!",
    "Your **digital art sketch** is waiting. Let's lay down the base colors. No shading, just color!",
    "Let's dedicate 15 minutes to your **fiction world-building**. Invent one character's backstory.",
    "You put **write three thank-you notes**. That’s a lovely, quick, creative task. Let’s do the first one!",
    "Think of this **next drawing as a gift** to your future self. Let's do the preliminary pencil sketch now.",
    "That **new recipe trial** is a fun hobby! Let's get the ingredients measured out. Mise en place is half the battle.",
    "Your **daily doodling session** is waiting. Don't skip the small things that bring you joy! Five minutes, max.",
    "Let's clear the desk for your **scrapbooking session**. Decluttering is a creative act, too!",
    "Time to practice your **favorite song on guitar**. You know you want to! Just one perfect run-through.",
    "That **podcast pitch outline** is a genius idea. Let's jot down three segment titles.",
    "I know you want to **design that t-shirt logo**. Let's draw five tiny thumbnails of different ideas.",
    "Your **poetry anthology** is begging for a title. Let’s name it right now! A title makes it real.",
    "That **short film storyboard** is your moment. Let's sketch out the first ten frames. Director mode!",
    "You need to **organize your music files**. It’s the admin step for a creative hobby. Let's categorize five albums.",
    "Let's spend 10 minutes looking for **inspiration photos** for your painting. Filling the creative well is key.",
    "I see **practice your dance routine** is on the list. Put on the music and just drill the first 30 seconds.",
    "That **sewing project** is a great way to de-stress. Let's cut the fabric for one piece. Just the cutting!",
    "You said you wanted to **try watercolor**. Let’s load the brush and paint one single wash of color. Simple start.",
    "Let's treat your **Instagram content planning** like a game. Plan the next three posts and find the hashtags.",
    "Your **personal website portfolio** is important. Let's upload your two best projects. Show off your work!",
    "That **lyric writing session** needs some space. Listen to a great song, then write down one great line.",
    "Time to polish up that **old portfolio piece**. You only need to adjust the typography. Quick visual win!",
    "The **knitting project** looks cozy! Cast on the first ten stitches. You'll be cozy by next week!",
    "Your **photo album organization** is a trip down memory lane. Let’s sort five photos into the right folder.",
    "That **creative writing prompt** is just a suggestion. Don't worry about following it perfectly. Just start typing!",
    "Let’s watch a short **tutorial on animation basics**. Ten minutes of learning a new creative skill!",
    "You wanted to **design a personal logo**. Let’s try three different font combinations for your initials.",
    "That **recipe for homemade soap** looks cool. Let’s list the ingredients we need to buy.",
    "Remember that feeling when you **finish a piece of music**? Let's chase that by recording the final track of your current piece.",
    "Time for a **quick sketch break**. Draw the view from your window. No pressure, just observation.",
    "Your **fantasy map design** is waiting! Let's name the four major regions. World-building win!",
    "You said you'd **learn three card tricks**. Grab the deck, and let's master the first one. Magician in the making!",
    "Let’s find the perfect title for your **photo series**. A good title captures the whole vibe!",
    "The most important part of your **creative task** is enjoying it. Let’s start the task you find most fun today!"
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
