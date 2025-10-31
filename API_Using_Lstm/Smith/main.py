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

# --- 3️⃣ Load or Train Model (same as your original logic)  Motivational_corpus---

MOTIVATIONAL_CORPUS = [
    "Class begins with… you submitting that **assignment**. Shall we?",
    "If knowledge is power, ticking this **research** box is your superpower today.",
    "Don’t let procrastination skip attendance. Skip that thought instead and open your **term paper**!",
    "Your **quiz review** is simply practicing the knowledge you already have. Let's make that knowledge shine for 20 minutes.",
    "Remember that feeling of clarity? Let's chase it by tackling the first two problems of your **problem set** now.",
    "The hardest part of **learning a new coding language** is the first syntax. Let's master that initial line of code together.",
    "That **reading list** is a journey. You only need to take the first step. Read the introduction to Chapter 1.",
    "A quick **check-in on a friend** is an investment in your social well-being. Send that text; it’s a necessary break.",
    "Let’s start your **essay outline**. A solid structure is the foundation of a great argument.",
    "You have 30 minutes for **Spanish verbs**. Think of it as intellectual calisthenics—essential for growth!",
    "The **logic puzzle** you’re avoiding is actually a great mental workout. Let’s dedicate 10 minutes to it before your focus fades.",
    "That **annotated bibliography** requires patience, but remember, every source summary saves you time later. Let's do two now.",
    "Time to write that **email to your old professor**. Communication is key to building your network. Professionalism starts now.",
    "Let’s treat this **group project planning** like a seminar. Define the goal and roles. Clarity is half the victory.",
    "You put **watch a documentary on marine life**. Learning is never wasted. Start the video and capture one key fact.",
    "**Practice calligraphy** is excellent for focus. Let your mind reset with 15 minutes of mindful practice.",
    "The **grammar exercise** is your quality control. Sharpen your writing skills with five quick examples.",
    "**Practice speaking a new language with an app**. Consistency beats intensity. Let's get today's streak maintained.",
    "Your **statistics equations** are a challenge, not a barrier. Solve just one, and build confidence for the rest.",
    "I know **attending a networking event** is scary. Just set the goal of speaking to one new person. That's a passing grade.",
    "Let's make a **concept map for your literature review**. Visualizing your ideas makes the writing flow.",
    "That **mandatory tutorial video** is a resource. Don't let it wait. Open it and take five key notes.",
    "You put **read up on current events**. Understanding the world is critical thinking in action. Check two reliable sources now.",
    "Let's focus on **revising your thesis statement**. A strong argument is like a compass—it guides all your work.",
    "**Leave a comment on a friend's photo**. A little positivity goes a long way. Take a second to connect.",
    "This **chapter summary** is your chance to solidify the material. Let's outline the main points for the first section.",
    "Time to **practice mental math exercises**. Keep those analytical muscles agile. 5 minutes of focused thought.",
    "That **presentation slide deck** needs polish. Let's check the first five slides for clarity and impact.",
    "Your **short film analysis** can start with just one scene breakdown. Small starts lead to deep dives.",
    "**Host a family reunion** is a wonderful social task! Send the first five invitations and set the date.",
    "A well-organized **notebook** is a well-organized mind. Take 10 minutes to consolidate your notes from last week.",
    "Let's draft three different answers to common questions for your **practice interview**. Preparation reduces anxiety.",
    "That **social media boundary setting** is important for focus. Review your screen time limits now.",
    "The **video editing tutorial** is a skill investment. Open it and learn one new shortcut today.",
    "You need to **send a funny meme to a group chat**. Sometimes, the best communication is a shared laugh!",
    "Let’s **review your flashcards** out loud for 10 minutes. Auditory review reinforces memory beautifully.",
    "That **mandatory book reading** is a lesson waiting to be absorbed. Read one page. That’s all I ask for now.",
    "You put **leave a positive review for a small business**. Your feedback helps others thrive. Take two minutes and be generous.",
    "**Set up a study group agenda**. A clear plan ensures everyone gets the most out of collaborative learning.",
    "That **research paper finding** is a breakthrough. Write down your 'Eureka!' moment before it fades.",
    "Time to **draft an email to the university registrar**. Good communication resolves administrative hurdles swiftly.",
    "Let's tackle the **pre-med reading**. Break it into 15-minute segments. Focused effort is always best.",
    "**Meet a neighbor for a walk**. Community connections are part of a balanced life. Say hello and get some steps in.",
    "The **art history term definition list** is a quick memory drill. Master the first ten words.",
    "That **language exchange meeting** is waiting. Schedule it now. Immersion is the fastest teacher.",
    "You need to **write a letter to a politician**. Civic engagement is important. Draft the first paragraph outlining your concern.",
    "Let’s dedicate 15 minutes to **organizing your digital study files**. A clean drive leads to a clear mind.",
    "**Volunteer at a local non-profit** is great social work. Send that initial inquiry email right now.",
    "That **philosophy paper argument** needs to be sharp. Let's challenge the premise of your first point.",
    "Time to **practice your three-minute pitch** for your final project. Clarity and brevity are your allies.",
    "You put **call a family member you haven't spoken to in a while**. Strengthen those bonds. A simple call goes a long way.",
    "The **science fair project outline** is waiting. Define your hypothesis. That’s the most crucial step.",
    "Let’s **review your scholarship application**. Check for clarity and tone. Your future self thanks you for the precision.",
    "That **mandatory video lecture** is not going to watch itself. Start the video and take five active notes.",
    "**Send a formal inquiry to a mentor**. Asking for help is a sign of wisdom, not weakness. Draft the email now.",
    "I know the **foreign language vocabulary** seems endless. Let’s master ten new words today. Ten is manageable.",
    "Your **thesis literature review** is a major undertaking. Identify the next three papers you need to read.",
    "Time to **update your resume for the career fair**. Communication of your skills needs to be flawless.",
    "Let's look at the first three items on your **email inbox organization**. Process the high-priority messages first.",
    "That **essay topic selection** is a decision, not a paper. Choose your top two options and justify why.",
    "You need to **research effective study habits**. Spend 10 minutes learning how to learn better.",
    "**Write a heartfelt recommendation for a colleague**. Your words have weight. Give them your best effort.",
    "The **debate preparation** starts now. Outline three arguments for your side of the topic.",
    "That **reading a book for fun** is necessary for a balanced intellect. Pick up the fun book for 15 minutes.",
    "Let’s tackle the first **problem in your textbook**. Confidence comes from tackling the known before the unknown.",
    "Your **public speaking practice** is key. Record yourself reading one paragraph out loud. Review the tape.",
    "**Join an academic online forum**. Engage with your peers. Learning happens outside the classroom too.",
    "That **math derivation** needs to be understood, not just memorized. Draw a diagram explaining the concept.",
    "Time to **schedule a virtual coffee with a classmate**. Collaborative learning makes everything easier.",
    "You put **review the syllabus for the semester**. Clarity on expectations is crucial. Highlight three important dates.",
    "The **mock trial prep** starts with case research. Find one precedent that supports your side.",
    "Let’s **draft a response to a controversial article**. Use respectful, evidence-based language to communicate your views.",
    "That **learning log entry** is essential documentation of your growth. Write down three things you learned this week.",
    "You need to **organize your physical study space**. A tidy desk equals less mental clutter.",
    "**Reach out to a potential collaborator**. Good social partnerships lead to great projects. Send the initial proposal.",
    "The **environmental science field report** needs its introduction written. Define the scope of your study.",
    "Time to **study for your final exams**. Break the material into small, weekly blocks. Start the first block now.",
    "That **proposal for a campus club** is a great social endeavor. Write the mission statement. Make it compelling.",
    "Let’s master one new **shortcut key** for your academic software. Efficiency is key to productivity.",
    "Your **student government campaign speech** needs its opening line. Write a hook that captures the audience's attention.",
    "**Read about financial literacy**. Knowledge in this area is empowering. Read one section on budgeting.",
    "That **debate club research** is crucial. Find three facts to support your upcoming argument.",
    "You need to **send a holiday card to a former teacher**. Maintaining connections is a lifelong skill.",
    "The **first chapter of your memoir** is waiting. Don’t worry about perfection; just get the story out.",
    "Let’s **review the rubric** for your next paper. Understanding the criteria is the first step to an A.",
    "That **online forum post** needs a thoughtful response. Contribute a valuable insight to the discussion.",
    "**Practice your note-taking technique**. Try the Cornell method for the next 15 minutes of reading.",
    "You put **write a reflection paper**. Focus on one key learning moment from the past week and analyze it.",
    "The **student handbook review** is administrative work, but it prevents future headaches. Read one policy now.",
    "Let’s **schedule a call with an academic advisor**. Get guidance for your next course selection.",
    "That **peer feedback session** is valuable communication. Give one constructive, specific comment on a classmate's work.",
    "Your **thesis chapter draft** is waiting. Write the topic sentence for the next paragraph. Momentum matters.",
    "**Read about a historical figure**. Learning their life story can provide valuable lessons.",
    "Time to **organize your syllabus library**. Digital files need order to be useful.",
    "That **educational podcast** is a great resource. Listen to the first 10 minutes and jot down three key ideas.",
    "Let’s **send a thank-you note to a guest speaker**. Gratitude is powerful communication.",
    "**Master one new formula** for your math class. Isolation makes memorization easier.",
    "You need to **reach out to an old high school friend**. Keep those long-distance bonds strong.",
    "The **final summary of your textbook** is a good review tool. Outline the last three chapters now."
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

DEADLINE_MISSED_SEED = 'Are we gonna finish it today or make it part of history?'
TASK_COMPLETED_RESPONSE = " Yippee! Task done! I’m so proud of us! "


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
