import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt_tab")

responses = [
    "I feel very mental pressure in study.",
    "Overthinking is not the way to lessening one's mental workload drink water and sleep 8 hours.",
    "The academic pressure is too much for me.",
    "I feel very mental pressure in study.",
    "Overthinking is not the way to lessening one's mental workload drink water and sleep 8 hours ",
    "The academic pressure is too much for me.",
    "Making student friendly syllabus could be helpful.",
    "Course lectures and schedules are too hard for me",
    "Courses are too much.",
    "Course workload affect me alot",
    "Course work could be reduce and give more time for assignements",
    "Courseworks are tough",
    "Coruse pressure too much",
    "Lectures are hard to understand",
    "Reducing the course load and amount of assignements would be good for the students",
    "Reduce the pressure of assignements",
    "Too much course",
    "Lots of assignments",
    "Course are lot's of work",
    "A lot of pressure",
    "Pressures are high",
    "It would be good if they give more time for the assignments",
    "Workloads are quite much",
    "Need to study much course load should be reduced",
    "Pressure should be reduced",
    "Course lectures and contents are tough and not easy. Need some easy content.",
    "Assignments are too much",
    "Workload could be reduced",
    "Assignments deadline should be flexible",
    "Less project work should be given ",
    "There should be more online class to reduce the physical stress.",
    "Work pressure of the courses are too much. If the syllabus could be reduce a little bit then it would be good.",
    "Easy going with student opinion regarding the lectures could reduce the workload",
    "The workload often leaves me alot of mental dissatisfaction",
    "The amount of coursework I have to manage often feels lots of work. And for this I cannot concentrate in other work",
    "Balancing multiple assignments and exams creates a lot of mental pressure.",
    "The fast pace of lectures leaves me mentally exhausted. Then I need to study those again and again by myself.",
    "Complex topics require intense concentration for me",
    "Trying to meet all the deadlines is stressful.",
    "Understanding complex topics takes a lot of mental stress.",
    "Peer projects are stress. Most of the team member does not want to work therefore I need to do all the project but they got pass easily in most cases.",
    "Studying for multiple subjects at once is hard on my mind.",
    "Late-night studying leaves me really tired. If the course load reduce then I don't need to do late night study.",
    "Trying to get good grades all the time is stressful. And it takes alot of effort to get good grades.",
    "Frequent assignments are hard.",
    "In group work some of the group members doesn't want to work.",
    "Solving difficult problems is very hard. Sometimes the assignments are hard.",
    "Having multiple exams close together is stressfull",
    "Constantly learning and remembering new things is alot.",
    "There’s too much reading, and it’s mentally exhausting.",
    "Balancing school with extracurricular activities is stressful.",
    "Difficult concepts takes a lot of mental effort.",
    "Long study sessions is boaring and also I don't understand",
    "Keeping up with online classes takes a lot of mental focus.",
    "Lots of quizzes, I feel mental pressure each time on quizes.",
    "Managing my time well is a big mental challenge. I need to work and also study, it's problem for me.",
    "I don’t have much time to relax because of all the work.",
    "Learning new technologies feels like lots of stress.",
    "Participating in the group task reduce the workload sometimes. But individual works are hard.",
    "I feel lot's of stress because lots of subjects. It annoys me alot.",
    "All of the assignments takes a lot of mental effort.",
    "Most of the course work I try to do earlier but when there are lots of assignments of every course then I feel stress.",
    "I feel tired from always having assignments, quizes and all.",
    "Participating in class discussions is mentally stressful to me.",
    "Balancing all my university responsibilities is tough.",
    "Staying organized takes a lot of mental stress. Sometimes I mix up with all of my courses.",
    "It feels like the work never ends, which is exhausting.",
    "Staying motivated is a big mental challenge with lots of task.",
    "I don’t have much time to rest because I have lots of quizes",
    "The pressure to get high grades is stressful. Then I need to focus more to the course.",
    "Understanding difficult theories is very hard. In that case I feel stress that I will not be able to do that.",
    "It’s hard to stay focused all the time with all the course.",
    "My course lectures are very tough. I cannot remember all these which gives me stress. If it could be reduce then we will have less stress.",
    "There are lots of study pressure which gives me mental stress. I cannot even enjoy my life for this workload.",
    "Working pressure and study both have lots of stress. The pressure of the workload should be reduced",
    "There’s just too much coursework to handle by the deadlines.",
    "There’s no time to relax because of all the coursework load",
    "Exercises are alot. Less exercise in each subject could be reduce.",
    "There are lots of hard topic may be those should be prepared easy slide so that we can understand fast.",
    "Learning and remembering everything is tough. Course load and less assignment will be helpfull.",
    "Keeping up with all the assignment and deadlines are stressfull. ",
    "In the logn class I feel very tired. These long classes and session should be reduced a little bit.",
    "Doing all the assignments of all the course and meet the deadline, in this situation I feel burden and sometimes I got chest pain because of all these pressure",
    "For alot of assignment,I make mistakes in some of my assignments which also impact in grades.",
    "I get tired from always having to adapt to new challenges.",
    "For lots of classes I feel very demotivated and tired.",
    "I try to be productive in all the courses. And that gives me lots of stress",
    "Staying engaged in every class is tough and also stressfull for me.",
    "The pressure to get high grades is mentally stressing for me.",
    "There’s no time to rest for all the university task and also work task.",
    "Sometimes the programming courses are very hard. ",
    "It’s hard to keep up with my studies in fast speed.",
    "I get mentally exhausted from always quizes assignments class task.",
    "The field is always changing. Learn all the technologies gives me lots of stress",
    "For the course pressure I don't have time for relaxation",
    "The pressure to succeed in every course gives stress",
    "Meeting expectations takes a lot of mental effort in every course",
    "Staying focused and productive is very stress full for me in all the classes.",
    "The workload is exhausting, especially during exams.",
    "There are lot's of classes which needs physical attendence. It could be better if those class could be online.",
    "Its hard to focus all the time in all the courses.",
    "The use of AI is sometimes stress full. I try to learn but eventually it give me wrong data",
    "The course contents are very tough the slides could be easier.",
    "Classes move too fast sometimes I don't understand.",
    "I loose concentration in longer classes.",
    "Switching between subjects is very challenging",
    "Group projects are stressful. Becuase sometimes I cannot find group members.",
    "Sometimes I need to do late night study for the courses and for the pressure I cannot even sleep.",
    "Problem solving task are hard.",
    "Getting good grades is very tough and sometimes its very stressfull for me.",
    "For lots of assignments I make alot of mistakes in my assignments",
    "Multiple exam in multiple course side by side is lots of stressfull.",
    "I always forget what read to much slides for reading",
    "it's always best if teacher care more about student opinion",
    "freindly teacher could give us better teaching",
    "The complex topic sometimes I don't understand I need focus on that I use AI but still those are hard to understand. These should be simplified.",
    "The pressure in work,life and study are a lot. Sometimes I feel mental disorder.",
    "I want to see every teacher notice how student can manage all the assignment in one day",
    "modern syllebus and teaching method needs to be applied",
    "High expectation in every course is a stress. I couldn't focus on every course.",
    "Paying attention in all the courses is alot. ",
    "competition in study is very harmfull",
    "teachers need to be more friendly for slow learners.",
    "Online classes are better than the physical one.",
    "Constantly learning all the time is feels pressure. the load of the courses are too much",
    "several exam in a row is frustrating",
    "trying to get good grade is bad practice in education system",
    "Creating competition in the class is a bad practice which are the cause of mental pressure among students",
    "Sometimes I don't understand anything in the class and after that I need to study those agains. It should be simplified",
    "Self study is a stress, so class should cover all the slides and reading contents.",
    "Theoretical exams are too hard. I cannot remember theory. It will be helpfull if theoritical questions are removed. ",
    "Classes are too long. 1.5 hours is a good length,",
    "Online classes are boaring.",
    "Because of work I spend less time in the study but I always try to attend all the classes. Classes should cover all the lectures and slides so that we don't need to study at home much.",
    "I don’t have much time to enjoy my life because of the study pressure",
    "more time taking exercise is frastrating",
    "Need longer vacation in education",
    "Compete with others is stressfull.",
    "physical attendence is stressfull",
    "Course material should more practical rather than just theoritical",
    "if teacher give more motivation in class then we may be found study more comfortable.",
    "There is so much study. Less course work would be better",
    "Class discussions are fearfull. ",
    "assignments are too much sometimes",
    "theoritical course are much harder, they needs to be more practical",
    "Study slides are hard.",
    "Concentrating in every class is tough. Less longer class is good.",
    "Doing all the assignments and following each deadlines are hard.",
    "Classes are boaring and teaching styles are boaring.",
    "Sometimes the deadlines should be flexible",
    "Morning class physical attendance is very stressfull",
    "New technologies are hard to understand",
    "I am a slow learner. Sometimes the teacher taught too fast.",
    "Programming languages are hard to understand those class should be a little slow so that we can understand better.",
]


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_responses = [preprocess_text(response) for response in responses]



from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    elif sentiment_scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"
sentiments = [analyze_sentiment(response) for response in preprocessed_responses]
    
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    elif sentiment_scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"


sentiments = [analyze_sentiment(response) for response in preprocessed_responses]

sentiment_counts = Counter(sentiments)

print(sentiment_counts)

palette = {"positive": "#77DD77", "negative": "#FF6961", "neutral": "#89CFF0"}

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(
    x=list(sentiment_counts.keys()),
    y=list(sentiment_counts.values()),
    palette=palette,
    hue=list(sentiment_counts.keys()),
    dodge=False,
)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()



import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    elif sentiment_scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"


sentiments = [analyze_sentiment(response) for response in preprocessed_responses]

positive_words = " ".join(
    [
        response
        for response, sentiment in zip(preprocessed_responses, sentiments)
        if sentiment == "positive"
    ]
)
negative_words = " ".join(
    [
        response
        for response, sentiment in zip(preprocessed_responses, sentiments)
        if sentiment == "negative"
    ]
)
neutral_words = " ".join(
    [
        response
        for response, sentiment in zip(preprocessed_responses, sentiments)
        if sentiment == "neutral"
    ]
)


def generate_wordcloud(text, color, title):
    wordcloud = WordCloud(
        background_color="white",
        colormap=color,
        width=800,
        height=400,
    ).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.show()

generate_wordcloud(positive_words, "Greens", "Positive Sentiment Word Cloud")
generate_wordcloud(negative_words, "Reds", "Negative Sentiment Word Cloud")
generate_wordcloud(neutral_words, "Blues", "Neutral Sentiment Word Cloud")