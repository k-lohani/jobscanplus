import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import re
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
from sklearn.decomposition import NMF
import json
import Levenshtein

# import custom json data files
def load_json(file):
  with open(file, 'r') as f:
    data = json.load(f)
  return data

# Cleaning the text

def jd_cleaning(text):
  # lowercasing
  text = text.lower()
  # HTML Tag removal
  soup = BeautifulSoup(text, 'html.parser')
  text = soup.get_text()
  # Special Character Removal
  text = re.sub(r'[^\w\s]', '', text)
  # Whitespace and Newline Removal
  text = text.replace('\n', ' ').replace('\t', '').replace('  ', ' ').strip()

  return str(text)

def jd_preprocess(text):
  # Sentence tokenization
  sentences = sent_tokenize(text)
  # Tokenization
  words = []
  for sent in sentences:
    tokens = word_tokenize(sent)
    for t in tokens:
      words.append(t)
  # Stopword Removal
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if w not in stop_words]
  # Lemmatization
  nlp = spacy.load('en_core_web_sm')
  words = " ".join(words)
  lem_obj = nlp(words)
  lemmatized_words = [token.lemma_ for token in lem_obj]
  # Noise Removal
  # spell check

  return lemmatized_words, lem_obj, words

def keyword_tf_idf(words):
  words = [" ".join(words)]
  tfidf = TfidfVectorizer()
  tfidf_matrix = tfidf.fit_transform(words)
  feature_names = tfidf.get_feature_names_out()

  tfidf_df = pd.DataFrame({
      'feature_names' : feature_names,
      'scores' : tfidf_matrix.toarray()[0]
  })
  tfidf_df.sort_values(by = 'scores', ascending = False, inplace = True)
  tfidf_df['scores'] = tfidf_df['scores'].round(3)
  # display(tfidf_df)

  # # plotting tf_idf_features and scores
  # plt.figure(figsize=(8, 6))
  # plt.barh(y = tfidf_df.feature_names[0:30], width = tfidf_df.scores[0:30])
  # plt.title('TF-IDF Scores')
  # plt.xlabel('Scores')
  # plt.ylabel('Keywords')

  return tfidf_df, tfidf_matrix, feature_names

# Word frequency analysis
def word_freq_analysis(words):
  word_freq = Counter(words)
  common_words = word_freq.most_common(10)
  word, freq = zip(*common_words)
  common_words_dict = {'keyword':[], 'frequency':[]}
  for w in common_words:
    common_words_dict['keyword'].append(w[0])
    common_words_dict['frequency'].append(w[1])
  common_words_df = pd.DataFrame(common_words_dict)
  return common_words_df

# Named Entity Recognition
def ner(words):
  text = " ".join(words)
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(text)
  entities = {'Entity Name': [], 'Entity Label': []}
  for ent in doc.ents:
    entities["Entity Name"].append(ent.text)
    entities['Entity Label'].append(ent.label_)
  entity_df = pd.DataFrame(entities)
  return entity_df

# Topic Modelling using NMF
def topic_modeling(tfidf_matrix, feature_names):
  nmf_model = NMF(n_components=5, random_state=1)
  nmf_topics = nmf_model.fit_transform(tfidf_matrix)
  topic_dict = {'topic_idx': [], 'top_words': []}
  for topic_idx, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topic_dict['topic_idx'].append(topic_idx)
    topic_dict['top_words'].append(top_words)
  topic_df = pd.DataFrame(topic_dict)
  # print(topic_df)
  return topic_df

# skills and qualification extraction

# Function to check similarity
def lev_similar(word1, word2, threshold):
    distance = Levenshtein.distance(word1, word2)
    max_length = max(len(word1), len(word2))
    similarity = (max_length - distance) / max_length
    return similarity

def extract_skills(words, og_text):
    # Loading Data from Master JSON
    skill_data_path = 'data/skills.json'
    skills = load_json(skill_data_path)

    # Extracting Matching Skills
    matched_skills = {key: [] for key in skills}
    skills_to_display = []
    for _, (skill_cat, skill_list) in enumerate(skills.items()):
      pattern = r'\b(?:' + '|'.join(re.escape(word.lower()) for word in skill_list) + r')\b'
      matches = re.findall(pattern, og_text.lower())
      matched_skills[skill_cat] = list(set(matches))
      for matched in  list(set(matches)):
        if matched not in skills_to_display:
          skills_to_display.append(matched)

    # Calculating Position Domain Based on Skill count
    percentage_dict = {key: int(round((len(matched_skills[key])/len(skills_to_display))*100, 0)) for _, (key, val) in enumerate(matched_skills.items()) if key != "technologies" and key != 'soft'}
    max_percentage_domain = max(percentage_dict, key=lambda k: percentage_dict[k])

    max_percentage_domain = max_percentage_domain.split('_')
    max_percentage_domain_pretty = ' '.join(word.capitalize() for word in max_percentage_domain[1:])

    return skills_to_display, max_percentage_domain_pretty

def extract_qualifications_and_experience(preprocessed_text, og_text):
    # Loading Data from Master JSON
    qual_exp_data_path = 'data/qualifications_and_experiece_master.json'
    master_quals = load_json(qual_exp_data_path)

    # Extracting Matching Qualifications and Experience
    matched_quals = {key: [] for key in master_quals}
    to_display = []
    for _, (qual_cat, qual_list) in enumerate(master_quals.items()):
      pattern = r'\b(?:' + '|'.join(re.escape(word.lower()) for word in qual_list) + r')\b'
      matches = re.findall(pattern, og_text.lower())
      matched_quals[qual_cat] = list(set(matches))
      for matched in  list(set(matches)):
        if matched not in to_display:
          to_display.append(matched)

    return to_display

def initiate(text):
  # cleaning text
  clean_text = jd_cleaning(text)
  # preprocessing text
  preprocessed_text, lem_obj, words = jd_preprocess(clean_text)
  # tfidf
  tfidf_df, tfidf_matrix, feature_names = keyword_tf_idf(preprocessed_text)
  # word frequency analysis
  word_freq_scores = word_freq_analysis(preprocessed_text)
  # named entity recognition
  entity_df = ner(preprocessed_text)
  # Topic Modeling
  topic_dict = topic_modeling(tfidf_matrix, feature_names)
  # Extracting skills
  skills, domain_based_on_skill_analysis = extract_skills(preprocessed_text, og_text=text)
  # Extracting Qualifications
  quals_exp_to_display = extract_qualifications_and_experience(preprocessed_text, og_text = text)

  return tfidf_df, word_freq_scores, entity_df, topic_dict, skills, domain_based_on_skill_analysis, quals_exp_to_display




# ************************************************************Testing**********************************************************************************************************************


# text = '''Minimum 1 year of work experience - fully remote position. Freshers are also encouraged to apply.

# About us: The Future of AI is Patterned We are a stealth-mode technology startup that is revolutionizing the way AI is used. Our platform uses pattern recognition to train AI models that are more accurate, efficient, and robust than ever before.

# We are backed by top investors and we are hiring for almost everything! If you are passionate about AI and want to be a part of something big, then we want to hear from you.

# Make a positive impact on the world. Be a part of a fast-growing startup. If you are interested in learning more, please visit our website.

# We Are Looking For People Who Are

# Passionate about AI.

# Excellent problem solvers.

# Team players.

# Driven to succeed.

# Requirements

# Skills and Abilities:

# Strong knowledge of R or Python for data analysis and modeling.
# Proficiency in statistical programs such as R, SAS, MATLAB, or Python.
# Familiarity with spreadsheets (VBA) and database applications (Access, Oracle, SQL, or equivalent technology).
# Basic understanding of SQL, Javascript, XML, JSON, and HTML.
# Ability to quickly learn new methods and work under deadlines.
# Excellent teamwork and communication skills.
# Strong analytical and problem-solving abilities.
# Basic understanding of SQL, Javascript, XML, JSON, and HTML.

# Preferred

# Knowledge of actuarial concepts and life, health, and/or annuity products.
# Experience with statistical modeling techniques such as GLM, Decision Trees, Time Series, Regression, etc.
# Familiarity with Microsoft DeployR.
# Exposure to insurance risk analysis.
# Basic experience in computational finance, econometrics, statistics, and math.
# Knowledge of SQL and VBA.
# Familiarity with R or Python for predictive modeling '''


# tfidf_df, word_freq_scores, entity_df, topic_dict, skills, domain_based_on_skill_analysis, quals_exp_to_display = initiate(text = text)

# print(tfidf_df) #w
# print(word_freq_scores) #w
# print(entity_df) #w
# print(topic_dict) #w
# print(skills) #w
# print(domain_based_on_skill_analysis) #w
# print(quals_exp_to_display) #