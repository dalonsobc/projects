import pandas as pd
import ast
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from math import pi
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords

# sns.set_theme(font_scale=1.4) 

# STATIC DATA

skills = [
    'python', 'r', 'sql', 'machine learning', 'artificial intelligence', 'ai', 'deep learning', 
    'statistics', 'data visualization', 'tableau', 'powerbi', 'excel', 'spark', 'hadoop', 
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'nlp', 'natural language processing',
    'computer vision', 'big data', 'data mining', 'data analysis', 'data engineering', 'etl', 
    'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'pandas', 'matplotlib', 'seaborn', 
    'pyspark', 'emr', 'ehr', 'regression', 'classification', 'clustering', 'neural networks',
    'data manipulation', 'modelling', 'statistical analysis', 'quantitative analysis',
    'research', 'communication', 'modeling', 'model', 'ml', 'business', 'llm', 'consult',
    'problem-solving', 'data science', 'visualization', 'analytics', 'data collection',
    'analyzing', 'models', 'data cleaning', 'large language models', 'llms', 'programming',
    'development', 'statistical', 'optimization', 'communicate', 'consulting', 'machine-learning',
    'project management', 'database', 'architecture', 'data governance', 'databases',
    'data quality', 'journals', 'data warehouses', 'data warehouse', 'data analyst',
    'data security', 'interpersonal skills', 'industry', 'analytical skills', 'problem solving',
    'natural language', 'datasets', 'data processing', 'consultant', 'communicates', 'marketing'
]

skill_topics = {
    'Machine Learning & AI': ['machine learning', 'artificial intelligence', 'ai', 'deep learning', 
                              'neural networks', 'ml', 'machine-learning', 'regression', 
                              'classification', 'clustering', 'scikit-learn', 'tensorflow', 
                              'pytorch', 'keras', 'nlp', 'natural language processing', 'natural language',
                              'llm', 'large language models', 'llms'],
    'Statistics & Modeling': ['statistics', 'statistical analysis', 'quantitative analysis', 
                                    'statistical', 'modelling', 'modeling', 'model', 'models'],
    'Data Visualization': ['data visualization', 'tableau', 'r', 'powerbi', 'excel', 'matplotlib', 
                           'seaborn', 'visualization', 'python'],
    'Other Technologies': ['spark', 'hadoop', 'big data', 'pyspark', 'data mining', 'aws', 'azure', 'gcp',
                              'docker', 'kubernetes', 'git', 'computer vision'],
    'Data Science & Analytics': ['data science', 'data scientist', 'data analyst', 'data analysis', 'analytics',
                                 'analyzing', 'analytical skills'],
    'Research & Academic': ['research', 'journals'],
    'Business & Consulting': ['business', 'consult', 'consulting', 'consultant', 'marketing'],
    'Soft Skills': ['communication', 'problem-solving', 'communicate', 'communicates', 
                    'interpersonal skills', 'project management'],
    'Specific Industry': ['industry', 'emr', 'ehr'],
    'Data Skills': ['datasets', 'data manipulation', 'data processing', 'optimization', 'data engineering',
                    'etl', 'data manipulation', 'data collection', 'data cleaning', 'data processing',
                    'data quality', 'data governance', 'data security', 'database', 'databases',
                    'data warehouses', 'data warehouse', 'sql'],
    'Software Development': ['development', 'programming', 'architecture']
}

provinces = {
    'British Columbia': ['bc'],
    'Quebec': ['quebec province', 'qc'],
    'Manitoba': ['mb'],
    'Ontario': ['on', 'ontario', 'remote in ontario'],
    'Remote': ['remote', 'canada'],
    'Saskatchewan': ['sk'],
    'Alberta': ['ab']
}

# FUNCTIONS

def extract_skills(text):
    if pd.isna(text):
        return []
    text = text.lower()
    found_skills = []
    for skill in skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.append(skill)
    return found_skills

def split_location(location):
    if pd.isna(location):
        return pd.Series({'City': None, 'Province': None})
    
    parts = location.split(',')
    if len(parts) >= 2:
        return pd.Series({'City': parts[0].strip(), 'Province': parts[1].strip()})
    else:
        return pd.Series({'City': parts[0].strip(), 'Province': parts[0].strip()})

def standardize_province(province):
    if pd.isna(province):
        return None
    province = province.lower()
    for variant, standard in province_mapping.items():
        if variant in province:
            return standard
    return province.title()  

def combine_skills(skills_list):
    return list(set([skill for sublist in skills_list for skill in sublist]))

def categorize_skill(skill):
    for category, skills in skill_topics.items():
        if skill.lower() in skills:
            return category
    return 'Other'

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_radar_chart(location, data):
    categories = data.columns
    values = data.loc[location].values
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=18)
    ax.set_yticklabels([])
    ax.set_title(f'Skill Demand in {location}', size=20, color='blue', y=1.1)
    plt.show(block=False)

# DATA PRE-PROCESS

# Read the data
df_raw = pd.read_csv("bigdata_scraped.csv")
df_raw['Requirements'] = df_raw['Requirements'].str.replace('\n', '', regex=False).str.strip()
df_raw['Requirements'] = df_raw['Requirements'].apply(ast.literal_eval)

# Explode the 'Requirements' column
df_reqs = df_raw.explode('Requirements')
df_reqs['Requirements'] = df_reqs['Requirements'].str.replace('\n', '', regex=False).str.strip()
df_to_clean = df_reqs[['Job Title', 'Company', 'Location', 'Requirements']]

# Extract the skills from the 'Requirements' column
df_to_clean['ExtractedSkills'] = df_to_clean['Requirements'].apply(extract_skills)

# Split location into City and Province
location_index = df_to_clean.columns.get_loc('Location')
new_columns = df_to_clean['Location'].apply(split_location)
df_to_clean = df_to_clean.drop('Location', axis=1)
df_to_clean.insert(location_index, 'Province', new_columns['Province'])
df_to_clean.insert(location_index, 'City', new_columns['City'])

# Standarize the Province names
province_mapping = {}
for standard, variants in provinces.items():
    for variant in variants:
        province_mapping[variant.lower()] = standard
df_to_clean['Province'] = df_to_clean['Province'].apply(standardize_province)

# Filter requirements with no data science skills (noise)
df_with_skills = df_to_clean[df_to_clean['ExtractedSkills'].apply(len) > 0]
# df_no_skills = df_to_clean[df_to_clean['ExtractedSkills'].apply(len) == 0] # To calibrate the filter

# Group the skills in one record per job
grouped = df_with_skills.groupby(['Job Title', 'Company', 'City', 'Province'])
df_by_job = grouped.agg({
    'Requirements': 'first',  # Keep the first requirement
    'ExtractedSkills': combine_skills  # Combine skills without repetition
}).reset_index()

# Add another DF with one skill per record 
df_by_skill = df_by_job.explode('ExtractedSkills')
df_by_skill['SkillCategory'] = df_by_skill['ExtractedSkills'].apply(categorize_skill)

# Calculate the number of skills for each job
df_by_job['SkillCount'] = df_by_job['ExtractedSkills'].apply(len)
# Sort by number of skills, descending
df_by_job = df_by_job.sort_values('SkillCount', ascending=False)

# ------------------------- EXPLORATORY DATA ANALYSIS -------------------------

# 1. WORD CLOUD

all_requirements = ' '.join(df_raw['Requirements'].fillna('').astype(str))
all_requirements = clean_text(all_requirements)
stop_words = set(stopwords.words('english'))
words = all_requirements.split()
all_requirements = ' '.join([word for word in words if word not in stop_words])
# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_requirements)
# Display the generated image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Job Requirements', fontsize=16)
plt.tight_layout(pad=0)
plt.show(block=False)

# 2. COUNT OF SKILL CATEGORY

plt.figure(figsize=(12, 6))
skill_counts = df_by_skill['SkillCategory'].value_counts()
sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='viridis', orient='h')
# skill_counts.plot(kind='bar')
plt.title('Count of Skill Category')
plt.xlabel('Count')
plt.ylabel('Skill Category')
# plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show(block=False)

# 3. ZOOM INTO "MACHINE LEARNING & IA" SKILL CATEGORY

ml_ai_skills = skill_topics['Machine Learning & AI']
skill_counts = df_by_skill[df_by_skill['ExtractedSkills'].isin(ml_ai_skills)]['ExtractedSkills'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='rocket', orient='h')
plt.title('Most Common Machine Learning & AI Skills', fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Skill', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# for i, v in enumerate(skill_counts.values):
#     plt.text(i, v, str(v), ha='center', va='center')
plt.tight_layout()
plt.show(block=False)

# 4. HISTOGRAM OF SKILLS NUMBER PER JOB
sns.set_theme(font_scale=1.5)
plt.figure(figsize=(10, 6))
sns.histplot(df_by_job['SkillCount'], kde=True)
plt.title('Number of Skills per Job Distribution')
plt.xlabel('Number of Skills')
plt.ylabel('Frequency')
plt.show(block=False)

# 5. STATISTICAL SUMMARY

print(df_by_job['SkillCount'].describe())

# 6. MOST COMMON SKILLS COMBINATION

from itertools import combinations
skill_combinations = df_by_job['ExtractedSkills'].apply(lambda x: list(combinations(x, 2)))
combination_counts = pd.Series([comb for combs in skill_combinations for comb in combs]).value_counts()
print("\nTop 10 most common skill pairs:")
print(combination_counts.head(10))

# 7. COUNT OF JOBS PER PROVINCE

province_job_counts = df_by_job['Province'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=province_job_counts.index, y=province_job_counts.values, palette='viridis')
plt.title('Number of Big Data Analytics Jobs by Province')
# plt.xlabel('Province', fontsize=12)
plt.ylabel('Number of Jobs')
# plt.xticks(rotation=45, ha='right')
for i, v in enumerate(province_job_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show(block=False)

# 8. RADAR CHART TOP 5 SKILLS CATEGORIES IN EACH PROVINCE

overall_top_categories = df_by_skill['SkillCategory'].value_counts().head(5).index.tolist()
filtered_df = df_by_skill[df_by_skill['SkillCategory'].isin(overall_top_categories)]
location_skill_counts = filtered_df.groupby(['Province', 'SkillCategory']).size().unstack().fillna(0)
for category in overall_top_categories:
    if category not in location_skill_counts.columns:
        location_skill_counts[category] = 0
for location in location_skill_counts.index:
    create_radar_chart(location, location_skill_counts)


plt.show()