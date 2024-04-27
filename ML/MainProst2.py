import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(dataframe):
    dataframe['Keys'] = dataframe['Keys'].apply(eval)
    return dataframe

def get_course_recommendations(vacancy_data, courses_df, top_n=3):
    vacancy_data['Keywords'] = vacancy_data['Keys'].apply(lambda x: ' '.join(x))
    
    tfidf_vectorizer = TfidfVectorizer()
    all_texts = vacancy_data['Keywords'].tolist() + courses_df['description'].tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    vacancy_vectors = tfidf_matrix[:len(vacancy_data)]
    course_vectors = tfidf_matrix[len(vacancy_data):]
    
    similarity_scores = cosine_similarity(vacancy_vectors, course_vectors)
    
    recommendations = []
    for idx, vacancy in enumerate(vacancy_data.itertuples()):
        vacancy_recommendations = {
            "vacancy_id": vacancy.Ids,
            "vacancy_name": vacancy.Name,
            "recommended_courses": []
        }
        course_sim_scores = similarity_scores[idx]
        top_courses = sorted(list(enumerate(course_sim_scores)), key=lambda x: x[1], reverse=True)[:top_n]
        
        for course_idx, score in top_courses:
            course_info = {
                "course_name": courses_df.iloc[course_idx]['name'],
                "course_description": courses_df.iloc[course_idx]['description'],
                "similarity_score": score
            }
            vacancy_recommendations["recommended_courses"].append(course_info)
        
        recommendations.append(vacancy_recommendations)
    
    return recommendations

vacancies = load_data('path_to_vacancies.csv')
vacancies = preprocess_data(vacancies)
courses_df = load_data('path_to_courses.csv')  # файл с курсами

recommendations = get_course_recommendations(vacancies, courses_df)
json_output = json.dumps(recommendations, indent=4)
print(json_output)
