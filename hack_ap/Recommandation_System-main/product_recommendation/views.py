import os
import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from django.core.files.storage import FileSystemStorage

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
    df = df.drop('timestamp', axis=1)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(float)

    counts = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(counts[counts >= 50].index)]
    final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)

    return df_final, final_ratings_matrix

def recommend_products(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(file_path)

        # Load and prepare the data
        df_final, final_ratings_matrix = load_data(file_path)

        # Convert to sparse matrix
        final_ratings_sparse = csr_matrix(final_ratings_matrix.values)
        U, s, Vt = svds(final_ratings_sparse, k=50)
        sigma = np.diag(s)

        # Predict ratings
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=final_ratings_matrix.columns)
        preds_matrix = csr_matrix(preds_df.values)

        # Extract top products
        top_products = df_final.groupby('prod_id').agg(
            average_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).sort_values(by='average_rating', ascending=False)
        top_products = top_products[top_products['rating_count'] > 50].head(5).reset_index().to_dict('records')

        # Calculate accuracy (placeholder for actual accuracy calculation)
        accuracy = 0.9  # Example value

        # Prepare sample data
        sample_data = df_final.head(5).to_dict('records')

        return render(request, 'output.html', {
            'top_products': top_products,
            'accuracy': accuracy,
            'sample_data': sample_data
        })
    else:
        return render(request, 'input.html')

def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
    user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Products')
    temp = temp.loc[temp.user_ratings == 0]
    temp = temp.sort_values('user_predictions', ascending=False)
    return temp['user_predictions'].head(num_recommendations)
