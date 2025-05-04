import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieLensDataLoader:
    """
    Data loader for the MovieLens dataset.
    
    This class handles loading, preprocessing, and splitting the MovieLens dataset
    for training and evaluating recommendation systems.
    """
    
    def __init__(
        self,
        data_path: str = "data/ml-100k",
        dataset_url: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the MovieLensDataLoader.
        
        Args:
            data_path (str): Path to the MovieLens dataset.
            dataset_url (str): URL to download the dataset if not found locally.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
        """
        self.data_path = data_path
        self.dataset_url = dataset_url
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize data containers
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.train_df = None
        self.test_df = None
        
        # User and movie mappings
        self.user_id_map = None
        self.movie_id_map = None
        self.n_users = 0
        self.n_movies = 0
        
        # Load data
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Load the MovieLens dataset.
        
        If the dataset is not found locally, it will be downloaded.
        """
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_path):
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                self._download_dataset()
            
            # Load ratings data
            ratings_path = os.path.join(self.data_path, "u.data")
            if not os.path.exists(ratings_path):
                ratings_path = os.path.join(self.data_path, "ml-100k/u.data")
            
            if not os.path.exists(ratings_path):
                self._download_dataset()
                if not os.path.exists(ratings_path):
                    ratings_path = os.path.join(self.data_path, "ml-100k/u.data")
            
            logger.info(f"Loading ratings data from {ratings_path}")
            self.ratings_df = pd.read_csv(
                ratings_path,
                sep='\t',
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            # Load user data
            users_path = os.path.join(self.data_path, "u.user")
            if not os.path.exists(users_path):
                users_path = os.path.join(self.data_path, "ml-100k/u.user")
            
            if os.path.exists(users_path):
                logger.info(f"Loading user data from {users_path}")
                self.users_df = pd.read_csv(
                    users_path,
                    sep='|',
                    names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
                )
            
            # Load movie data
            movies_path = os.path.join(self.data_path, "u.item")
            if not os.path.exists(movies_path):
                movies_path = os.path.join(self.data_path, "ml-100k/u.item")
            
            if os.path.exists(movies_path):
                logger.info(f"Loading movie data from {movies_path}")
                self.movies_df = pd.read_csv(
                    movies_path,
                    sep='|',
                    encoding='latin-1',
                    names=[
                        'movie_id', 'title', 'release_date', 'video_release_date',
                        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western'
                    ]
                )
            
            # Create user and movie ID mappings
            self._create_id_mappings()
            
            # Split data into train and test sets
            self._split_data()
            
            logger.info(f"Data loaded successfully: {len(self.ratings_df)} ratings, "
                       f"{self.n_users} users, {self.n_movies} movies")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _download_dataset(self) -> None:
        """
        Download the MovieLens dataset.
        """
        import requests
        import zipfile
        from io import BytesIO
        
        try:
            logger.info(f"Downloading MovieLens dataset from {self.dataset_url}")
            response = requests.get(self.dataset_url)
            response.raise_for_status()
            
            # Extract the zip file
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(os.path.dirname(self.data_path))
            
            logger.info("Dataset downloaded and extracted successfully")
        
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def _create_id_mappings(self) -> None:
        """
        Create mappings between original IDs and continuous indices.
        """
        # Create user ID mapping
        unique_user_ids = self.ratings_df['user_id'].unique()
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.n_users = len(self.user_id_map)
        
        # Create movie ID mapping
        unique_movie_ids = self.ratings_df['movie_id'].unique()
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
        self.n_movies = len(self.movie_id_map)
        
        # Add mapped IDs to the ratings dataframe
        self.ratings_df['user_idx'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['movie_idx'] = self.ratings_df['movie_id'].map(self.movie_id_map)
    
    def _split_data(self) -> None:
        """
        Split the ratings data into training and testing sets.
        """
        # Split data
        self.train_df, self.test_df = train_test_split(
            self.ratings_df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        logger.info(f"Data split: {len(self.train_df)} training samples, "
                   f"{len(self.test_df)} testing samples")
    
    def get_train_data(self) -> pd.DataFrame:
        """
        Get the training data.
        
        Returns:
            pd.DataFrame: Training data.
        """
        return self.train_df
    
    def get_test_data(self) -> pd.DataFrame:
        """
        Get the testing data.
        
        Returns:
            pd.DataFrame: Testing data.
        """
        return self.test_df
    
    def get_user_movie_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get user-movie rating matrices for training and testing.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Training and testing user-movie matrices.
        """
        # Create empty matrices
        train_matrix = np.zeros((self.n_users, self.n_movies))
        test_matrix = np.zeros((self.n_users, self.n_movies))
        
        # Fill training matrix
        for _, row in self.train_df.iterrows():
            train_matrix[row['user_idx'], row['movie_idx']] = row['rating']
        
        # Fill testing matrix
        for _, row in self.test_df.iterrows():
            test_matrix[row['user_idx'], row['movie_idx']] = row['rating']
        
        return train_matrix, test_matrix
    
    def get_data_loaders(
        self,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> Tuple[object, object]:
        """
        Get PyTorch DataLoader objects for training and testing.
        
        Args:
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            Tuple[object, object]: Training and testing DataLoader objects.
        """
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert training data to tensors
            train_user_tensor = torch.LongTensor(self.train_df['user_idx'].values)
            train_movie_tensor = torch.LongTensor(self.train_df['movie_idx'].values)
            train_rating_tensor = torch.FloatTensor(self.train_df['rating'].values)
            
            # Convert testing data to tensors
            test_user_tensor = torch.LongTensor(self.test_df['user_idx'].values)
            test_movie_tensor = torch.LongTensor(self.test_df['movie_idx'].values)
            test_rating_tensor = torch.FloatTensor(self.test_df['rating'].values)
            
            # Create datasets
            train_dataset = TensorDataset(train_user_tensor, train_movie_tensor, train_rating_tensor)
            test_dataset = TensorDataset(test_user_tensor, test_movie_tensor, test_rating_tensor)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, test_loader
        
        except ImportError:
            logger.error("PyTorch is required for creating data loaders")
            raise
    
    def get_movie_title(self, movie_idx: int) -> str:
        """
        Get the title of a movie by its index.
        
        Args:
            movie_idx (int): Movie index.
            
        Returns:
            str: Movie title.
        """
        if self.movies_df is None:
            return f"Movie {movie_idx}"
        
        # Find the original movie ID
        movie_id = None
        for original_id, idx in self.movie_id_map.items():
            if idx == movie_idx:
                movie_id = original_id
                break
        
        if movie_id is None:
            return f"Movie {movie_idx}"
        
        # Get the movie title
        movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]
        if len(movie_row) == 0:
            return f"Movie {movie_id}"
        
        return movie_row.iloc[0]['title']
    
    def get_user_info(self, user_idx: int) -> Dict:
        """
        Get information about a user by their index.
        
        Args:
            user_idx (int): User index.
            
        Returns:
            Dict: User information.
        """
        if self.users_df is None:
            return {"user_idx": user_idx}
        
        # Find the original user ID
        user_id = None
        for original_id, idx in self.user_id_map.items():
            if idx == user_idx:
                user_id = original_id
                break
        
        if user_id is None:
            return {"user_idx": user_idx}
        
        # Get the user information
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if len(user_row) == 0:
            return {"user_idx": user_idx, "user_id": user_id}
        
        user_info = user_row.iloc[0].to_dict()
        user_info["user_idx"] = user_idx
        
        return user_info
    
    def get_user_ratings(self, user_idx: int) -> Dict[int, float]:
        """
        Get all ratings by a user.
        
        Args:
            user_idx (int): User index.
            
        Returns:
            Dict[int, float]: Dictionary mapping movie indices to ratings.
        """
        user_ratings = {}
        
        # Get ratings from training data
        user_train_ratings = self.train_df[self.train_df['user_idx'] == user_idx]
        for _, row in user_train_ratings.iterrows():
            user_ratings[row['movie_idx']] = row['rating']
        
        # Get ratings from testing data
        user_test_ratings = self.test_df[self.test_df['user_idx'] == user_idx]
        for _, row in user_test_ratings.iterrows():
            user_ratings[row['movie_idx']] = row['rating']
        
        return user_ratings
    
    def get_movie_ratings(self, movie_idx: int) -> Dict[int, float]:
        """
        Get all ratings for a movie.
        
        Args:
            movie_idx (int): Movie index.
            
        Returns:
            Dict[int, float]: Dictionary mapping user indices to ratings.
        """
        movie_ratings = {}
        
        # Get ratings from training data
        movie_train_ratings = self.train_df[self.train_df['movie_idx'] == movie_idx]
        for _, row in movie_train_ratings.iterrows():
            movie_ratings[row['user_idx']] = row['rating']
        
        # Get ratings from testing data
        movie_test_ratings = self.test_df[self.test_df['movie_idx'] == movie_idx]
        for _, row in movie_test_ratings.iterrows():
            movie_ratings[row['user_idx']] = row['rating']
        
        return movie_ratings
    
    def get_user_movie_rating(self, user_idx: int, movie_idx: int) -> Optional[float]:
        """
        Get the rating of a movie by a user.
        
        Args:
            user_idx (int): User index.
            movie_idx (int): Movie index.
            
        Returns:
            Optional[float]: Rating or None if not rated.
        """
        # Check training data
        user_train_ratings = self.train_df[
            (self.train_df['user_idx'] == user_idx) &
            (self.train_df['movie_idx'] == movie_idx)
        ]
        if len(user_train_ratings) > 0:
            return user_train_ratings.iloc[0]['rating']
        
        # Check testing data
        user_test_ratings = self.test_df[
            (self.test_df['user_idx'] == user_idx) &
            (self.test_df['movie_idx'] == movie_idx)
        ]
        if len(user_test_ratings) > 0:
            return user_test_ratings.iloc[0]['rating']
        
        return None
    
    def get_user_stats(self, user_idx: int) -> Dict:
        """
        Get statistics about a user's ratings.
        
        Args:
            user_idx (int): User index.
            
        Returns:
            Dict: User statistics.
        """
        user_ratings = self.get_user_ratings(user_idx)
        
        if not user_ratings:
            return {
                "user_idx": user_idx,
                "num_ratings": 0,
                "avg_rating": 0.0,
                "min_rating": 0.0,
                "max_rating": 0.0
            }
        
        ratings = list(user_ratings.values())
        
        return {
            "user_idx": user_idx,
            "num_ratings": len(ratings),
            "avg_rating": np.mean(ratings),
            "min_rating": np.min(ratings),
            "max_rating": np.max(ratings)
        }
    
    def get_movie_stats(self, movie_idx: int) -> Dict:
        """
        Get statistics about a movie's ratings.
        
        Args:
            movie_idx (int): Movie index.
            
        Returns:
            Dict: Movie statistics.
        """
        movie_ratings = self.get_movie_ratings(movie_idx)
        
        if not movie_ratings:
            return {
                "movie_idx": movie_idx,
                "title": self.get_movie_title(movie_idx),
                "num_ratings": 0,
                "avg_rating": 0.0,
                "min_rating": 0.0,
                "max_rating": 0.0
            }
        
        ratings = list(movie_ratings.values())
        
        return {
            "movie_idx": movie_idx,
            "title": self.get_movie_title(movie_idx),
            "num_ratings": len(ratings),
            "avg_rating": np.mean(ratings),
            "min_rating": np.min(ratings),
            "max_rating": np.max(ratings)
        }
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dict: Dataset statistics.
        """
        return {
            "num_users": self.n_users,
            "num_movies": self.n_movies,
            "num_ratings": len(self.ratings_df),
            "sparsity": 1.0 - (len(self.ratings_df) / (self.n_users * self.n_movies)),
            "avg_rating": self.ratings_df['rating'].mean(),
            "min_rating": self.ratings_df['rating'].min(),
            "max_rating": self.ratings_df['rating'].max(),
            "num_train_ratings": len(self.train_df),
            "num_test_ratings": len(self.test_df)
        }
    
    def get_popular_movies(self, n: int = 10) -> List[Dict]:
        """
        Get the most popular movies based on number of ratings.
        
        Args:
            n (int): Number of movies to return.
            
        Returns:
            List[Dict]: List of movie information.
        """
        # Count ratings for each movie
        movie_counts = self.ratings_df['movie_idx'].value_counts()
        
        # Get the top N movies
        top_movies = []
        for movie_idx, count in movie_counts.head(n).items():
            movie_stats = self.get_movie_stats(movie_idx)
            movie_stats["num_ratings"] = count
            top_movies.append(movie_stats)
        
        return top_movies
    
    def get_top_rated_movies(self, n: int = 10, min_ratings: int = 10) -> List[Dict]:
        """
        Get the top rated movies.
        
        Args:
            n (int): Number of movies to return.
            min_ratings (int): Minimum number of ratings required.
            
        Returns:
            List[Dict]: List of movie information.
        """
        # Count ratings for each movie
        movie_counts = self.ratings_df['movie_idx'].value_counts()
        
        # Filter movies with enough ratings
        qualified_movies = movie_counts[movie_counts >= min_ratings].index
        
        # Calculate average rating for each qualified movie
        movie_ratings = {}
        for movie_idx in qualified_movies:
            movie_stats = self.get_movie_stats(movie_idx)
            movie_ratings[movie_idx] = movie_stats["avg_rating"]
        
        # Sort by average rating
        sorted_movies = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N movies
        top_movies = []
        for movie_idx, avg_rating in sorted_movies[:n]:
            movie_stats = self.get_movie_stats(movie_idx)
            top_movies.append(movie_stats)
        
        return top_movies
    
    def get_active_users(self, n: int = 10) -> List[Dict]:
        """
        Get the most active users based on number of ratings.
        
        Args:
            n (int): Number of users to return.
            
        Returns:
            List[Dict]: List of user information.
        """
        # Count ratings for each user
        user_counts = self.ratings_df['user_idx'].value_counts()
        
        # Get the top N users
        top_users = []
        for user_idx, count in user_counts.head(n).items():
            user_info = self.get_user_info(user_idx)
            user_stats = self.get_user_stats(user_idx)
            user_info.update(user_stats)
            top_users.append(user_info)
        
        return top_users
    
    def get_similar_users(self, user_idx: int, n: int = 10) -> List[Dict]:
        """
        Get users similar to the given user based on rating patterns.
        
        Args:
            user_idx (int): User index.
            n (int): Number of similar users to return.
            
        Returns:
            List[Dict]: List of similar user information.
        """
        # Get the user's ratings
        user_ratings = self.get_user_ratings(user_idx)
        
        if not user_ratings:
            return []
        
        # Calculate similarity with other users
        similarities = {}
        for other_idx in range(self.n_users):
            if other_idx == user_idx:
                continue
            
            other_ratings = self.get_user_ratings(other_idx)
            
            # Find common movies
            common_movies = set(user_ratings.keys()) & set(other_ratings.keys())
            
            if len(common_movies) < 5:  # Require at least 5 common ratings
                continue
            
            # Calculate Pearson correlation
            user_common = np.array([user_ratings[m] for m in common_movies])
            other_common = np.array([other_ratings[m] for m in common_movies])
            
            correlation = np.corrcoef(user_common, other_common)[0, 1]
            
            if not np.isnan(correlation):
                similarities[other_idx] = correlation
        
        # Sort by similarity
        sorted_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N similar users
        similar_users = []
        for other_idx, similarity in sorted_users[:n]:
            user_info = self.get_user_info(other_idx)
            user_stats = self.get_user_stats(other_idx)
            user_info.update(user_stats)
            user_info["similarity"] = similarity
            similar_users.append(user_info)
        
        return similar_users
    
    def get_similar_movies(self, movie_idx: int, n: int = 10) -> List[Dict]:
        """
        Get movies similar to the given movie based on rating patterns.
        
        Args:
            movie_idx (int): Movie index.
            n (int): Number of similar movies to return.
            
        Returns:
            List[Dict]: List of similar movie information.
        """
        # Get the movie's ratings
        movie_ratings = self.get_movie_ratings(movie_idx)
        
        if not movie_ratings:
            return []
        
        # Calculate similarity with other movies
        similarities = {}
        for other_idx in range(self.n_movies):
            if other_idx == movie_idx:
                continue
            
            other_ratings = self.get_movie_ratings(other_idx)
            
            # Find common users
            common_users = set(movie_ratings.keys()) & set(other_ratings.keys())
            
            if len(common_users) < 5:  # Require at least 5 common ratings
                continue
            
            # Calculate Pearson correlation
            movie_common = np.array([movie_ratings[u] for u in common_users])
            other_common = np.array([other_ratings[u] for u in common_users])
            
            correlation = np.corrcoef(movie_common, other_common)[0, 1]
            
            if not np.isnan(correlation):
                similarities[other_idx] = correlation
        
        # Sort by similarity
        sorted_movies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N similar movies
        similar_movies = []
        for other_idx, similarity in sorted_movies[:n]:
            movie_stats = self.get_movie_stats(other_idx)
            movie_stats["similarity"] = similarity
            similar_movies.append(movie_stats)
        
        return similar_movies


# Example usage
if __name__ == "__main__":
    # Create data loader
    data_loader = MovieLensDataLoader()
    
    # Get dataset statistics
    stats = data_loader.get_dataset_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get popular movies
    popular_movies = data_loader.get_popular_movies(n=5)
    print("\nPopular Movies:")
    for movie in popular_movies:
        print(f"  {movie['title']} - {movie['num_ratings']} ratings, "
              f"avg: {movie['avg_rating']:.2f}")
    
    # Get top rated movies
    top_rated_movies = data_loader.get_top_rated_movies(n=5)
    print("\nTop Rated Movies:")
    for movie in top_rated_movies:
        print(f"  {movie['title']} - avg: {movie['avg_rating']:.2f}, "
              f"{movie['num_ratings']} ratings")
    
    # Get active users
    active_users = data_loader.get_active_users(n=5)
    print("\nActive Users:")
    for user in active_users:
        print(f"  User {user['user_idx']} - {user['num_ratings']} ratings, "
              f"avg: {user['avg_rating']:.2f}")