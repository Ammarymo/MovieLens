# Load necessary libraries
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(raster)) install.packages("raster")

library(tidyverse)
library(caret)
library(raster)

# Download and load the MovieLens dataset
dl <- "ml-10M100K.zip"
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
}

ratings_file <- "ml-10M100K/ratings.dat"
if (!file.exists(ratings_file)) {
  unzip(dl, ratings_file)
}

movies_file <- "ml-10M100K/movies.dat"
if (!file.exists(movies_file)) {
  unzip(dl, movies_file)
}

# Load ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Load movies data
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join ratings and movies data
movielens <- left_join(ratings, movies, by = "movieId")

# Split data into edx and final_holdout_test sets
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up unnecessary objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split edx data into training and testing sets
set.seed(1, sample.kind = "Rounding")
test_indices <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_data <- edx[-test_indices,]
test_data <- edx[test_indices,]

# Ensure userId and movieId in test_data are also in train_data
test_data <- test_data %>%
  semi_join(train_data, by = "movieId") %>%
  semi_join(train_data, by = "userId")

# Add rows removed from test_data back into train_data
removed_data <- anti_join(edx[test_indices,], test_data)
train_data <- rbind(train_data, removed_data)

# Clean up unnecessary objects
rm(test_indices, removed_data)

# Separate genres in the training and testing sets
train_genres <- train_data %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

test_genres <- test_data %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

# Function to calculate RMSE
calculate_rmse <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Naive Model: Predict the average rating for all movies and users
mean_rating <- mean(train_data$rating)
naive_model_rmse <- calculate_rmse(test_data$rating, mean_rating)

# Movie Effect Model: Incorporate movie-specific biases
movie_effects <- train_data %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - mean_rating))

predicted_ratings_movie_effect <- mean_rating + test_data %>%
  left_join(movie_effects, by = 'movieId') %>%
  pull(movie_effect)

predicted_ratings_movie_effect <- clamp(predicted_ratings_movie_effect, 0.5, 5)

movie_effect_rmse <- calculate_rmse(predicted_ratings_movie_effect, test_data$rating)

# User Effect Model: Add user-specific biases
user_effects <- train_data %>%
  left_join(movie_effects, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - mean_rating - movie_effect))

predicted_ratings_user_effect <- test_data %>%
  left_join(movie_effects, by = 'movieId') %>%
  left_join(user_effects, by = 'userId') %>%
  mutate(pred = mean_rating + movie_effect + user_effect) %>%
  pull(pred)

predicted_ratings_user_effect <- clamp(predicted_ratings_user_effect, 0.5, 5)

user_effect_rmse <- calculate_rmse(predicted_ratings_user_effect, test_data$rating)

# Genre Effect Model: Include genre-specific effects
genre_effects <- train_genres %>%
  left_join(movie_effects, by = 'movieId') %>%
  left_join(user_effects, by = 'userId') %>%
  group_by(genres) %>%
  summarize(genre_effect = mean(rating - mean_rating - movie_effect - user_effect))

predicted_ratings_genre_effect <- test_genres %>%
  left_join(movie_effects, by = 'movieId') %>%
  left_join(user_effects, by = 'userId') %>%
  left_join(genre_effects, by = 'genres') %>%
  mutate(pred = mean_rating + movie_effect + user_effect + genre_effect) %>%
  pull(pred)

predicted_ratings_genre_effect <- clamp(predicted_ratings_genre_effect, 0.5, 5)

genre_effect_rmse <- calculate_rmse(predicted_ratings_genre_effect, test_genres$rating)

# Genre-User Interaction Model: Introduce interaction effects between users and genres
genre_user_interaction <- train_genres %>%
  left_join(movie_effects, by = 'movieId') %>%
  left_join(user_effects, by = 'userId') %>%
  left_join(genre_effects, by = 'genres') %>%
  group_by(genres, userId) %>%
  summarize(genre_user_effect = mean(rating - mean_rating - movie_effect - user_effect - genre_effect))

predicted_ratings_genre_user_effect <- test_genres %>%
  left_join(movie_effects, by = 'movieId') %>%
  left_join(user_effects, by = 'userId') %>%
  left_join(genre_effects, by = 'genres') %>%
  left_join(genre_user_interaction, by = c("userId", "genres")) %>%
  mutate(genre_user_effect = ifelse(is.na(genre_user_effect), 0, genre_user_effect),
         pred = mean_rating + movie_effect + user_effect + genre_effect + genre_user_effect) %>%
  pull(pred)

predicted_ratings_genre_user_effect <- clamp(predicted_ratings_genre_user_effect, 0.5, 5)

genre_user_effect_rmse <- calculate_rmse(predicted_ratings_genre_user_effect, test_genres$rating)

# Regularized Model: Apply regularization to penalize overfitting
regularization_function <- function(lambda) {
  mean_rating <- mean(train_data$rating)
  
  movie_effects_reg <- train_data %>%
    group_by(movieId) %>%
    summarize(movie_effect = sum(rating - mean_rating) / (n() + lambda))
  
  user_effects_reg <- train_data %>%
    left_join(movie_effects_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarize(user_effect = sum(rating - movie_effect - mean_rating) / (n() + lambda))
  
  genre_effects_reg <- train_genres %>%
    left_join(movie_effects_reg, by = "movieId") %>%
    left_join(user_effects_reg, by = "userId") %>%
    group_by(genres) %>%
    summarize(genre_effect = sum(rating - mean_rating - movie_effect - user_effect) / (n() + lambda))
  
  genre_user_interaction_reg <- train_genres %>%
    left_join(movie_effects_reg, by = "movieId") %>%
    left_join(user_effects_reg, by = "userId") %>%
    left_join(genre_effects_reg, by = "genres") %>%
    group_by(genres, userId) %>%
    summarize(genre_user_effect = sum(rating - mean_rating - movie_effect - user_effect - genre_effect) / (n() + lambda))
  
  predicted_ratings_reg <- test_genres %>%
    left_join(movie_effects_reg, by = "movieId") %>%
    left_join(user_effects_reg, by = "userId") %>%
    left_join(genre_effects_reg, by = "genres") %>%
    left_join(genre_user_interaction_reg, by = c("userId", "genres")) %>%
    mutate(genre_user_effect = ifelse(is.na(genre_user_effect), 0, genre_user_effect),
           pred = mean_rating + movie_effect + user_effect + genre_effect + genre_user_effect) %>%
    pull(pred)
  
  predicted_ratings_reg <- clamp(predicted_ratings_reg, 0.5, 5)
  
  return(calculate_rmse(predicted_ratings_reg, test_genres$rating))
}

# Find the optimal lambda value
lambda_values <- seq(11.5, 12.5, 0.2)
rmse_values <- sapply(lambda_values, regularization_function)

# Plot RMSE values against lambda values
plot(lambda_values, rmse_values)

# Select the optimal lambda value
optimal_lambda <- lambda_values[which.min(rmse_values)]
regularized_model_rmse <- min(rmse_values)

# Evaluate the final model on the final holdout test set
final_holdout_genres <- final_holdout_test %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

# Calculate effects using the optimal lambda
mean_rating_final <- mean(train_data$rating)

movie_effects_final <- train_data %>%
  group_by(movieId) %>%
  summarize(movie_effect = sum(rating - mean_rating_final) / (n() + optimal_lambda))

user_effects_final <- train_data %>%
  left_join(movie_effects_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = sum(rating - movie_effect - mean_rating_final) / (n() + optimal_lambda))

genre_effects_final <- train_genres %>%
  left_join(movie_effects_final, by = "movieId") %>%
  left_join(user_effects_final, by = "userId") %>%
  group_by(genres) %>%
  summarize(genre_effect = sum(rating - mean_rating_final - movie_effect - user_effect) / (n() + optimal_lambda))

genre_user_interaction_final <- train_genres %>%
  left_join(movie_effects_final, by = "movieId") %>%
  left_join(user_effects_final, by = "userId") %>%
  left_join(genre_effects_final, by = "genres") %>%
  group_by(genres, userId) %>%
  summarize(genre_user_effect = sum(rating - mean_rating_final - movie_effect - user_effect - genre_effect) / (n() + optimal_lambda))

# Predict ratings on the final holdout test set
predicted_ratings_final <- final_holdout_genres %>%
  left_join(movie_effects_final, by = "movieId") %>%
  left_join(user_effects_final, by = "userId") %>%
  left_join(genre_effects_final, by = "genres") %>%
  left_join(genre_user_interaction_final, by = c("userId", "genres")) %>%
  mutate(genre_user_effect = ifelse(is.na(genre_user_effect), 0, genre_user_effect),
         pred = mean_rating_final + movie_effect + user_effect + genre_effect + genre_user_effect) %>%
  pull(pred)

# Clamp predictions to be between 0.5 and 5
predicted_ratings_final <- clamp(predicted_ratings_final, 0.5, 5)

# Calculate RMSE for the final holdout test set
final_holdout_rmse <- calculate_rmse(predicted_ratings_final, final_holdout_genres$rating)

# Print the final RMSE results
rmse_results <- data.frame(
  Model = c("Naive Model", "Movie Effect Model", "User Effect Model",
            "Genre Effect Model", "Genre-User Interaction Model", "Regularized Model"),
  RMSE = c(naive_model_rmse, movie_effect_rmse, user_effect_rmse,
           genre_effect_rmse, genre_user_effect_rmse, regularized_model_rmse)
)

print(rmse_results)
print(paste("Final Holdout Test Set RMSE:", final_holdout_rmse))