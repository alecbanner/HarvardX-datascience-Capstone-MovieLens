################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Building Models
library(tidyverse)
library(caret)

#display top 10 rows of data
edx %>% as_tibble()

#number of users and number of movies
edx %>%
  summarise(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

#add publication year to edx and validation datasets
edx <- edx %>% mutate(year = as.numeric(str_sub(title, -5, -2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title, -5, -2)))

#years from which films are from
edx %>%
  summarise(earliest = year[which.min(year)], latest = year[which.max(year)])

#RMSE equation
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((predicted_ratings-true_ratings)^2))
}

#naive model
mu <- mean(edx$rating)

#movie model effects
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))
#plot movie model effects
qplot(b_i, data = movie_avgs, bins = 10)

#user effects
user_effects <- edx %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating))
#plot movie user effects
qplot(b_u, data = user_effects, bins = 30)

#year effects
year_effects <- edx %>%
  group_by(year) %>%
  summarise(avg_rating = mean(rating))
#plot year effects
year_effects %>%
  ggplot(aes(year, avg_rating)) +
  geom_bar(stat="identity") +
  geom_hline(yintercept = mu)

#genre effects

#separate movies under multiple rows into one row per genre
#edx_genres <- edx %>% separate_rows(genres,sep = "\\|")
#validation_genres <- validation %>% separate_rows(genres,sep = "\\|")

#Regularisation
#split edx into training_set and test_set so that validation set isn't used for regularisation
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "year")

#naive model predictions
naive_RMSE <- RMSE(validation$rating, mu)
#initiate table of RMSEs
library(knitr)
table_RMSE_comparison <- tibble(method = 'naive', RMSE = naive_RMSE)
kable(table_RMSE_comparison)

#make predictions based on movie model effects
predicted_ratings_movie <- mu + validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

movie_model_RMSE <- RMSE(validation$rating, predicted_ratings_movie)

#add movie model to RMSE table
table_RMSE_comparison <- bind_rows(table_RMSE_comparison, tibble(method = 'Movie Model', RMSE = movie_model_RMSE))
kable(table_RMSE_comparison)

#user effects model
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))


predicted_ratings_movie_user <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movie_user_model_RMSE <- RMSE(validation$rating, predicted_ratings_movie_user)

#add user effects model to RMSE table
table_RMSE_comparison <- bind_rows(table_RMSE_comparison, tibble(method = 'User Model', RMSE = movie_user_model_RMSE))
kable(table_RMSE_comparison)

#year model
year_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(year) %>%
  summarise(b_y = mean(rating - mu - b_i - b_u))

predicted_ratings_movie_user_year <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)

movie_user_year_model_RMSE <- RMSE(validation$rating, predicted_ratings_movie_user)

#add year model to RMSE table
table_RMSE_comparison <- bind_rows(table_RMSE_comparison, tibble(method = 'Year Model', RMSE = movie_user_year_model_RMSE))
kable(table_RMSE_comparison)

#Regularisation

lambdas <- seq(0, 10, 1)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set%>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(year) %>%
    summarise(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

plot(lambdas, rmses)

#second round of regularisation
lambdas <- seq(4, 5, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set%>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(year) %>%
    summarise(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

plot(lambdas, rmses)
lambdas[which.min(rmses)]
which.min(rmses)
rmses[which.min(rmses)]

#final Model for the edx dataset using lambda which gave the minimum rmse on the test set

l <- lambdas[which.min(rmses)]
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n()+l))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i)/(n()+l))

b_y <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(year) %>%
  summarise(b_y = sum(rating - mu - b_i - b_u)/(n()+l))

predicted_ratings_regularisation <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)


regularisation_model <- RMSE(validation$rating, predicted_ratings_regularisation)

#add final model to RMSE table
table_RMSE_comparison <- bind_rows(table_RMSE_comparison, tibble(method = 'Regulisation', RMSE = regularisation_model))
kable(table_RMSE_comparison)

#Recommenderlab model
library(recommenderlab)
#create minimised train and test sets
train_recommender <- train_set %>%
  group_by(movieId) %>%
  filter(n()>5000) %>%
  ungroup() %>%
  group_by(userId) %>%
  filter(n()>100) %>%
  ungroup()

test_recommender <- test_set %>%
  semi_join(train_recommender, by = "movieId") %>%
  semi_join(train_recommender, by = "userId")

#create recommender_matrix
recommender_matrix <- train_recommender %>%
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()
#add row names by userId
rownames(recommender_matrix) <- recommender_matrix[,1]
recommender_matrix <- recommender_matrix[,-1]
movie_titles <- train_set %>%
  select(movieId, title) %>%
  distinct()
#add column names, title by movieId
colnames(recommender_matrix) <- with(movie_titles, title[match(colnames(recommender_matrix), movieId)])

#creat test_matrix
test_matrix <- test_recommender %>%
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()
#add row names by userId
rownames(test_matrix) <- test_matrix[,1]
test_matrix <- test_matrix[,-1]
movie_titles_test <- test_set %>%
  select(movieId, title) %>%
  distinct()
#add column names, title by movieId
colnames(test_matrix) <- with(movie_titles_test, title[match(colnames(test_matrix), movieId)])

#coerce to realRatingMatrix as required by recommender
recommender_matrix <- as(recommender_matrix, "realRatingMatrix")
test_matrix <- as(test_matrix, "realRatingMatrix")

#check the class of the new matrix
class(recommender_matrix)
class(test_matrix)

#create model based on user-user (UBCF) interactions
rec_model <-Recommender(recommender_matrix, method = "UBCF")

#complete the testMatrix based on the model
pred <- predict(rec_model, test_matrix, type="ratingMatrix")

#calculatepredictionaccuracy
calcPredictionAccuracy(pred, test_matrix)