##########################################
##title: "Movie Recommendation System"####
##author: yfu2015                     ####
##date: "2022-09-20"                  ####
##########################################

## ----setup, include=FALSE------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ------------------------------------------------------------------------------------------------------------------------------------------
# install packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")

# Loading packages

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(dslabs)


## ------------------------------------------------------------------------------------------------------------------------------------------
# MovieLens 10M dataset:

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))    ##str(ratings)

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))    ##str(movies)

movielens <- left_join(ratings, movies, by = "movieId")


## ------------------------------------------------------------------------------------------------------------------------------------------
# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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


# Convert the 'timestamp' to date in which the rating was provided,
# and then to week in which the rating was provided

edx <- mutate(edx, date = round_date(as_datetime(timestamp), unit = "week"))
validation <- mutate(validation, date = round_date(as_datetime(timestamp), unit = "week"))


# Remove the column of timestamp
edx <- edx[, c("movieId", "userId", "rating", "title", "genres", "date") ]
validation <- validation[, c("movieId", "userId", "rating", "title", "genres", "date") ]



## ------------------------------------------------------------------------------------------------------------------------------------------

## Build regularized  movie + user Effect model

## This model will be compared to the final Modeling movie + User + date Effects

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
    return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)
min(rmses)








## ------------------------------------------------------------------------------------------------------------------------------------------

# 1. Compute the average rating for each week ( the week since January 1, 1970) 
# and plot this average against date

edx %>% 
	group_by(date) %>%
	summarize(rating = mean(rating)) %>%
	ggplot(aes(date, rating)) +
	geom_point() +
	geom_smooth()



## ------------------------------------------------------------------------------------------------------------------------------------------

#2. Define the function RMSE 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Define a range of lambdas 

lambdas <- seq(0, 10, 0.25)

# Construct a function and calculate residual mean squared error 

rmses <- sapply(lambdas, function(l){
  
  # Average rating on the training data set
  
  mu <- mean(edx$rating)   

  # calculate b_i by applying penalized lambda l in the equation
  # -- b_i is regularized
    
  b_i <- edx  %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))  
  
  # calculate b_u for a user of a movie by applying penalized lambda l in the equation
  # -- b_u is regularized
  
  b_u <- edx  %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))   
  
  # calculate b_date for a user of a movie by applying penalized lambda l in the equation
  # -- b_date is regularized
  
  b_date_m <- edx  %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId')   %>% 
    group_by(date) %>%
    summarize( b_date = sum(rating - b_i - b_u - mu)/(n() + l))   
  
  # replace those NA values of b_date with median value of b_date_m$b_date
  
  b_date_m$b_date[is.na(b_date_m$b_date)] <- median(b_date_m$b_date, na.rm=T)
  
  
  # Calculate the predicted ratings

  predicted_ratings <- 
    validation %>% 
    #left join dataset b_i in order to retrieve regularized b_i
    left_join(b_i, by = "movieId") %>%  
    #left join dataset b_u in order to retrieve regularized b_u 
    left_join(b_u, by = "userId") %>%
    #left join b_date_m in order to retrieve regularized value b_date
    left_join(b_date_m, by=c('date' )) %>%
    # the predicted rating 
    mutate(pred = mu + b_i + b_u + b_date) %>%
    pull(pred)                                               
  
  # Replace those NA values of predicted_ratings with median value of predicted_ratings 
  
  predicted_ratings[is.na(predicted_ratings)] <- median(predicted_ratings, na.rm=T)
  

  # Calculate RMSEs
  
  return(RMSE(predicted_ratings, validation$rating))
  
})

qplot(lambdas, rmses, main="RMSEs vs Lamdas")  


rmse_results <- tibble(Name="min rmse", Value=min(rmses) )


rmse_results <- bind_rows(rmse_results,
                              tibble(Name="Optimal Lambda",  
                                         Value=lambdas[which.min(rmses)]   ))
rmse_results %>% knitr::kable(align='c')


