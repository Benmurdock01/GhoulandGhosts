# Load the libraries
library(vroom)
library(tidymodels)
library(embed)

#import data
data_train <- vroom("./train.csv") %>%
  mutate(type = as.factor(type))
data_test <- vroom("./test.csv")a

#recipe
rf_recipe <- recipe(type ~ ., data = data_train) %>%
  step_dummy(color, one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

#rf model
rf_spec <- rand_forest() %>%
  set_args(
    trees = 3000,  
    mtry = 3
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "permutation")
  

#workflow
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_spec)

#cross validation
data_folds <- vfold_cv(data_train, v = 15, strata = type)

cv_results <- rf_workflow %>%
  fit_resamples(
    resamples = data_folds
  )

#train model
rf_fit <- rf_workflow %>%
  fit(data = data_train)

#fit predictions
predictions <- predict(rf_fit, new_data = data_test)

#format submission
submission <- bind_cols(
  data_test %>% select(id),
  predictions
) %>%
  rename(type = .pred_class)

vroom_write(submission, "submissionRF_CV.csv", delim = ",")



