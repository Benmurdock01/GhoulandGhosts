library(tidymodels)
library(vroom)
library(dplyr) 

#load data
data_train <- vroom("./train.csv") %>% 
  mutate(type = as.factor(type))
data_test <- vroom("./test.csv")

#KNN Recipe
knn_recipe <- recipe(type ~ ., data = data_train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate(
    hair_bone_ratio = hair_length / (bone_length + 1e-6),
    soul_minus_flesh = has_soul - rotting_flesh
  ) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

#KNN Model

knn_spec <- nearest_neighbor(neighbors = 2) %>%
  set_mode("classification") %>%
  set_engine("kknn")

#Workflow
knn_workflow <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_spec)

#train model
knn_fit <- knn_workflow %>%
  fit(data = data_train)

#Make Predictions and save
predictions_knn <- predict(knn_fit, new_data = data_test)

submission_knn <- bind_cols(
  data_test %>% select(id),
  predictions_knn
) %>%
  rename(type = .pred_class)

vroom_write(submission_knn, "submissionKNN_simple.csv", delim = ",")

