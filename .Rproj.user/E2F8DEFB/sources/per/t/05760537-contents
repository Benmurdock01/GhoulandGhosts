# Libraries
library(tidymodels)
library(vroom)
library(parsnip)

# Load data
train_data <- vroom("./train.csv", show_col_types = FALSE)
test_data  <- vroom("./test.csv", show_col_types = FALSE)

# Recipe
creature_recipe <-
  recipe(type ~ ., data = train_data) %>%
  step_rm(id, color) %>%
  step_mutate(
    hair_soul  = hair_length * has_soul,
    bone_flesh = bone_length * rotting_flesh,
    bone_hair  = bone_length * hair_length,
    bone_soul  = bone_length * has_soul,
    flesh_hair = rotting_flesh * hair_length,
    flesh_soul = rotting_flesh * has_soul
  ) %>%
  step_normalize(all_numeric_predictors())

# Model specification (multinomial regression for multiclass classification)
glmnet_spec <-
  multinom_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Workflow
creature_workflow <-
  workflow() %>%
  add_recipe(creature_recipe) %>%
  add_model(glmnet_spec)

# Cross-validation
cv_folds <- vfold_cv(train_data, v = 10, strata = type)

# Hyperparameter grid
tune_grid_values <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 20
)

# Tuning
tune_results <- tune_grid(
  creature_workflow,
  resamples = cv_folds,
  grid = tune_grid_values,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = FALSE, save_workflow = FALSE, verbose = TRUE)
)

# Finalize the model
best_params <- select_best(tune_results, metric = "accuracy")
final_workflow <- finalize_workflow(creature_workflow, best_params)

# Train final model
final_fit <- fit(final_workflow, data = train_data)

# Predictions
predictions <- predict(final_fit, new_data = test_data)

# Submission file
submission <- tibble(
  id = test_data$id,
  type = predictions$.pred_class
)

vroom_write(submission, "submissionGLM.csv", delim = ",")
