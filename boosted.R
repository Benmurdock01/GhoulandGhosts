library(tidymodels)
library(vroom)
library(doParallel)
library(kernlab)

#Setup Parallel Processing
num_cores <- 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

#Load Data
dat_train <- vroom("train.csv") %>%
  mutate(
    type = as.factor(type),
    color = as.factor(color)
  )

dat_test <- vroom("test.csv") %>%
  mutate(color = as.factor(color))

#SVM Recipe ---
svm_recipe <- recipe(type ~ ., data = dat_train) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(color, one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

#Define the Model (SVM)
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#Create the Workflow
svm_workflow <- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(svm_recipe)

#Set up Tuning
monster_folds <- vfold_cv(dat_train, v = 5, strata = type)

svm_grid <- grid_space_filling(
  cost(range = c(-2, 10)),
  rbf_sigma(range = c(-5, -1)),
  size = 15
)

#Run Tuning
svm_results <- svm_workflow %>%
  tune_grid(
    resamples = monster_folds,
    grid = svm_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

best_svm <- select_best(svm_results, metric = "roc_auc")

#Finalize and Fit
final_workflow <- svm_workflow %>%
  finalize_workflow(best_svm)

final_fit <- final_workflow %>%
  fit(data = dat_train)

#Generate Predictions
monster_preds <- predict(final_fit, new_data = dat_test)

#Create Submission File
submission <- tibble(
  id = dat_test$id,
  type = monster_preds$.pred_class
)

vroom_write(submission, "submissionSVM.csv", delim = ",")

stopCluster(cl)
