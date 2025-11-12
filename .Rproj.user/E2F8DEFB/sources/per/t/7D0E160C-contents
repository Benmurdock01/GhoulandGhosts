library(tidymodels)
library(vroom)
library(doParallel)
library(stacks)    
library(kernlab)     
library(kknn)       


num_cores <- 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

dat_train <- vroom("train.csv") %>%
  mutate(type = as.factor(type), color = as.factor(color))
dat_test <- vroom("test.csv") %>%
  mutate(color = as.factor(color))

monster_folds <- vfold_cv(dat_train, v = 5, strata = type)

ctrl_stack <- control_stack_grid()


# RF Recipe
rf_recipe <- recipe(type ~ ., data = dat_train) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(color, one_hot = TRUE) %>%
  step_zv(all_predictors())

# Normalized Recipe
norm_recipe <- recipe(type ~ ., data = dat_train) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(color, one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# Random Forest
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 750 # Slightly fewer trees for faster tuning
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# SVM
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#k-Nearest Neighbor
knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

wf_rf <- workflow(rf_recipe, rf_spec)
wf_svm <- workflow(norm_recipe, svm_spec)
wf_knn <- workflow(norm_recipe, knn_spec)

#Tune
#RF Tuning
set.seed(234)
rf_grid <- grid_space_filling(mtry(range = c(1, 5)), min_n(range = c(2, 20)), size = 10)
rf_results <- wf_rf %>%
  tune_grid(
    resamples = monster_folds,
    grid = rf_grid,
    control = ctrl_stack,
    metrics = metric_set(roc_auc)
  )

#SVM Tuning
set.seed(234)
svm_grid <- grid_space_filling(cost(range = c(-2, 10)), rbf_sigma(range = c(-5, -1)), size = 10)
svm_results <- wf_svm %>%
  tune_grid(
    resamples = monster_folds,
    grid = svm_grid,
    control = ctrl_stack,
    metrics = metric_set(roc_auc)
  )

#knn Tuning
set.seed(234)
knn_grid <- grid_regular(neighbors(range = c(5, 30)), levels = 10)
knn_results <- wf_knn %>%
  tune_grid(
    resamples = monster_folds,
    grid = knn_grid,
    control = ctrl_stack,
    metrics = metric_set(roc_auc)
  )

monster_stack <- stacks() %>%
  add_candidates(rf_results) %>%
  add_candidates(svm_results) %>%
  add_candidates(knn_results)

blended_model <- monster_stack %>%
  blend_predictions(
    metric = metric_set(roc_auc),
    penalty = 10^seq(-4, -1, length = 20)
  )


#Fit the Final Stack
final_stack_fit <- blended_model %>%
  fit_members()

#Generate Predictions
stack_preds <- predict(final_stack_fit, new_data = dat_test)

#Create Submission File
submission <- tibble(
  id = dat_test$id,
  type = stack_preds$.pred_class
)

vroom_write(submission, "submission_STACK.csv", delim = ",")

stopCluster(cl)
