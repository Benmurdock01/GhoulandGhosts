library(vroom)
library(dplyr)
library(xgboost)

# --- 1. Load Data ---
data_train_raw <- vroom("./train.csv")
data_test_raw <- vroom("./test.csv")

# --- 2. Feature Engineering ---
# Create the same new features
data_train_fe <- data_train_raw %>%
  mutate(
    hair_bone_ratio = hair_length / (bone_length + 1e-6),
    flesh_soul_index = rotting_flesh * has_soul
  )

data_test_fe <- data_test_raw %>%
  mutate(
    hair_bone_ratio = hair_length / (bone_length + 1e-6),
    flesh_soul_index = rotting_flesh * has_soul
  )

# --- 3. Preprocessing for XGBoost ---
# XGBoost requires ALL data to be numeric.
# We must one-hot encode 'color' and convert 'type' to a number.

# Save original 'type' labels and convert to 0-indexed numeric
y_train_labels <- as.factor(data_train_fe$type)
y_train_numeric <- as.integer(y_train_labels) - 1
label_levels <- levels(y_train_labels) # To map back later

# Add a dummy 'type' to test set. This ensures 'model.matrix'
# creates identical columns for both train and test.
data_test_fe$type <- "Ghost" # Placeholder, value doesn't matter

# Bind rows to run one-hot encoding just once
full_data_fe <- bind_rows(data_train_fe, data_test_fe)

# Create the numeric matrix. '-1' removes the intercept.
# This formula uses all variables except 'id' to predict 'type'
full_sparse_matrix <- model.matrix(type ~ . - id - 1, data = full_data_fe)

# Split the matrix back into train and test
train_idx <- 1:nrow(data_train_fe)
test_idx <- (nrow(data_train_fe) + 1):nrow(full_data_fe)

X_train_matrix <- full_sparse_matrix[train_idx, ]
X_test_matrix <- full_sparse_matrix[test_idx, ]

# --- 4. Train XGBoost Model ---
# Create the optimized xgb.DMatrix data structures
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train_numeric)
dtest <- xgb.DMatrix(data = X_test_matrix)

# Set model parameters
params <- list(
  objective = "multi:softmax", # Multi-class classification
  num_class = 3,               # 3 types of creatures
  eta = 0.1,                   # learning_rate
  max_depth = 4,
  nthread = 4                  # Use 4 CPU threads
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200 # Number of trees
)

# --- 5. Predict ---
pred_numeric <- predict(xgb_model, dtest)

# --- 6. Format Submission ---
# Convert numeric predictions (0,1,2) back to text ("Ghost", "Ghoul", "Goblin")
pred_text <- label_levels[pred_numeric + 1] # +1 for 1-based R indexing

submission <- data.frame(id = data_test_raw$id, type = pred_text)

# --- 7. Write File (using vroom) ---
vroom_write(submission, "submissionsxgboot.csv", delim = ",")

