#!/usr/bin/env Rscript

# Step 2: R machine learning comparison for the research layer.
# This script trains 5 ML methods on the Dataset 3 influence classification task.

SCRIPT_VERSION <- "research_layer_step02_five_ml_methods_v1"

TARGET_COLUMN <- "medium_or_high_influence"
ID_COLUMN <- "canonical_entity_id"

OUTPUT_FOLDER <- "research_layer/outputs"

REQUIRED_PACKAGES <- c("rpart", "randomForest", "e1071")

LEAKAGE_COLUMNS <- c(
  "rank_overall",
  "influence_score",
  "influence_band",
  "influence_band_clean"
)


input_candidates <- function() {
  c(
    "research_layer/outputs/dataset3_model_feature_table.csv",
    "outputs/dataset3_model_feature_table.csv",
    "dataset3_model_feature_table.csv"
  )
}


find_input_file <- function() {
  candidate_paths <- input_candidates()
  existing_paths <- candidate_paths[file.exists(candidate_paths)]
  if (length(existing_paths) == 0) {
    stop("Could not find dataset3_model_feature_table.csv.")
  }
  existing_paths[[1]]
}


install_missing_packages <- function() {
  for (package_name in REQUIRED_PACKAGES) {
    if (!requireNamespace(package_name, quietly = TRUE)) {
      install.packages(package_name, repos = "https://cloud.r-project.org")
    }
  }
}


load_feature_table <- function(file_path) {
  tryCatch(
    read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(read_error) {
      stop(paste("Could not read feature table:", read_error$message))
    }
  )
}


validate_feature_table <- function(feature_table) {
  if (!TARGET_COLUMN %in% names(feature_table)) {
    stop(paste("Missing target column:", TARGET_COLUMN))
  }
  if (nrow(feature_table) < 2) {
    stop("Feature table needs at least two rows.")
  }
  TRUE
}


clean_number_column <- function(raw_column) {
  clean_text <- gsub(",", "", as.character(raw_column))
  numeric_values <- suppressWarnings(as.numeric(clean_text))
  numeric_values[is.na(numeric_values)] <- 0
  numeric_values
}


candidate_predictors <- function(feature_table) {
  blocked_columns <- c(TARGET_COLUMN, ID_COLUMN, LEAKAGE_COLUMNS)
  setdiff(names(feature_table), blocked_columns)
}


coerce_predictors <- function(feature_table, predictor_names) {
  for (column_name in predictor_names) {
    feature_table[[column_name]] <- clean_number_column(feature_table[[column_name]])
  }
  feature_table
}


has_variation <- function(feature_column) {
  unique_values <- unique(feature_column[!is.na(feature_column)])
  length(unique_values) > 1
}


remove_zero_variance <- function(model_table, predictor_names) {
  useful_predictors <- predictor_names[sapply(model_table[predictor_names], has_variation)]
  if (length(useful_predictors) == 0) {
    stop("No useful predictors remain after cleaning.")
  }
  list(table = model_table, predictors = useful_predictors)
}


make_target_columns <- function(feature_table) {
  target_numeric <- as.integer(feature_table[[TARGET_COLUMN]])
  target_numeric[is.na(target_numeric)] <- 0
  target_factor <- factor(ifelse(target_numeric == 1, "class_1", "class_0"))
  list(numeric = target_numeric, factor = target_factor)
}


prepare_model_table <- function(feature_table) {
  validate_feature_table(feature_table)
  predictor_names <- candidate_predictors(feature_table)
  feature_table <- coerce_predictors(feature_table, predictor_names)
  cleaned_result <- remove_zero_variance(feature_table, predictor_names)
  build_model_table(cleaned_result$table, cleaned_result$predictors)
}


build_model_table <- function(feature_table, predictor_names) {
  target_values <- make_target_columns(feature_table)
  model_table <- feature_table[c(ID_COLUMN, predictor_names)]
  model_table$target_numeric <- target_values$numeric
  model_table$target_factor <- target_values$factor
  validate_target_classes(model_table)
  model_table
}


validate_target_classes <- function(model_table) {
  class_count <- length(unique(model_table$target_numeric))
  if (class_count < 2) {
    stop("Target column must contain both positive and negative classes.")
  }
  TRUE
}


predictor_names_from_table <- function(model_table) {
  setdiff(names(model_table), c(ID_COLUMN, "target_numeric", "target_factor"))
}


sample_class_rows <- function(row_numbers, train_share) {
  sample_count <- max(1, floor(length(row_numbers) * train_share))
  sample(row_numbers, min(sample_count, length(row_numbers)))
}


make_stratified_split <- function(model_table, train_share) {
  positive_rows <- which(model_table$target_numeric == 1)
  negative_rows <- which(model_table$target_numeric == 0)
  train_rows <- c(sample_class_rows(positive_rows, train_share), sample_class_rows(negative_rows, train_share))
  test_rows <- setdiff(seq_len(nrow(model_table)), train_rows)
  list(train = model_table[train_rows, ], test = model_table[test_rows, ])
}


validate_split <- function(split_tables) {
  if (nrow(split_tables$train) == 0 || nrow(split_tables$test) == 0) {
    stop("Train/test split failed because one split is empty.")
  }
  TRUE
}


model_formula <- function(target_name, predictor_names) {
  as.formula(paste(target_name, "~", paste(predictor_names, collapse = " + ")))
}


positive_class_weight <- function(train_table) {
  positive_count <- sum(train_table$target_numeric == 1)
  negative_count <- sum(train_table$target_numeric == 0)
  if (positive_count == 0) {
    return(1)
  }
  negative_count / positive_count
}


row_class_weights <- function(train_table) {
  positive_weight <- positive_class_weight(train_table)
  ifelse(train_table$target_numeric == 1, positive_weight, 1)
}


class_weight_vector <- function(train_table) {
  c(class_0 = 1, class_1 = positive_class_weight(train_table))
}


balanced_training_table <- function(train_table) {
  positive_rows <- train_table[train_table$target_numeric == 1, ]
  negative_rows <- train_table[train_table$target_numeric == 0, ]
  repeated_positive <- positive_rows[rep(seq_len(nrow(positive_rows)), length.out = nrow(negative_rows)), ]
  rbind(negative_rows, repeated_positive)
}


train_logistic_regression <- function(train_table, predictor_names) {
  train_table$class_weight <- row_class_weights(train_table)
  fit_formula <- model_formula("target_numeric", predictor_names)
  suppressWarnings(glm(fit_formula, data = train_table, family = binomial(), weights = class_weight))
}


train_decision_tree <- function(train_table, predictor_names) {
  train_table$class_weight <- row_class_weights(train_table)
  fit_formula <- model_formula("target_factor", predictor_names)
  rpart::rpart(fit_formula, data = train_table, method = "class", weights = class_weight)
}


train_random_forest <- function(train_table, predictor_names) {
  fit_formula <- model_formula("target_factor", predictor_names)
  randomForest::randomForest(fit_formula, data = train_table, ntree = 200, classwt = class_weight_vector(train_table))
}


train_support_vector_machine <- function(train_table, predictor_names) {
  fit_formula <- model_formula("target_factor", predictor_names)
  e1071::svm(fit_formula, data = train_table, kernel = "radial", probability = TRUE, class.weights = class_weight_vector(train_table))
}


train_naive_bayes <- function(train_table, predictor_names) {
  balanced_table <- balanced_training_table(train_table)
  fit_formula <- model_formula("target_factor", predictor_names)
  e1071::naiveBayes(fit_formula, data = balanced_table)
}


train_all_models <- function(train_table, predictor_names) {
  list(
    logistic_regression = train_logistic_regression(train_table, predictor_names),
    decision_tree = train_decision_tree(train_table, predictor_names),
    random_forest = train_random_forest(train_table, predictor_names),
    support_vector_machine = train_support_vector_machine(train_table, predictor_names),
    naive_bayes = train_naive_bayes(train_table, predictor_names)
  )
}


safe_probability_column <- function(probability_table) {
  probability_table <- as.data.frame(probability_table)
  if ("class_1" %in% names(probability_table)) {
    return(probability_table[["class_1"]])
  }
  probability_table[[ncol(probability_table)]]
}


predict_logistic_scores <- function(model_object, data_table) {
  score_values <- suppressWarnings(predict(model_object, newdata = data_table, type = "response"))
  score_values[is.na(score_values)] <- 0
  score_values
}


predict_tree_scores <- function(model_object, data_table) {
  probability_table <- predict(model_object, newdata = data_table, type = "prob")
  safe_probability_column(probability_table)
}


predict_random_forest_scores <- function(model_object, data_table) {
  probability_table <- predict(model_object, newdata = data_table, type = "prob")
  safe_probability_column(probability_table)
}


predict_svm_scores <- function(model_object, data_table) {
  predicted_values <- predict(model_object, newdata = data_table, probability = TRUE)
  probability_table <- attr(predicted_values, "probabilities")
  safe_probability_column(probability_table)
}


predict_naive_bayes_scores <- function(model_object, data_table) {
  probability_table <- predict(model_object, newdata = data_table, type = "raw")
  safe_probability_column(probability_table)
}


predict_model_scores <- function(model_name, model_object, data_table) {
  if (model_name == "logistic_regression") {
    return(predict_logistic_scores(model_object, data_table))
  }
  if (model_name == "decision_tree") {
    return(predict_tree_scores(model_object, data_table))
  }
  predict_non_tree_scores(model_name, model_object, data_table)
}


predict_non_tree_scores <- function(model_name, model_object, data_table) {
  if (model_name == "random_forest") {
    return(predict_random_forest_scores(model_object, data_table))
  }
  if (model_name == "support_vector_machine") {
    return(predict_svm_scores(model_object, data_table))
  }
  predict_naive_bayes_scores(model_object, data_table)
}


safe_divide <- function(numerator, denominator) {
  if (denominator == 0) {
    return(0)
  }
  numerator / denominator
}


confusion_counts <- function(actual_values, predicted_values) {
  list(
    true_positive = sum(actual_values == 1 & predicted_values == 1),
    false_positive = sum(actual_values == 0 & predicted_values == 1),
    true_negative = sum(actual_values == 0 & predicted_values == 0),
    false_negative = sum(actual_values == 1 & predicted_values == 0)
  )
}


evaluate_predictions <- function(actual_values, predicted_values) {
  counts <- confusion_counts(actual_values, predicted_values)
  precision <- safe_divide(counts$true_positive, counts$true_positive + counts$false_positive)
  recall <- safe_divide(counts$true_positive, counts$true_positive + counts$false_negative)
  specificity <- safe_divide(counts$true_negative, counts$true_negative + counts$false_positive)
  build_metric_row(counts, precision, recall, specificity)
}


build_metric_row <- function(counts, precision, recall, specificity) {
  accuracy <- safe_divide(counts$true_positive + counts$true_negative, sum(unlist(counts)))
  f1_score <- safe_divide(2 * precision * recall, precision + recall)
  data.frame(accuracy = accuracy, precision = precision, recall = recall, specificity = specificity, balanced_accuracy = (recall + specificity) / 2, f1_score = f1_score)
}


auc_score <- function(actual_values, score_values) {
  positive_scores <- score_values[actual_values == 1]
  negative_scores <- score_values[actual_values == 0]
  if (length(positive_scores) == 0 || length(negative_scores) == 0) {
    return(0)
  }
  mean(outer(positive_scores, negative_scores, ">") + 0.5 * outer(positive_scores, negative_scores, "=="))
}


threshold_f1_score <- function(threshold_value, actual_values, score_values) {
  predictions <- ifelse(score_values >= threshold_value, 1, 0)
  evaluate_predictions(actual_values, predictions)$f1_score
}


best_threshold <- function(actual_values, score_values) {
  threshold_values <- seq(0.05, 0.95, by = 0.05)
  f1_scores <- sapply(threshold_values, threshold_f1_score, actual_values, score_values)
  threshold_values[which.max(f1_scores)]
}


build_model_metric_row <- function(model_name, actual_values, predictions, scores, threshold_value) {
  metrics <- evaluate_predictions(actual_values, predictions)
  metrics$model_name <- model_name
  metrics$auc <- auc_score(actual_values, scores)
  metrics$threshold <- threshold_value
  metrics$script_version <- SCRIPT_VERSION
  metrics[c("model_name", setdiff(names(metrics), "model_name"))]
}


evaluate_model_scores <- function(model_name, train_scores, test_scores, train_table, test_table) {
  threshold_value <- best_threshold(train_table$target_numeric, train_scores)
  test_predictions <- ifelse(test_scores >= threshold_value, 1, 0)
  build_model_metric_row(model_name, test_table$target_numeric, test_predictions, test_scores, threshold_value)
}


baseline_metrics <- function(test_table) {
  baseline_predictions <- rep(0, nrow(test_table))
  build_model_metric_row("majority_class_baseline", test_table$target_numeric, baseline_predictions, baseline_predictions, NA)
}


build_prediction_rows <- function(model_name, test_table, score_values, threshold_value) {
  predicted_values <- ifelse(score_values >= threshold_value, 1, 0)
  data.frame(canonical_entity_id = test_table[[ID_COLUMN]], model_name = model_name, actual = test_table$target_numeric, score = score_values, prediction = predicted_values)
}


model_threshold <- function(model_name, train_scores, train_table) {
  if (model_name == "majority_class_baseline") {
    return(NA)
  }
  best_threshold(train_table$target_numeric, train_scores)
}


evaluate_all_models <- function(trained_models, split_tables) {
  metric_rows <- list(baseline_metrics(split_tables$test))
  for (model_name in names(trained_models)) {
    metric_rows[[model_name]] <- evaluate_single_model(model_name, trained_models[[model_name]], split_tables)
  }
  do.call(rbind, metric_rows)
}


evaluate_single_model <- function(model_name, model_object, split_tables) {
  train_scores <- predict_model_scores(model_name, model_object, split_tables$train)
  test_scores <- predict_model_scores(model_name, model_object, split_tables$test)
  evaluate_model_scores(model_name, train_scores, test_scores, split_tables$train, split_tables$test)
}


prediction_tables_for_models <- function(trained_models, split_tables) {
  prediction_rows <- list(baseline_prediction_rows(split_tables$test))
  for (model_name in names(trained_models)) {
    prediction_rows[[model_name]] <- prediction_rows_for_model(model_name, trained_models[[model_name]], split_tables)
  }
  do.call(rbind, prediction_rows)
}


prediction_rows_for_model <- function(model_name, model_object, split_tables) {
  train_scores <- predict_model_scores(model_name, model_object, split_tables$train)
  test_scores <- predict_model_scores(model_name, model_object, split_tables$test)
  threshold_value <- model_threshold(model_name, train_scores, split_tables$train)
  build_prediction_rows(model_name, split_tables$test, test_scores, threshold_value)
}


baseline_prediction_rows <- function(test_table) {
  data.frame(canonical_entity_id = test_table[[ID_COLUMN]], model_name = "majority_class_baseline", actual = test_table$target_numeric, score = 0, prediction = 0)
}


screening_importance <- function(train_table, predictor_names) {
  positive_table <- train_table[train_table$target_numeric == 1, ]
  negative_table <- train_table[train_table$target_numeric == 0, ]
  sapply(predictor_names, function(feature_name) {
    abs(mean(positive_table[[feature_name]]) - mean(negative_table[[feature_name]]))
  })
}


importance_from_named_vector <- function(model_name, importance_values) {
  data.frame(model_name = model_name, feature_name = names(importance_values), importance = as.numeric(importance_values), script_version = SCRIPT_VERSION)
}


logistic_importance <- function(model_object) {
  coefficient_values <- abs(coef(model_object))
  coefficient_values <- coefficient_values[names(coefficient_values) != "(Intercept)"]
  importance_from_named_vector("logistic_regression", coefficient_values)
}


tree_importance <- function(model_object) {
  importance_values <- model_object$variable.importance
  importance_from_named_vector("decision_tree", importance_values)
}


forest_importance <- function(model_object) {
  importance_values <- randomForest::importance(model_object)[, 1]
  importance_from_named_vector("random_forest", importance_values)
}


generic_importance <- function(model_name, train_table, predictor_names) {
  importance_values <- screening_importance(train_table, predictor_names)
  importance_from_named_vector(model_name, importance_values)
}


build_feature_importance <- function(trained_models, train_table, predictor_names) {
  rows <- list(logistic_importance(trained_models$logistic_regression))
  rows$decision_tree <- tree_importance(trained_models$decision_tree)
  rows$random_forest <- forest_importance(trained_models$random_forest)
  rows$support_vector_machine <- generic_importance("support_vector_machine", train_table, predictor_names)
  rows$naive_bayes <- generic_importance("naive_bayes", train_table, predictor_names)
  do.call(rbind, rows)
}


selected_feature_table <- function(predictor_names) {
  data.frame(feature_name = predictor_names, script_version = SCRIPT_VERSION, stringsAsFactors = FALSE)
}


comparison_report <- function(metrics_table, predictor_names, input_file) {
  best_model <- metrics_table$model_name[which.max(metrics_table$f1_score)]
  data.frame(metric = c("input_file", "model_count", "feature_count", "best_model_by_f1", "script_version"), value = c(input_file, 6, length(predictor_names), best_model, SCRIPT_VERSION))
}


ensure_output_folder <- function() {
  if (!dir.exists(OUTPUT_FOLDER)) {
    dir.create(OUTPUT_FOLDER, recursive = TRUE)
  }
}


write_csv_safely <- function(data_table, file_name) {
  ensure_output_folder()
  write.csv(data_table, file.path(OUTPUT_FOLDER, file_name), row.names = FALSE)
}


write_outputs <- function(metrics_table, prediction_table, importance_table, feature_table, report_table, trained_models) {
  write_csv_safely(metrics_table, "dataset3_ml_all_model_metrics.csv")
  write_csv_safely(prediction_table, "dataset3_ml_all_predictions.csv")
  write_csv_safely(importance_table, "dataset3_ml_feature_importance.csv")
  write_csv_safely(feature_table, "dataset3_ml_selected_features.csv")
  write_csv_safely(report_table, "dataset3_ml_model_comparison_report.csv")
  saveRDS(trained_models, file.path(OUTPUT_FOLDER, "dataset3_ml_trained_models.rds"))
}


run_machine_learning_step <- function() {
  set.seed(42)
  install_missing_packages()
  input_file <- find_input_file()
  feature_table <- load_feature_table(input_file)
  model_table <- prepare_model_table(feature_table)
  predictor_names <- predictor_names_from_table(model_table)
  train_and_write_results(model_table, predictor_names, input_file)
}


train_and_write_results <- function(model_table, predictor_names, input_file) {
  split_tables <- make_stratified_split(model_table, 0.75)
  validate_split(split_tables)
  trained_models <- train_all_models(split_tables$train, predictor_names)
  metrics_table <- evaluate_all_models(trained_models, split_tables)
  prediction_table <- prediction_tables_for_models(trained_models, split_tables)
  importance_table <- build_feature_importance(trained_models, split_tables$train, predictor_names)
  write_outputs(metrics_table, prediction_table, importance_table, selected_feature_table(predictor_names), comparison_report(metrics_table, predictor_names, input_file), trained_models)
  print(metrics_table)
}


build_sample_feature_table <- function() {
  data.frame(canonical_entity_id = c("A", "B", "C", "D"), artefact_count = c(10, 1, 8, 0), degree_total = c(50, 2, 40, 0), influence_score = c(90, 1, 80, 0), medium_or_high_influence = c(1, 0, 1, 0), stringsAsFactors = FALSE)
}


test_sunny_day_model_table <- function() {
  sample_table <- build_sample_feature_table()
  model_table <- prepare_model_table(sample_table)
  # Test 1 (Sunny Day): normal feature rows should become a model table because training depends on this table.
  stopifnot(nrow(model_table) == 4)
  print("PASS: Sunny day test passed because the model table is created.")
}


test_edge_case_constant_column <- function() {
  sample_table <- build_sample_feature_table()
  sample_table$constant_feature <- 1
  model_table <- prepare_model_table(sample_table)
  # Test 2 (Edge Case): constant columns are removed because they cannot help a classifier learn.
  stopifnot(!"constant_feature" %in% names(model_table))
  print("PASS: Edge case test passed because constant features are removed.")
}


test_weird_gotcha_leakage <- function() {
  sample_table <- build_sample_feature_table()
  model_table <- prepare_model_table(sample_table)
  # Test 3 (Weird Gotcha): influence_score is removed because it would leak the answer.
  stopifnot(!"influence_score" %in% names(model_table))
  print("PASS: Weird gotcha test passed because leakage columns are removed.")
}


test_bug_catcher_metric_safety <- function() {
  actual_values <- c(1, 1, 0, 0)
  predicted_values <- c(0, 0, 0, 0)
  # Test 4 (Bug Catcher): precision must not crash when a model predicts no positive cases.
  stopifnot(evaluate_predictions(actual_values, predicted_values)$precision == 0)
  print("PASS: Bug catcher test passed because zero-positive predictions are safe.")
}


run_self_tests <- function() {
  test_sunny_day_model_table()
  test_edge_case_constant_column()
  test_weird_gotcha_leakage()
  test_bug_catcher_metric_safety()
}


if (sys.nframe() == 0) {
  run_self_tests()
  run_machine_learning_step()
}
