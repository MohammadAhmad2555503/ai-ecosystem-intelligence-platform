#!/usr/bin/env Rscript

# Step 4: Combined ML vs DL comparison.
# This script combines R ML and Python DL results into dissertation-ready tables.

SCRIPT_VERSION <- "research_layer_step04_ml_dl_comparison_v1"

ML_METRICS_FILE <- "dataset3_ml_all_model_metrics.csv"
DL_METRICS_FILE <- "dataset3_dl_all_model_metrics.csv"

ML_PREDICTIONS_FILE <- "dataset3_ml_all_predictions.csv"
DL_PREDICTIONS_FILE <- "dataset3_dl_all_predictions.csv"

METRIC_COLUMNS <- c(
  "accuracy",
  "precision",
  "recall",
  "specificity",
  "balanced_accuracy",
  "f1_score",
  "auc",
  "threshold"
)


get_script_folder <- function() {
  command_arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- grep("--file=", command_arguments, value = TRUE)
  
  if (length(file_argument) == 0) {
    return(getwd())
  }
  
  normalizePath(dirname(sub("--file=", "", file_argument[[1]])), winslash = "/")
}


file_candidates <- function(file_name) {
  script_folder <- get_script_folder()
  
  c(
    file.path(script_folder, "..", "outputs", file_name),
    file.path(script_folder, "outputs", file_name),
    file.path(getwd(), "outputs", file_name),
    file.path(getwd(), "research_layer", "outputs", file_name),
    file.path(getwd(), file_name)
  )
}


find_required_file <- function(file_name) {
  candidate_paths <- file_candidates(file_name)
  existing_paths <- candidate_paths[file.exists(candidate_paths)]
  
  if (length(existing_paths) == 0) {
    stop(paste("Could not find required file:", file_name))
  }
  
  normalizePath(existing_paths[[1]], winslash = "/")
}


read_csv_safely <- function(file_path) {
  tryCatch(
    read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(read_error) {
      stop(paste("Could not read file:", file_path, read_error$message))
    }
  )
}


coerce_metric_columns <- function(metrics_table) {
  for (column_name in intersect(METRIC_COLUMNS, names(metrics_table))) {
    metrics_table[[column_name]] <- suppressWarnings(as.numeric(metrics_table[[column_name]]))
  }
  
  metrics_table
}


validate_metric_columns <- function(metrics_table) {
  required_columns <- c("model_name", "accuracy", "precision", "recall", "f1_score", "auc")
  missing_columns <- setdiff(required_columns, names(metrics_table))
  
  if (length(missing_columns) > 0) {
    stop(paste("Missing metric columns:", paste(missing_columns, collapse = ", ")))
  }
  
  TRUE
}


model_type_from_name <- function(model_name) {
  if (grepl("baseline", model_name, ignore.case = TRUE)) {
    return("baseline")
  }
  
  "model"
}


normalise_model_name <- function(model_name) {
  cleaned_name <- gsub("_", " ", model_name)
  tools::toTitleCase(cleaned_name)
}


display_model_name <- function(method_family, model_name) {
  paste(method_family, normalise_model_name(model_name), sep = " - ")
}


add_model_metadata <- function(metrics_table, method_family, source_stage) {
  metrics_table$method_family <- method_family
  metrics_table$source_stage <- source_stage
  metrics_table$model_type <- sapply(metrics_table$model_name, model_type_from_name)
  metrics_table$display_model_name <- mapply(display_model_name, method_family, metrics_table$model_name)
  
  metrics_table
}


prepare_metrics_table <- function(file_path, method_family, source_stage) {
  metrics_table <- read_csv_safely(file_path)
  validate_metric_columns(metrics_table)
  metrics_table <- coerce_metric_columns(metrics_table)
  add_model_metadata(metrics_table, method_family, source_stage)
}


combine_metric_tables <- function(ml_file, dl_file) {
  ml_metrics <- prepare_metrics_table(ml_file, "Machine Learning", "R")
  dl_metrics <- prepare_metrics_table(dl_file, "Deep Learning", "Python")
  
  rbind(ml_metrics, dl_metrics)
}


real_model_rows <- function(metrics_table) {
  metrics_table[metrics_table$model_type == "model", , drop = FALSE]
}


rank_metric_table <- function(metrics_table) {
  ordered_rows <- order(
    -metrics_table$f1_score,
    -metrics_table$auc,
    -metrics_table$balanced_accuracy
  )
  
  ranked_table <- metrics_table[ordered_rows, , drop = FALSE]
  ranked_table$overall_rank <- seq_len(nrow(ranked_table))
  ranked_table
}


best_real_model <- function(ranked_table) {
  real_models <- real_model_rows(ranked_table)
  real_models[which.min(real_models$overall_rank), , drop = FALSE]
}


family_summary_row <- function(family_table) {
  data.frame(
    method_family = family_table$method_family[[1]],
    model_count = nrow(family_table),
    best_f1_score = max(family_table$f1_score),
    mean_f1_score = mean(family_table$f1_score),
    mean_auc = mean(family_table$auc),
    mean_balanced_accuracy = mean(family_table$balanced_accuracy)
  )
}


family_summary_table <- function(metrics_table) {
  real_models <- real_model_rows(metrics_table)
  family_tables <- split(real_models, real_models$method_family)
  
  do.call(rbind, lapply(family_tables, family_summary_row))
}


build_best_model_summary <- function(ranked_table) {
  best_model <- best_real_model(ranked_table)
  
  data.frame(
    metric = c("best_model", "method_family", "f1_score", "auc", "script_version"),
    value = c(
      best_model$display_model_name,
      best_model$method_family,
      best_model$f1_score,
      best_model$auc,
      SCRIPT_VERSION
    )
  )
}


build_step4_report <- function(combined_table, ranked_table) {
  best_model <- best_real_model(ranked_table)
  
  data.frame(
    metric = c("total_rows", "real_model_count", "baseline_count", "best_model_by_f1", "script_version"),
    value = c(
      nrow(combined_table),
      nrow(real_model_rows(combined_table)),
      sum(combined_table$model_type == "baseline"),
      best_model$display_model_name,
      SCRIPT_VERSION
    )
  )
}


prediction_file_exists <- function(file_name) {
  length(file_candidates(file_name)[file.exists(file_candidates(file_name))]) > 0
}


read_prediction_table <- function(file_name, method_family, source_stage) {
  file_path <- find_required_file(file_name)
  prediction_table <- read_csv_safely(file_path)
  add_prediction_metadata(prediction_table, method_family, source_stage)
}


add_prediction_metadata <- function(prediction_table, method_family, source_stage) {
  prediction_table$method_family <- method_family
  prediction_table$source_stage <- source_stage
  prediction_table$model_type <- sapply(prediction_table$model_name, model_type_from_name)
  
  prediction_table
}


combine_prediction_tables <- function() {
  ml_predictions <- read_prediction_table(ML_PREDICTIONS_FILE, "Machine Learning", "R")
  dl_predictions <- read_prediction_table(DL_PREDICTIONS_FILE, "Deep Learning", "Python")
  
  rbind(ml_predictions, dl_predictions)
}


ensure_output_folder <- function(output_folder) {
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
}


write_csv_safely <- function(data_table, output_folder, file_name) {
  ensure_output_folder(output_folder)
  write.csv(data_table, file.path(output_folder, file_name), row.names = FALSE)
}


write_prediction_outputs <- function(output_folder) {
  combined_predictions <- combine_prediction_tables()
  write_csv_safely(combined_predictions, output_folder, "dataset3_combined_model_predictions.csv")
}


write_all_outputs <- function(output_folder, output_bundle) {
  write_csv_safely(output_bundle$combined, output_folder, "dataset3_combined_model_metrics.csv")
  write_csv_safely(output_bundle$ranking, output_folder, "dataset3_overall_model_ranking.csv")
  write_csv_safely(output_bundle$families, output_folder, "dataset3_ml_vs_dl_family_summary.csv")
  write_csv_safely(output_bundle$best, output_folder, "dataset3_best_model_summary.csv")
  write_csv_safely(output_bundle$report, output_folder, "dataset3_step4_comparison_report.csv")
  write_prediction_outputs(output_folder)
}


build_output_bundle <- function(combined_table) {
  ranked_table <- rank_metric_table(combined_table)
  
  list(
    combined = combined_table,
    ranking = ranked_table,
    families = family_summary_table(combined_table),
    best = build_best_model_summary(ranked_table),
    report = build_step4_report(combined_table, ranked_table)
  )
}


output_folder_from_file <- function(file_path) {
  dirname(file_path)
}


run_comparison_step <- function() {
  ml_file <- find_required_file(ML_METRICS_FILE)
  dl_file <- find_required_file(DL_METRICS_FILE)
  combined_table <- combine_metric_tables(ml_file, dl_file)
  output_bundle <- build_output_bundle(combined_table)
  
  write_all_outputs(output_folder_from_file(ml_file), output_bundle)
  print(output_bundle$ranking)
}


build_sample_metrics <- function() {
  data.frame(
    model_name = c("model_a", "majority_class_baseline", "model_b"),
    accuracy = c(0.9, 0.8, 0.95),
    precision = c(0.7, 0, 0.8),
    recall = c(0.6, 0, 0.9),
    specificity = c(0.9, 1, 0.95),
    balanced_accuracy = c(0.75, 0.5, 0.925),
    f1_score = c(0.65, 0, 0.85),
    auc = c(0.8, 0.5, 0.9)
  )
}


test_sunny_day_ranking <- function() {
  sample_table <- add_model_metadata(build_sample_metrics(), "Machine Learning", "R")
  
  # Test 1 (Sunny Day): ranking should place the best F1-score model first.
  stopifnot(rank_metric_table(sample_table)$model_name[[1]] == "model_b")
  
  print("PASS: Sunny day test passed because ranking uses F1-score correctly.")
}


test_edge_case_baseline_count <- function() {
  sample_table <- add_model_metadata(build_sample_metrics(), "Machine Learning", "R")
  
  # Test 2 (Edge Case): baseline rows must be labelled separately from real models.
  stopifnot(sum(sample_table$model_type == "baseline") == 1)
  
  print("PASS: Edge case test passed because baselines are labelled correctly.")
}


test_weird_gotcha_best_model <- function() {
  sample_table <- add_model_metadata(build_sample_metrics(), "Machine Learning", "R")
  ranked_table <- rank_metric_table(sample_table)
  
  # Test 3 (Weird Gotcha): the best model summary must ignore baseline-only logic.
  stopifnot(best_real_model(ranked_table)$model_name[[1]] == "model_b")
  
  print("PASS: Weird gotcha test passed because best model ignores baselines.")
}


test_bug_catcher_missing_columns <- function() {
  broken_table <- data.frame(model_name = "model_a", accuracy = 1)
  
  # Test 4 (Bug Catcher): missing metric columns should fail before bad files are written.
  stopifnot(inherits(try(validate_metric_columns(broken_table), silent = TRUE), "try-error"))
  
  print("PASS: Bug catcher test passed because missing columns are detected.")
}


run_self_tests <- function() {
  test_sunny_day_ranking()
  test_edge_case_baseline_count()
  test_weird_gotcha_best_model()
  test_bug_catcher_missing_columns()
}


if (sys.nframe() == 0) {
  run_self_tests()
  run_comparison_step()
}
