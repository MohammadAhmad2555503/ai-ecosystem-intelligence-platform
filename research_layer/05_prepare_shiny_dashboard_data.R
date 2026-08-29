#!/usr/bin/env Rscript

# Step 5: Prepare R Shiny dashboard data.
# This version matches your actual folder structure:
# research_layer/05_prepare_shiny_dashboard_data.R
# research_layer/outputs/

SCRIPT_VERSION <- "research_layer_step05_prepare_shiny_dashboard_data_v4_root_final"

REQUIRED_FILES <- c(
  "dataset3_combined_model_metrics.csv",
  "dataset3_overall_model_ranking.csv",
  "dataset3_ml_vs_dl_family_summary.csv",
  "dataset3_best_model_summary.csv",
  "dataset3_combined_model_predictions.csv"
)

METRIC_NAMES <- c(
  "accuracy",
  "precision",
  "recall",
  "specificity",
  "balanced_accuracy",
  "f1_score",
  "auc"
)

IMPORTANCE_FILE <- "dataset3_ml_feature_importance.csv"
ORGANISATION_FILE <- "dataset3_clean_research_table.csv"


get_script_folder <- function() {
  arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- grep("--file=", arguments, value = TRUE)
  
  if (length(file_argument) == 0) {
    return(normalizePath(getwd(), winslash = "/"))
  }
  
  normalizePath(dirname(sub("--file=", "", file_argument[[1]])), winslash = "/")
}


project_folder_candidates <- function() {
  script_folder <- get_script_folder()
  
  unique(c(
    script_folder,
    getwd(),
    dirname(getwd())
  ))
}


folder_has_outputs <- function(folder_path) {
  outputs_path <- file.path(folder_path, "outputs")
  all(file.exists(file.path(outputs_path, REQUIRED_FILES)))
}


find_project_folder <- function() {
  candidates <- project_folder_candidates()
  useful_folders <- candidates[sapply(candidates, folder_has_outputs)]
  
  if (length(useful_folders) == 0) {
    stop("Could not find research_layer folder with the required outputs files.")
  }
  
  normalizePath(useful_folders[[1]], winslash = "/")
}


outputs_folder <- function(project_folder) {
  file.path(project_folder, "outputs")
}


dashboard_folder <- function(project_folder) {
  file.path(project_folder, "shiny_dashboard", "dashboard_data")
}


ensure_folder <- function(folder_path) {
  if (!dir.exists(folder_path)) {
    dir.create(folder_path, recursive = TRUE)
  }
}


read_csv_safely <- function(file_path) {
  tryCatch(
    read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(error_details) {
      stop(paste("Could not read file:", file_path, error_details$message))
    }
  )
}


write_csv_safely <- function(data_table, folder_path, file_name) {
  ensure_folder(folder_path)
  write.csv(data_table, file.path(folder_path, file_name), row.names = FALSE)
}


required_path <- function(outputs_path, file_name) {
  file.path(outputs_path, file_name)
}


optional_path <- function(outputs_path, file_name) {
  target_path <- file.path(outputs_path, file_name)
  
  if (!file.exists(target_path)) {
    return(NA_character_)
  }
  
  target_path
}


read_optional_csv <- function(outputs_path, file_name) {
  target_path <- optional_path(outputs_path, file_name)
  
  if (is.na(target_path)) {
    return(data.frame())
  }
  
  read_csv_safely(target_path)
}


validate_columns <- function(data_table, required_columns, table_name) {
  missing_columns <- setdiff(required_columns, names(data_table))
  
  if (length(missing_columns) > 0) {
    stop(paste(table_name, "missing:", paste(missing_columns, collapse = ", ")))
  }
  
  TRUE
}


coerce_numeric_column <- function(data_table, column_name) {
  if (column_name %in% names(data_table)) {
    data_table[[column_name]] <- suppressWarnings(as.numeric(data_table[[column_name]]))
  }
  
  data_table
}


coerce_metric_columns <- function(metrics_table) {
  for (column_name in c(METRIC_NAMES, "threshold", "overall_rank")) {
    metrics_table <- coerce_numeric_column(metrics_table, column_name)
  }
  
  metrics_table
}


normalise_model_name <- function(model_name) {
  tools::toTitleCase(gsub("_", " ", model_name))
}


model_type_from_name <- function(model_name) {
  if (grepl("baseline", model_name, ignore.case = TRUE)) {
    return("baseline")
  }
  
  "model"
}


fallback_method_family <- function(data_table) {
  if ("method_family" %in% names(data_table)) {
    return(data_table$method_family)
  }
  
  rep("Unknown Family", nrow(data_table))
}


fallback_source_stage <- function(data_table) {
  if ("source_stage" %in% names(data_table)) {
    return(data_table$source_stage)
  }
  
  rep("Unknown Stage", nrow(data_table))
}


fallback_model_type <- function(data_table) {
  if ("model_type" %in% names(data_table)) {
    return(data_table$model_type)
  }
  
  sapply(data_table$model_name, model_type_from_name)
}


make_display_names <- function(data_table) {
  paste(
    data_table$method_family,
    normalise_model_name(data_table$model_name),
    sep = " - "
  )
}


make_model_keys <- function(data_table) {
  paste(
    data_table$method_family,
    data_table$source_stage,
    data_table$model_name,
    sep = "__"
  )
}


add_model_metadata <- function(data_table) {
  data_table$method_family <- fallback_method_family(data_table)
  data_table$source_stage <- fallback_source_stage(data_table)
  data_table$model_type <- fallback_model_type(data_table)
  data_table$display_model_name <- make_display_names(data_table)
  data_table$model_key <- make_model_keys(data_table)
  
  data_table
}


prepare_metric_table <- function(metrics_table) {
  validate_columns(metrics_table, c("model_name", "f1_score", "auc"), "metrics")
  metrics_table <- coerce_metric_columns(metrics_table)
  add_model_metadata(metrics_table)
}


prepare_prediction_table <- function(prediction_table) {
  validate_columns(prediction_table, c("model_name", "actual", "prediction"), "predictions")
  prediction_table$actual <- suppressWarnings(as.integer(prediction_table$actual))
  prediction_table$prediction <- suppressWarnings(as.integer(prediction_table$prediction))
  
  add_model_metadata(prediction_table)
}


read_metrics_table <- function(outputs_path) {
  file_path <- required_path(outputs_path, "dataset3_combined_model_metrics.csv")
  prepare_metric_table(read_csv_safely(file_path))
}


read_ranking_table <- function(outputs_path) {
  file_path <- required_path(outputs_path, "dataset3_overall_model_ranking.csv")
  prepare_metric_table(read_csv_safely(file_path))
}


read_family_table <- function(outputs_path) {
  file_path <- required_path(outputs_path, "dataset3_ml_vs_dl_family_summary.csv")
  read_csv_safely(file_path)
}


read_best_table <- function(outputs_path) {
  file_path <- required_path(outputs_path, "dataset3_best_model_summary.csv")
  read_csv_safely(file_path)
}


read_prediction_table <- function(outputs_path) {
  file_path <- required_path(outputs_path, "dataset3_combined_model_predictions.csv")
  prepare_prediction_table(read_csv_safely(file_path))
}


read_required_tables <- function(outputs_path) {
  list(
    metrics = read_metrics_table(outputs_path),
    ranking = read_ranking_table(outputs_path),
    family = read_family_table(outputs_path),
    best = read_best_table(outputs_path),
    predictions = read_prediction_table(outputs_path)
  )
}


read_optional_tables <- function(outputs_path) {
  list(
    importance = read_optional_csv(outputs_path, IMPORTANCE_FILE),
    organisations = read_optional_csv(outputs_path, ORGANISATION_FILE)
  )
}


real_model_rows <- function(metrics_table) {
  metrics_table[metrics_table$model_type == "model", , drop = FALSE]
}


ranking_order <- function(ranking_table) {
  if ("overall_rank" %in% names(ranking_table)) {
    return(order(ranking_table$overall_rank))
  }
  
  order(-ranking_table$f1_score, -ranking_table$auc)
}


best_model_row <- function(ranking_table) {
  real_models <- real_model_rows(ranking_table)
  ordered_rows <- ranking_order(real_models)
  
  real_models[ordered_rows[[1]], , drop = FALSE]
}


kpi_row <- function(metric_name, metric_value) {
  data.frame(metric = metric_name, value = as.character(metric_value))
}


build_dashboard_kpis <- function(tables) {
  best_model <- best_model_row(tables$ranking)
  
  rbind(
    kpi_row("total_model_rows", nrow(tables$metrics)),
    kpi_row("real_model_count", nrow(real_model_rows(tables$metrics))),
    kpi_row("baseline_count", sum(tables$metrics$model_type == "baseline")),
    kpi_row("best_model", best_model$display_model_name[[1]]),
    kpi_row("best_model_f1_score", round(best_model$f1_score[[1]], 6)),
    kpi_row("best_model_auc", round(best_model$auc[[1]], 6)),
    kpi_row("script_version", SCRIPT_VERSION)
  )
}


metric_long_row <- function(metrics_table, row_index, metric_name) {
  data.frame(
    model_key = metrics_table$model_key[[row_index]],
    model_name = metrics_table$model_name[[row_index]],
    display_model_name = metrics_table$display_model_name[[row_index]],
    method_family = metrics_table$method_family[[row_index]],
    source_stage = metrics_table$source_stage[[row_index]],
    model_type = metrics_table$model_type[[row_index]],
    metric_name = metric_name,
    metric_value = metrics_table[[metric_name]][[row_index]]
  )
}


metric_rows_for_model <- function(metrics_table, row_index) {
  rows <- lapply(METRIC_NAMES, function(metric_name) {
    metric_long_row(metrics_table, row_index, metric_name)
  })
  
  do.call(rbind, rows)
}


build_metric_long_table <- function(metrics_table) {
  metrics_table <- prepare_metric_table(metrics_table)
  
  rows <- lapply(seq_len(nrow(metrics_table)), function(row_index) {
    metric_rows_for_model(metrics_table, row_index)
  })
  
  do.call(rbind, rows)
}


count_confusion_type <- function(actual_values, predicted_values, actual_target, predicted_target) {
  sum(actual_values == actual_target & predicted_values == predicted_target)
}


confusion_row <- function(prediction_table) {
  actual_values <- as.integer(prediction_table$actual)
  predicted_values <- as.integer(prediction_table$prediction)
  
  data.frame(
    model_key = prediction_table$model_key[[1]],
    model_name = prediction_table$model_name[[1]],
    display_model_name = prediction_table$display_model_name[[1]],
    method_family = prediction_table$method_family[[1]],
    source_stage = prediction_table$source_stage[[1]],
    model_type = prediction_table$model_type[[1]],
    true_positive = count_confusion_type(actual_values, predicted_values, 1, 1),
    false_positive = count_confusion_type(actual_values, predicted_values, 0, 1),
    true_negative = count_confusion_type(actual_values, predicted_values, 0, 0),
    false_negative = count_confusion_type(actual_values, predicted_values, 1, 0)
  )
}


build_confusion_summary <- function(prediction_table) {
  prediction_table <- prepare_prediction_table(prediction_table)
  split_tables <- split(prediction_table, prediction_table$model_key)
  rows <- lapply(split_tables, confusion_row)
  
  do.call(rbind, rows)
}


prediction_distribution_row <- function(prediction_table) {
  data.frame(
    model_key = prediction_table$model_key[[1]],
    model_name = prediction_table$model_name[[1]],
    display_model_name = prediction_table$display_model_name[[1]],
    method_family = prediction_table$method_family[[1]],
    source_stage = prediction_table$source_stage[[1]],
    model_type = prediction_table$model_type[[1]],
    actual_positive = sum(prediction_table$actual == 1),
    predicted_positive = sum(prediction_table$prediction == 1),
    predicted_negative = sum(prediction_table$prediction == 0)
  )
}


build_prediction_distribution <- function(prediction_table) {
  prediction_table <- prepare_prediction_table(prediction_table)
  split_tables <- split(prediction_table, prediction_table$model_key)
  rows <- lapply(split_tables, prediction_distribution_row)
  
  do.call(rbind, rows)
}


empty_importance_table <- function() {
  data.frame(
    model_name = character(),
    feature_name = character(),
    importance = numeric(),
    script_version = character()
  )
}


importance_table_is_valid <- function(importance_table) {
  all(c("model_name", "feature_name", "importance") %in% names(importance_table))
}


clean_importance_table <- function(importance_table) {
  if (nrow(importance_table) == 0 || !importance_table_is_valid(importance_table)) {
    return(empty_importance_table())
  }
  
  importance_table$importance <- suppressWarnings(as.numeric(importance_table$importance))
  importance_table[order(-importance_table$importance), , drop = FALSE]
}


top_feature_rows <- function(importance_table, row_limit) {
  cleaned_table <- clean_importance_table(importance_table)
  
  if (nrow(cleaned_table) == 0) {
    return(cleaned_table)
  }
  
  head(cleaned_table, row_limit)
}


top_features_by_model <- function(importance_table, row_limit) {
  cleaned_table <- clean_importance_table(importance_table)
  
  if (nrow(cleaned_table) == 0) {
    return(cleaned_table)
  }
  
  rows <- lapply(split(cleaned_table, cleaned_table$model_name), head, row_limit)
  do.call(rbind, rows)
}


first_existing_column <- function(data_table, preferred_columns) {
  matching_columns <- intersect(preferred_columns, names(data_table))
  
  if (length(matching_columns) == 0) {
    return(NA_character_)
  }
  
  matching_columns[[1]]
}


column_or_default <- function(data_table, column_name, default_value) {
  if (is.na(column_name)) {
    return(rep(default_value, nrow(data_table)))
  }
  
  data_table[[column_name]]
}


organisation_id_column <- function(data_table) {
  first_existing_column(data_table, c("canonical_entity_id", "canonical_org_id", "org_id", "id"))
}


organisation_name_column <- function(data_table) {
  first_existing_column(data_table, c("canonical_name", "organisation_name", "org_name", "name"))
}


organisation_score_column <- function(data_table) {
  first_existing_column(data_table, c("influence_score", "final_influence_score", "score"))
}


organisation_band_column <- function(data_table) {
  first_existing_column(data_table, c("influence_band", "band", "influence_category"))
}


organisation_rank_column <- function(data_table) {
  first_existing_column(data_table, c("rank_overall", "overall_rank", "rank"))
}


empty_top_organisations_table <- function() {
  data.frame(
    canonical_entity_id = character(),
    organisation_name = character(),
    influence_score = numeric(),
    influence_band = character(),
    rank_overall = integer()
  )
}


build_standard_organisation_table <- function(organisation_table) {
  data.frame(
    canonical_entity_id = as.character(column_or_default(organisation_table, organisation_id_column(organisation_table), "")),
    organisation_name = as.character(column_or_default(organisation_table, organisation_name_column(organisation_table), "")),
    influence_score = suppressWarnings(as.numeric(column_or_default(organisation_table, organisation_score_column(organisation_table), 0))),
    influence_band = as.character(column_or_default(organisation_table, organisation_band_column(organisation_table), "")),
    rank_overall = suppressWarnings(as.integer(column_or_default(organisation_table, organisation_rank_column(organisation_table), NA)))
  )
}


standardise_organisation_table <- function(organisation_table) {
  if (nrow(organisation_table) == 0) {
    return(empty_top_organisations_table())
  }
  
  if (is.na(organisation_score_column(organisation_table))) {
    return(empty_top_organisations_table())
  }
  
  build_standard_organisation_table(organisation_table)
}


top_organisations <- function(organisation_table) {
  standard_table <- standardise_organisation_table(organisation_table)
  
  if (nrow(standard_table) == 0) {
    return(standard_table)
  }
  
  ordered_table <- standard_table[order(-standard_table$influence_score), , drop = FALSE]
  head(ordered_table, 25)
}


manifest_status <- function(data_table) {
  if (nrow(data_table) == 0) {
    return("created_empty")
  }
  
  "created"
}


manifest_row <- function(file_name, data_table) {
  data.frame(
    file_name = file_name,
    status = manifest_status(data_table),
    row_count = nrow(data_table),
    script_version = SCRIPT_VERSION
  )
}


manifest_self_row <- function(current_manifest) {
  data.frame(
    file_name = "dashboard_manifest.csv",
    status = "created",
    row_count = nrow(current_manifest) + 1,
    script_version = SCRIPT_VERSION
  )
}


build_manifest <- function(output_tables) {
  rows <- lapply(names(output_tables), function(file_name) {
    manifest_row(file_name, output_tables[[file_name]])
  })
  
  manifest <- do.call(rbind, rows)
  rbind(manifest, manifest_self_row(manifest))
}


build_output_tables <- function(tables, optional_tables) {
  list(
    "dashboard_kpis.csv" = build_dashboard_kpis(tables),
    "model_metric_long_table.csv" = build_metric_long_table(tables$metrics),
    "model_ranking_table.csv" = tables$ranking,
    "family_summary_table.csv" = tables$family,
    "best_model_summary_table.csv" = tables$best,
    "confusion_summary_table.csv" = build_confusion_summary(tables$predictions),
    "prediction_distribution_table.csv" = build_prediction_distribution(tables$predictions),
    "top_feature_importance_table.csv" = top_feature_rows(optional_tables$importance, 30),
    "feature_importance_by_model_table.csv" = top_features_by_model(optional_tables$importance, 10),
    "top_organisations_table.csv" = top_organisations(optional_tables$organisations)
  )
}


write_output_tables <- function(output_folder, output_tables) {
  for (file_name in names(output_tables)) {
    write_csv_safely(output_tables[[file_name]], output_folder, file_name)
  }
}


clear_old_dashboard_files <- function(output_folder) {
  ensure_folder(output_folder)
  old_files <- list.files(output_folder, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(old_files) > 0) {
    unlink(old_files, force = TRUE)
  }
}


run_dashboard_data_step <- function() {
  project_path <- find_project_folder()
  outputs_path <- outputs_folder(project_path)
  dashboard_path <- dashboard_folder(project_path)
  
  clear_old_dashboard_files(dashboard_path)
  
  tables <- read_required_tables(outputs_path)
  optional_tables <- read_optional_tables(outputs_path)
  output_tables <- build_output_tables(tables, optional_tables)
  output_tables[["dashboard_manifest.csv"]] <- build_manifest(output_tables)
  
  write_output_tables(dashboard_path, output_tables)
  print(output_tables[["dashboard_manifest.csv"]])
}


build_sample_metrics <- function() {
  data.frame(
    model_name = c("model_a", "majority_class_baseline"),
    method_family = c("Machine Learning", "Machine Learning"),
    source_stage = c("R", "R"),
    model_type = c("model", "baseline"),
    accuracy = c(0.9, 0.8),
    precision = c(0.7, 0),
    recall = c(0.6, 0),
    specificity = c(0.9, 1),
    balanced_accuracy = c(0.75, 0.5),
    f1_score = c(0.65, 0),
    auc = c(0.8, 0.5),
    overall_rank = c(1, 2)
  )
}


build_sample_predictions <- function() {
  data.frame(
    model_name = c("majority_class_baseline", "majority_class_baseline"),
    method_family = c("Machine Learning", "Deep Learning"),
    source_stage = c("R", "Python"),
    actual = c(0, 1),
    prediction = c(0, 0)
  )
}


build_sample_organisations <- function() {
  data.frame(
    canonical_name = c("Org B", "Org A"),
    influence_score = c(10, 20),
    influence_band = c("medium", "high")
  )
}


test_sunny_day_metric_long <- function() {
  sample_table <- build_sample_metrics()
  
  # Test 1: two models should expand across seven dashboard metrics.
  stopifnot(nrow(build_metric_long_table(sample_table)) == 14)
  
  print("PASS: Sunny day test passed because metric table expands correctly.")
}


test_edge_case_baseline_separation <- function() {
  confusion_table <- build_confusion_summary(build_sample_predictions())
  
  # Test 2: same baseline name must remain separate for ML and DL.
  stopifnot(nrow(confusion_table) == 2)
  
  print("PASS: Edge case test passed because duplicate baselines are separated.")
}


test_weird_gotcha_top_organisations <- function() {
  organisation_table <- top_organisations(build_sample_organisations())
  
  # Test 3: dashboard organisations must be sorted by influence score.
  stopifnot(organisation_table$organisation_name[[1]] == "Org A")
  
  print("PASS: Weird gotcha test passed because organisations are sorted correctly.")
}


test_bug_catcher_missing_columns <- function() {
  broken_table <- data.frame(model_name = "model_a")
  
  # Test 4: bad prediction files must fail before dashboard files are written.
  stopifnot(inherits(try(build_confusion_summary(broken_table), silent = TRUE), "try-error"))
  
  print("PASS: Bug catcher test passed because missing columns are detected.")
}


run_self_tests <- function() {
  test_sunny_day_metric_long()
  test_edge_case_baseline_separation()
  test_weird_gotcha_top_organisations()
  test_bug_catcher_missing_columns()
}


run_self_tests()
run_dashboard_data_step()
