
# Setting up work environment and libraries -------------------------------

#setwd(dir = "C:/Users/camer/Documents/Stat 348/AmazonEmployeeAccess/")

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)

#Initial reading of the data

rawdata <- vroom(file = "train.csv")
test_input <- vroom(file = "test.csv")

my_recipe <- recipe(ACTION ~ ., data = rawdata) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  update_role(ACTION, new_role = "outcome") %>% 
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>% 
  step_other(all_predictors(), -all_outcomes(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
  # step_dummy(all_nominal_predictors())

prep_recipe <- prep(my_recipe)
baked_data <- bake(prep_recipe, new_data = rawdata)

# Write and read function

format_and_write <- function(predictions){
  final_preds <- predictions %>% 
    mutate(Action = .pred_1) %>% 
    mutate(Id = c(1:length(.pred_1))) %>% 
    select(Id, Action)
  
  vroom_write(final_preds,"preds.csv",delim = ",")
}

# Logistic Regression -----------------------------------------------------

log_mod <- logistic_reg() %>%
  set_engine("glm")

log_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_mod) %>%
  fit(data = rawdata)

log_predictions <- predict(log_workflow,
                              new_data=test_input,
                              type="prob")

format_and_write(log_predictions)

# Penalized Logistic Regression ------------------------------------------

pog_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
  set_engine("glmnet")

pog_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pog_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 4)

folds <- vfold_cv(rawdata, v = 10, repeats=1)

CV_results <- pog_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_pog_wf <-
  pog_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=rawdata)


pog_predictions <- final_pog_wf %>%
  predict(new_data = test_input, type="prob")

format_and_write(pog_predictions)
