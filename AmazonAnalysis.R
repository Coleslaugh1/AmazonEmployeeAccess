
# Setting up work environment and libraries -------------------------------

setwd(dir = "C:/Users/camer/Documents/Stat 348/AmazonEmployeeAccess/")

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
  step_other(all_predictors(), -all_outcomes(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

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
