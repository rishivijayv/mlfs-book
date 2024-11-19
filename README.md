# Predicting Air Quality (PM2.5) in Downtown Toronto

## Introduction
In this lab, we utilized weather forecast data (such as precipitation, wind speed, wind direction, etc) to predict PM2.5 levels in the air in Downtown Toronto. We also added an additional feature of the **3 day lagging mean of PM2.5** to try and improve our prediction model. In this report, we will briefly talk about the contents of our system, and how we implemeneted the additional 3 day lagging PM2.5 mean feature. We also offer a brief analysis and interpretation of the performance of our models, and offer our insight on which model seem to be better according to us. Finally, we end with some enhancements that can be made to this system in the future to improve prediction accuracy. 

The code for our lab can be found in **8 Jupyter Notebooks**. 4 of these notebooks include code for the simpler model without the 3-day lagging mean, and the other 4 of these notebooks include code the model/system with the 3-day lagging mean. They can be differentiated as follows:

### Model without Lagging Mean
The code can be found in: 
- `1_air_quality_feature_backfill`
- `2_air_quality_feature_pipeline`
- `3_air_quality_training_pipeline`
- `4_air_quality_batch_inference`

### Model with the 3-day lagging PM2.5 mean
The code can be found in: 
- `1_air_quality_feature_backfill-lagging_feature`
- `2_air_quality_feature_pipeline-lagging_feature`
- `3_air_quality_training_pipeline-lagging_feature`
- `4_air_quality_batch_inference-lagging_feature`

All 8 of these files are present under the `notebooks/ch03` directory.

The dashboard for our Air Quality prediction service, with graphs for both models, can be found at this URL: https://rishivijayv.github.io/mlfs-book/air-quality/

## Chosing a City: Toronto
We will first briefly explain the reason for us chosing the Downtown Toronto location for our PM2.5 measurements. The [AQICN Page for Downtown Toronto](https://aqicn.org/city/canada/ontario/toronto-downtown/) seems to have copious amounts of historical PM2.5 data. Additionally, there were no glaringly obvious errors in the sensor for a prolonged period of time (which would have indicated a faulty sensor and thus unreliable prediction). In general, the air-quality levels predicted by our chosen sensor seem to be consistent. Thus, we felt this sensor for Toronto Downtown would be a good choice. Below, you can see the historical data for this sensor

![Toronto Downtown Historical](https://github.com/rishivijayv/mlfs-book/blob/main/data/toronto-downtown-historical-data-snapshot.png?raw=true)

## Overview of System: Without Lagging PM2.5

The base system predicts PM2.5 levels based on 4 weather indicators:
- temperature
- precipitation
- wind speed
- wind direction

However, before a system can make predictions, it needs to be trained. To do this, we used the historical PM2.5 data referred to in the previous section, along with historical weather forecast for Toronto. This data is retrieved and stored in the relevant Feature Groups in Hopsworks in the notebook `1_air_quality_feature_backfill`. This data is then used to train the model, with the 4 categories of weather data being our features, and the PM2.5 measurement for a given day being our labels. We then train a **regression model** fit our training data (which is a subset of the historical data) to the **XGBoost Regressor**. This is done in the nodebook `3_air_quality_training_pipeline`. These are the 2 static notebooks used by our system.

The other 2 notebooks are run daily using github actions. The notebook `2_air_quality_feature_pipeline` fetches the day's PM2.5 measurements for Toronto along with the weather forecast for the next few days, and the notebook `4_air_quality_batch_inference` uses the retrieved weather forecast data to make PM2.5 predictions for the next few days, along with updating a the hindcast graph which shows how accurately our model has predicted the weather forecast for the last few days. The result of this is the updated [Air Quality dashboard for Downtown Toronto](https://rishivijayv.github.io/mlfs-book/air-quality/)

## Overview of System: With Lagging PM2.5 For Last 3 Days
We have a separate model which uses an additional deature -- the lagging mean of the PM2.5 measurements for the last 3 days. We kept this a separate model instead of changing the original model so that we can compare and contrast the performances of the 2 models. 

The purpose of the files responsible for this model (all 4 files end with `*-lagging-feature.ipynb`) is the same as the previous model described in the last section. However, the code has been changed slightly to account for the 3-day-lagging mean. We will briefly go over the changes here.

### Introducing Lagging 3-Day Mean of PM2.5 to Historic Data
We first updated notebook `1_air_quality_feature_backfill-lagging_feature` to include the mean of the last 3 days of PM2.5 measurements. This was done by modifying the `air-quality` feature group to include a column for the 3-day lagging mean, called `pm25_lagging_3day_mean`. This was computed as follows: If `PM25(y)` denotes the PM2.5 measurement for day `y`, then for a given day `x`, the value of `pm25_lagging_3day_mean` for that day was `[PM25(x-1) + PM25(x-2) + PM25(x-3)] / 3`. This value was stored in the `air-quality-lagging-feature` along with the actual `pm2.5` measurements.

### Manually Computing the 3-Day Mean for Fetched Air Quality
In `2_air_quality_feature_pipeline-lagging_feature` where we fetch the PM2.5 measurements for the day to insert in to our feature group, we manually compute the `pm25_lagging_3day_mean` for the new day by taking the average of the PM2.5 data present in the feature group for the last 3 days.

### Computing Predictions for Upcoming Days
This work is done in `4_air_quality_batch_inference-lagging_feature`. Now, to predict PM2.5 for a day, we need weather data in addition to the mean of the last 3 days' PM2.5 measurement. So, we predict PM2.5 measurements for future dates using a *rolling window* strategy, that works as follows. 

Let `x+1` be the first day for which we need to make a prediction. Then, we compute the mean of the last 3 observed **observed** PM2.5 measurements -- `obs(x), obs(x-1), obs(x-2)` (available to us through the air-quality feature group), and use this, along with the weather forecast for the day, as inputs to our model. This gives us the our prediction for day `x+1`: `pred(x+1)`

For day `x+2`, we use the mean of `pred(x+1), obs(x), obs(x-1)` as the value of `pm25_lagging_3day_mean` for that day, and make the prediction. 

For day `x+3`, the value for the feature is the mean of `pred(x+2), pred(x+1), obs(x)`. For `x+4`, this is the mean of `pred(x+2), pred(x+1), pred(x)`...and so on.

This would likely mean that *if* our prediction for the day `x+1` is a little off, this *deviation* could trickle down to the days further ahead and negatively impact the accuracy of those predictions as well. This is one thing to keep in mind when making such rolling-window predictions.


## Model Analysis and Comparisons
We will now offer brief comparisons of the 2 models, based on their performance on the testing set. We will refer to the model **without** the lagging 3-day PM2.5 mean as Model-1, and to the model **with** the lagging 3-day PM2.5 mean as Model-2.  

### MSE and R^2
For **Model-1**, the MSE was 149.14 and the R^2 was 0.2461. 
For **Model-2**, the MSE was 115.14 and the R^2 was 0.4196. 

A regression model's R^2 score is between 0 and 1, and it tells us how well the model can explain the variance in the data. A higher R^2 for a model means that it can explain the data better (ie, its explanatory power is higher). Of the 2 models, Model-2 has a higher R^2. Additionally, R^2 for our models is computed over the **test set** -- which is data that it has not seen before. So, a higher R^2 would also indicate that the model is likely to generalize well to unseen data. So, based on R^2 computed over the test data, we feel that Model-2 seems to be behaving better.

This also seems to be the case in terms of the MSE computed for our 2 models on the testing set. A Regression Model's MSE indicates how much the predicted values differ from the actual values. So, a lower MSE would generally be preferred and would indicate a more accurate model. Model-2 has a lower MSE than Model-1, so here as well, Model-2 seems to be behaving better based on its performance on the testing set. 

So, combining our observations of MSE and R^2 for the 2 models, we feel that **Model-2 seems to be the better model** among the two. It has a R^2 that is higher than Model-1, and an MSE that is lower than Model-1. This indicates to us that it has more explanatory power than Model-1, and is also more accurate, and would thus be our suggested model to stakeholders based on this performance.

This aligns with our intuition as well -- we felt that adding a feature that takes in to account the trend of PM2.5 values while making predictions would likely perform better, and that indeed seems to be the case.

### Feature Importance
Below is the Feature Importance graph for Model-1 
![Model-1-Feature-Importance](https://github.com/rishivijayv/mlfs-book/blob/main/data/feature-importance.png?raw=true)
The higher the F-score, the more important the feature is in guiding the predictions made by the model. This indicates to us that the temperature seems to be the most important in making predictions for Model-1

Below is the Featuer Importance graph for Model-2
![Model-2-Feature-Importance](https://github.com/rishivijayv/mlfs-book/blob/main/data/feature-importance-lagging-feature.png?raw=true)
Here, the general trend looks similar compared to the first model. The order of wind-speed and wind-direction seems to have changed, but the F-scores of these 2 were incredibly similar in Model-1's F-Score graph as well, which indicates to us that they are similar to each other in terms of importance for both models. The biggest difference in Model-2, however, is how important the `pm25_lagging_3day_mean` seems to be as a feature in predicting air-quality. To us, this seems to line up with the different performance of Model-2 compared to Model-1 based on R^2 and MSE scores as well -- this new feature was the only difference between the 2 models, and its high importance in making predictions could explain why Model-2 performs better than Model-1 on the test set.

### Historic Hindcast Graphs
This is the hindcast graph based on the Model-1's prediction on the test data
![Model-1-Hindcast-Historic](https://github.com/rishivijayv/mlfs-book/blob/main/notebooks/ch03/air_quality_model/images/pm25_hindcast.png?raw=true)

And below, is the hindcast graph based on Model-2's prediction on the test data
![Model-2-Hindcast-Historic](https://github.com/rishivijayv/mlfs-book/blob/main/notebooks/ch03/air_quality_model/images/pm25_hindcast_lagging_feature.png?raw=true)

Based on a quick eye-ball test, Model's prediction seem to be a little closer to the actual PM2.5 measurements for the day. However, the prediction graphs both seem to have a similar shape. This indicates to us that introducing the additional feature to Model-2 seems to **decrease** the intensity with which the predictions deviate from the actual data, however the general shape of the predictions seem to be the same. 

*Note: There are some odd straight-lines near the end of both hindcast graphs. We believe this is because teh weather api seems to be missing weather data for a couple days when making retrieving this data for Toronto*

## Future Work

### Experimenting with Different Models
   For this project, we started with XGBoost, but going forward, it would be valuable to try different models like Random Forest, LightGBM, or even deep learning models like RNNs or LSTMs. Exploring these alternatives could help us find new ways to improve prediction accuracy and understand how different model types handle temporal patterns in air quality data.

### Implementing Periodic Model Retraining
   Rather than training the model just once, another goal is to set up periodic retraining based on ongoing performance metrics. This would let the model adapt better to seasonal changes, shifts in air quality patterns, and other time-related factors, allowing it to stay accurate and reliable over time.

### Testing Location Dependence with Multiple Sensors
   We’d also like to test the model’s generalizability by using air quality data from multiple sensors in different locations. This would help us see if model performance is tied to specific locations or if it can predict well across various areas. If we notice the model performs better at certain locations, it might be worth creating location-specific models for more accurate predictions.

### Dataset Cleanup and Duplication Removal
   Finally, we need to clean up x_test by removing any duplicate entries (e.g., multiple entries from November 13). Doing this will give us more accurate MSE and R² values, helping us evaluate the model’s performance without skewed metrics. This cleanup is a small but essential step to ensure reliable performance evaluation.


