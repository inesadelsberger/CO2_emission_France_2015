# CO2_emission_France_2015
A data analysis project on CO2 emissions by cars in France 2015
Datascientest
Bootcamp January 2024
Data Analyst

PROJECT NAME: CO2
Emission of cars in France 2015

Students' names: Ines Adelsberger and Tina Schreck

1. Report
1.1 Introduction to the project
CO2 emission is an important issue these days. Businesses and governments try to
reduce CO2 emissions by implementing new technologies and finding alternative energy
sources. Therefore the topic of this project, the CO2 emission of cars, is highly relevant.
Cars and vans accounted for 48% of global transport carbon dioxide emissions in 2022,
according to an analysis by Statista based on International Energy Agency data (IEA).

France started collecting data on cars and their CO2
emissions in 2001, at least our first dataset dates back to that year. From year to year the
mass of collected data increases, with later datasets being much bigger than the early
ones, running till 2015. 

1.2 Objectives
The data deal with CO2 emissions of different brands of cars. The objective of this project
will be the analysis of the influence of the different variables (brand, horsepower, inner or
outer city traffic etc) on CO2-emissions of cars in France. Furthermoremachine learning
and modeling was done. 

1.3 Framework
The data came from an official French government website
(https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehiculescommercialises-
en-france/#_).

1.4 Relevance
In order to save our planet from turning into a CO2 polluted gas bubble where the air is
toxic, it is necessary to reduce emissions, but also to have a look at where emissions
come from and to predict how they may develop in future years. As cars (including heavier
vehicles like SUVs and vans, but not buses, medium or heavy freight vehicles) alone make
up almost half of the transportation emissions, analyzing their data is highly relevant.
There were a lot of variables, some not useful, like ID numbers for cars by the
manufacturer or horsepower for tax reasons in addition to the actual horsepower. Irrelevant
variables had to be dropped. The brands were divided into different makes and models,
which resulted in more than 350 different names - impossible to work with or show in a
graph. They were renamed according to brand and when necessary further grouped. For
the CO2 emission a flashy name is not important; a particular subtype is only relevant due
to its weight, horsepower and consumption, which is why these variables were kept.
Our target variable is CO2 emission.

1.5 Pre-processing and feature engineering
To facilitate preprocessing a function was set up in which the following steps were
aggregated: head and info to give a short overview, duplicated to find duplicates, isna to
find missing values as well as calculating their percentage, unique to understand the
values of the file; and finally a visual representation with histplot to show the distribution of
every variable.
After applying this function we used the drop method to delete columns we did not need:
columns full of NaNs, serial numbers, model types (we kept only the brand) or unnamed
columns. The cleaned files were given new names to distinguish them from the originals.

1.6 Visualizations and statistics
After cleaning the dataset comprised seven columns, including brands, horsepower,
urban consumption, rural consumption, mixed consumption, CO2 emission and weight.
There is one categorical variable to work with, which is ‘Brand’. By grouping the values of the CO2
emission within the brands, we receive the means of this value for each brand. Because of the high
number of modalities we just display the 5 highest and the 5 lowest in the following part.
After that we have to reject the variable ‘Brand’ in our dataset. The variable has too many
modalities to work with it. The encoding would be too complex.

(visualizations can be found in the notebook)

Furthermore we checked the numerical variables with qqplotting for normal distribution.The plots
follow a clear normal distribution. This means we need a transformation in the following steps.
We want to see if there are correlations between the numerical values. Therefore a heatmap of the
correlation between them has been computed. It shows that the three different consumption values
have an influence on the CO2 emission.

After finding no normal distribution inside the numerical values it is not recommended to do a
Pearson correlation. Instead of that we choose the Spearman correlation to look for relationships
between the variables. We explicitly look which variables have an influence on the variable CO2
emission.

Like it is shown in the scatterplots, it looks like all variables influence the CO2 emission. The
Spearman Correlation shows that there is a significant influence of all other numerical variables on
CO2 emission. The Spearman Rho says the highest level of correlation is shown between mixed
consumption and CO2 emission.

2. Modeling
2.1 Preparation
The dataset was separated into a training and a test set with sklearn.model_selection and
train_test_split and the variable Brand (the only categorical variable) dropped. Therefore
no encoding was necessary, as the other variables are of type float or integer. The
modalities of the variable Brand were too many and there was no possibility to divide them
in a meaningful way which would render a valid result. Hence we will look at the CO2
emission correlating to fuel consumption, independent of the car brands.
The test set was transformed with StandardScaler before the actual modeling.

2.2 Choice of models – Regression
There are various models to train a dataset. We started with regression models because
we wanted to see how our data did in predicting numeric values and to understand
relationships. Some of the most common models are:
• Linear Regression: The most basic form of regression modeling, assuming a linear
relationship between the dependent and independent variables.
• Logistic Regression: Despite its name, logistic regression is a linear model used for
binary classification. It can be extended to multiclass classification through
techniques like one-vs-rest or multinomial logistic regression.
• Decision Tree Regressor: Decision trees split the dataset into subsets based on the
most significant attribute. They are easy to interpret and can handle both numerical
and categorical data, which is why we will use its classification variant later on.
• Random Forests Regressor: Random forests are an ensemble learning method that
consists of multiple decision trees. They improve upon decision trees by reducing
overfitting and increasing accuracy. (It also has a classification version.)
• K-Nearest Neighbors (KNN) Regressor: KNN classifies instances based on the
1
majority class among their K nearest neighbors in feature space. (It can also be
used for classification.)
We used the above on our data and evaluated the outcome with the metrics below. To
make the calculation process more efficient, a loop was created to run all of the models
and metrics on the dataset.
2.3 Metrics - Regression
With help of the R2 (coefficient of determination) metric the fit of the model to the data was
calculated. The closer the values are to 1, the better the model fits the data. R2 is useful
for comparing different models and assessing their explanatory power: It provides an
indication of how well the independent variables explain the variability of the dependent
variable.
Our results were rather good, with
r_squared of linear regression on test set: 0.97 (first and second run),
r_squared of logistic regression on test set: 0.92 (first and second run), ,
r_squared of Decision Tree Regressor on test set: 0.99 (first run), 0.99 (second run),
r_squared of Random Forest Regressor on test set: 0.99 (first run), 0.99 (second run), and
r_squared of KNNeighbors Regressor on test set: 0.99 (first run), 0.99 (second run).
While values above 0.99 might indicate overfitting, it is nevertheless the goal to get as
close to 1 as possible, and we had not even changed the hyperparameters in this first run.
Therefore we left it at that for the linear regression, KNN, Decision Tree and Random
Forest.
For the logistic regression we got better results after tweaking the hyperparameters:
r_squared of logistic regression on test set: 0.96 (third run).
The MSE (Mean Squared Error) metric on the other hand measures the average squared
difference between the actual values and the predicted values from the model. Since it
squares the errors, larger errors are penalized more heavily than smaller ones and lower is
better: In terms of interpretation, a lower MSE indicates a better fit of the model to the
data. This means that, on average, the predictions made by the model are closer to the
actual values. Keeping that in mind, we evaluated our predictions:
Mean Squared Error (MSE) on linear regression: 49.4 (first and second run),
Mean Squared Error (MSE) on logistic regression: 152.15 (first and second run),
Mean Squared Error (MSE) on Decision Tree: 7.64 (first run), 8.15 (second run),
Mean Squared Error (MSE) on Random Forest: 6.08 (first run), 6.23 (second run), ,
Mean Squared Error (MSE) on KNN: 9.14 (first and second run),
Mean Squared Error (MSE) on logistic regression: 78.5 (third run).
The Decision Tree and Random Forest models along with KNN did a lot better than the
others according to the MSE. Especially the MSE of logistic regression was very high,
which dropped 50% after adjusting the hyperparameters, but still the numbers suggest that
linear and logistic regression are not suited for our dataset if we are looking for accurate
predictions.
The low MSEs on Decision Tree, Random Forest and KNN suggest a good fit of these
models to the data. As the units of MSE are squared units of the dependent variable, in
our case the units of MSE is CO2 in g/km squared.

2.4 Grouping into energy efficiency classes and correlations
Next came the grouping of the CO2 emissions according to energy efficiency ratings to
create a categorical target variable. The CO2 emissions were transformed into classes
from A+++ to G in accordance with the EU standards. To calculate the efficiency there is a
formula for the reference value per car. With a function to assign energy classes in respect
to CO2 emission the classes were created. A new column named 'energy' was set up to
contain these values.
Wikipedia references the official calculations for energy ratings as follows: CO2 reference
value R: R [g/km] = 36,59079 + 0,08987 × M, and total vehicle CO2 value: CO2Diff. [%] =
(CO2Pkw – R) × 100 / R. (https://de.wikipedia.org/wiki/Pkw-
Energieverbrauchskennzeichnungsverordnung).
That percentage is ordered like this:
Energy class percentage
A+++ ≤ −55 %
A++ −54,99 % to −46 %
A+ −45,99 % to −37 %
A −36,99 % to −28 %
B −27,99 % to −19 %
C −18,99 % to −10 %
D −9,99 % to −1 %
E −0,99 % to +8 %
F +8,01 % to +17 %
G > +17,01 %

Next we were interested in correlations among our variables to see how they depend on
one another. A Seaborn boxplot shows the energy classes versus mixed consumption:
This illustrates nicely a smooth progression from lowest to highest consumption,
corresponding to energy classes. There are no outliers; the extreme values on the tails of
the distribution are still within range. Calculating the Spearman's Rho and p-Value brought
the following result: Spearman's Rho: 0.56, p-Value: 0.0 .
Hence there is a significant correlation between Mixed consumption in l/100km and energy
class. The same held true for consumption in urban areas as well as in non-urban areas;
there was a clear correlation of fuel consumption and energy class: the more l/100km were
burned, the more CO2 was emitted.

There is also a significant correlation between Horsepower in kW and energy class:
Spearman's Rho: 0.37, p-Value: 0.0.

And finally, the variables weight and energy showed a strong correlation:
Spearman's Rho: -0.1, p-Value: 0.0.

2.5 Choice of models – Classification
As we created a categorical variable out of the CO2 emission data by grouping them into
energy classes, classification models come in handy, of which there are also several to
choose from. We achieved very good results in regression with Decision Trees and
Random Forest. Hence we decided to use their classification capabilities on our enhanced
dataset, as well as SVM.
Of course there are many more classification models, like AdaBoost, Naive Bayes, Neural
Networks, XGBoost, LightGBM and CatBoost, but for our purposes the aforementioned
suffice.
Our new dataset was separated into training and testset, deleting the variable Brand, and
transformed with the StandardScaler.
2.6 Metrics – Classification
To gain insight on the viability of the Decision Tree, Random Forest and SVM models, a
confusion matrix and classification report were employed.
The confusion matrices depicted high diagonal values for each model, indicating that the
model correctly predicts the majority of instances for each class. The low off-diagonal
values showed that the models made few errors in terms of misclassifying instances.
Here are the results of the accuracy score:
Accuracy of Decision Tree Classifier on test set: 0.97,
Accuracy of Random Forest Classifier on test set: 0.97,
Accuracy of SVM on test set: 0.87.
A classification report provides a comprehensive evaluation of a classification model's
performance, including its ability to correctly classify instances of each class and its overall
accuracy and effectiveness and is a simple way to get the following with just one line of
code: Precision (measures the accuracy of positive predictions, a high precision indicates
that the model makes fewer false positive predictions), Recall (measures the ability of the
model to identify all relevant instances - high recall indicates that the model captures most
of the positive instances), F1 Score (a measure of the harmonic mean of precision and
recall), and Support (the number of actual occurrences of each class in the dataset).
The classification report gave values between 0.91 and 1.00 for precision, recall and f1-
score where Decision Tree and Random Forest were concerned. The SVM model
performed slightly weaker, with most of the values ranging from 0.76 to 1.00 and one
single recall surprisingly low at 0.47. The support numbers proposed a high fidelity for all
three models.
The models were run again on a reduced dataset, without the variables urban
consumption and ex-urban consumption, keeping only mixed consumption, as that suffices
as indicator, due to the strong correlations between these three variables.
The same metrics were used for evaluation. As before, the confusion matrices showed
high diagonal values for each model and excellent classification reports for Decision Tree
and Random Forest, but the SVM performance was found lacking in some instances.
Consequently, the SVM was optimized with better parameters found through GridSearch.
Still the accuracy remained at 0.87.

2.7 Clustering
Before we started with the clusters, we wanted to gain an overview of the distribution of
Energy classes within the variables:

Then we performed PCA (Principal Component Analysis) to to reduce the dimensionality
of data while retaining most of its original information. A lineplot illustrates the explained
variance as indicated by the number of its components:
The slopes are smooth, without outliers that might represent anomalies. As most of the
slopes are steep, the contribution of the principal components is strong. The flatter lines
contribute less to the variance.

For the clustering the energy classes were replaced with numbers. The scatterplots show
the relationship between horsepower and urban consumption.


3. Conclusion
The best performing models for our dataset were the Decision Tree and Random Forest
Classification Models. As noted before, the excellent results might indicate overfitting, and
we would have liked to do more tests as well as compare with another dataset. Also, we
could have created more categorical variables and tried more models. With unsupervised
learning predictions for future years could have been made. Unfortunately, we did not have
enough time for that due to the deadline of this project.
With our models for the 2015 dataset one could deduce how the variables may evolve in
the future and optimize the target CO2 variable, though predicting (for brands and in
general) is difficult because of the ever-changing features of cars and the consequent
variance in CO2 emission. A proper prediction model would be very complicated, trying to
take into account the sinking CO2 emissions of the past years and increase in electric
cars, then projecting that tendency. This is material for a much bigger project.
We would like to do that if time permits, because the topic in itself is very important – after
all, this is about saving our planet.

4. Bibliography
For information on calculations, models and metrics:
Stack Overflow
W3 Schools
Wikipedia https://de.wikipedia.org/wiki/Pkw-Energieverbrauchskennzeichnungsverordnung
FreeCodeCamp
DataScientest Notebooks
Origin of datasets:
https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-
9c9b3267c02b
https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehiculescommercialises-
en-france/#_
16
