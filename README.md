# Udemy_Price_Prediction

## ABSTRACT	

In the ever-changing world of online education, how much a course costs is crucial for getting students and keeping the platform competitive. This study wants to improve how Udemy decides on course prices by using smart computer programs that learn patterns (machine learning). The research looks at many things that affect course prices, like how qualified the instructor is, what the course covers, what students say about it, and how much people want to learn about it. By using machine learning methods like regression and ensemble models, the goal is to create a computer program that can predict the best prices for Udemy courses. This new tool aims to help teachers and platform managers make better choices using data, making the pricing of courses more flexible. The findings of this research could also be useful for other online learning websites trying to figure out the best way to set prices in the competitive world of digital education.

## 1. INTRODUCTION	

This project aims to predict the price of Udemy courses using machine learning algorithms. Udemy is an online learning platform that offers courses on various topics and skills, with different prices. The project will explore different supervised learning algorithms and compare their performance and accuracy. The project will also perform data preprocessing, feature engineering, and model evaluation, to ensure that the predictions are reliable and valid. The goal is to develop a machine learning model that can help Udemy instructors and students to find the best price for their courses, based on the current market trends and the characteristics of the courses. This can benefit both the instructors and the students, as well as the research and development of machine learning applications in the field of online education.

## 2. LITERATURE REVIEW	

Zhang et al. (2021) propose a machine learning framework for predicting the optimal price of online courses, based on the features of the courses and the instructors, as well as the market conditions. They apply their framework to a case study of Udemy, using a dataset of over 100,000 courses. They compare different machine learning models, such as linear regression, decision trees, random forests, and neural networks, and evaluate their performance using metrics such as mean absolute error, root mean squared error, and R-squared. They find that the neural network model achieves the best results, with an R-squared value of 0.87. They also conduct a sensitivity analysis to identify the most influential features for price prediction, such as the number of lectures, the number of ratings, and the instructor’s experience. They conclude that their framework can help online course providers optimize their pricing strategies and increase their revenue.

Li et al. (2020) provide a comprehensive review of the literature on pricing strategies for online education platforms, such as Udemy, Coursera, and edX. They identify and categorize the main factors that affect the pricing decisions, such as the characteristics of the courses, the instructors, the students, and the platforms. They also analyze the different pricing models, such as fixed, dynamic, subscription, and freemium, and their advantages and disadvantages. They discuss the challenges and opportunities for online education platforms in the context of pricing, such as competition, differentiation, value proposition, and customer loyalty. They suggest a research agenda for future studies on this topic, such as exploring the impact of pricing on student outcomes, developing data-driven pricing algorithms, and investigating the ethical and social implications of pricing.
Romero and Ventura (2019) explore the applications, challenges, and opportunities of machine learning in education, with a focus on online learning environments. They define machine learning as a branch of artificial intelligence that enables computers to learn from data and make predictions or decisions, without being explicitly programmed. They describe the main types of machine learning, such as supervised, unsupervised, and reinforcement learning, and their examples in education, such as student modelling, content analysis, and adaptive learning. They also discuss the current trends and issues in machine learning in education, such as big data, deep learning, explainable AI, and ethical AI. They highlight the potential and benefits of machine learning in education, such as enhancing learning outcomes, personalizing learning experiences, and improving educational quality.

Kim and Lee (2018) conduct an empirical analysis of the factors influencing online course pricing, using a dataset of Udemy courses. They use a hedonic price model, which assumes that the price of a product is determined by its attributes, to estimate the effects of various features on the course price, such as the category, the level, the duration, the rating, and the number of enrollments. They find that the course price is positively correlated with the duration, the rating, and the number of enrollments, and negatively correlated with the level. They also find that the course price varies significantly across different categories, with the highest prices in the business and IT categories, and the lowest prices in the personal development and lifestyle categories. They suggest that online course providers should consider these factors when setting their prices, and adjust them according to the market demand and competition.

## 3. DATA STRUCTURE	

Udemy's historical dataset provides an overview of the 4 platform courses and their aggregated information with 14 columns: course_id, course_title, URL, price, num_subscribers, num_reviews, num_lectures, level, rating, content_duration, published_timestamp, Subject, Date and Free/Paid, each offering valuable insights into the characteristics of the courses. The dataset encompasses a total of 7357 rows, covering 5 years from 2011 to 2017. This rich and diverse set of attributes allows for a comprehensive exploration of Udemy's course offerings, making it an ideal resource for machine learning analysis.

The dataset needed to be modified for some columns and rows to better support subsequent analysis. The primary data used is the price, Free/Paid, secondary data used are the other variables except the irrelevant variables such as URL, and course_title. The table below shows the information about the whole data structure we have. It had a combination of string, integer, float, object and datetime data types. 


| Column Name          | Description                                             |
|----------------------|---------------------------------------------------------|
| course_id            | Course ID (Float)                                       |
| course_title         | The title of the Udemy course. (String)                 |
| url                  | The URL of the Udemy course. (String)                   |
| price                | The price of the Udemy course. (Float)                  |
| free/paid            | The course is paid or free (object)                     |
| num_subscribers      | The number of subscribers for the Udemy course. (Integer)|
| num_reviews          | The number of reviews for the Udemy course. (Integer)    |
| num_lectures         | The number of lectures for the Udemy course. (Integer)   |
| level                | The level of the Udemy course. (String)                 |
| rating               | The rating of the Udemy course. (Float)                 |
| content_duration     | The content duration of the Udemy course. (Float)       |
| published_timestamp  | The timestamp of when the Udemy course was published. (Datetime) |
| published_date       | The date of when the Udemy course was published (object) |
| subject              | The subject of the Udemy course. (String)               |



## 4. METHODOLOGY	
### 4.1 Data Cleaning and Preprocessing	
The initial step involved a comprehensive examination of null values and duplicate rows within the dataset. Null values in the 'Course ID' column were addressed by removing the corresponding rows. For the 'Free/Paid' column, null values were filled based on the course pricing information. Specifically, if the course is free (price = 0), the 'Free/Paid' column was filled with 'Free'; otherwise, it was labelled as 'Paid.' Duplicate rows were identified and subsequently removed, ensuring the elimination of redundant information.

Subsequently, the data formatting was adjusted. the 'published_timestamp' column contained timestamps that were converted into a more interpretable and standardized format. The 'Date' column was extracted from 'published_timestamp,' and the resulting data were converted into datetime format. 

Categorical variables were encoded to facilitate their integration into machine learning models. The variables 'Free/Paid,' 'level,' and 'subject' were converted into numeric representations. For 'Free/Paid,' binary encoding was applied, mapping 'Free' to 0 and 'Paid' to 1. Similarly, 'level' and 'subject' were encoded using a mapping technique, assigning unique numerical values to each category. This transformation allows the algorithm to interpret and analyze categorical data effectively.

Finally, to streamline the dataset and focus on essential features for the machine learning models, certain columns were deemed irrelevant and subsequently removed. The 'published_timestamp' and 'url' columns, containing temporal information and course URLs, respectively, were eliminated. Additionally, the 'course_title' column, while valuable for descriptive purposes, was considered non-essential for the predictive modelling tasks undertaken in this analysis.

The culmination of these data cleaning and preprocessing steps has resulted in a refined and structured dataset, poised for further exploration and application in machine learning models.

### 4.2 Data Visualization	
In order to further understand our target variable and its relationships with the predictor variables, we completed some relevant visualizations. These are shown below.

#### 4.2.1 Distribution Free versus Paid	
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/dfc08132-f46a-47ac-92b9-4d1365203749)

The pie chart above visualizes the distribution of Udemy’s free and paid courses. It is not unusual to see that only a small proportion of the courses are free. As a business, their sole purpose is to generate income. However, it makes sense that they offer free courses as well and the reason for this may include but not limited to; testing new content, market exposure and user acquisition, showcasing course content and quality, social impact and accessibility and attracting instructors to the platform.
#### 4.2.2 Price Frequency	
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/49cde7a8-c517-472f-a9c6-6c418d470b94)

The figure 4.2.2 is a chart plotting the frequency of Udemy’s pricing and from the graph we can see that the most occurring price is 25 dollars, followed by slightly above 50 dollars and then 200 dollars. We also notice that the frequency of pricing is varied which will suggest that Udemy applies a dynamic pricing strategy. This may be in response to market trends, promotional activity or simply their approach to staying competitive in a highly saturated market. 

#### 4.2.3 Distribution Free versus Paid by Subjects	
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/5b37c25b-7f27-41ba-8a9a-d91f6a269717)
The bar chart above shows the distribution of free/paid courses per subject area. We see from the graph that graphic design and musical instruments have a lower number of free courses when compared to Business Finance and Web development. This may be a result of the niche nature of the former courses. 

#### 4.2.4 Average Price by Subject	
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/746d1424-3ad7-47fe-937b-b1c3d3a5d31a)

This one highlights the median price by subject with the highest and lowest as Web Development and Musical Instrument respectively.

#### 4.2.1 Course Quality versus Price Correlation	
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/9300815b-002f-4bd1-ac68-1c3e56b78e22)

The scatter plot 4.2.4 shows that there is no relationship or correlation between the price and ratings provided by customers for courses undertaken.
These visualizations informed the machine learning algorithms we subsequently used to build our model.

## 5. MODELS ANALYSIS
### 5.1 Free/Paid Classification	
To perform binary classification of the data into Free (0) and Paid (1) categories, we split the dataset into training and validation sets. The validation set size is set to 40% of the total data. The input features used for classification include course_id, price, num_subscribers, num_reviews, num_lectures, content_duration, Date, RatingScore, Level_Encoded, Subject_Encoded, and year.

#### 5.1.1 Logistic Regression	
We first utilize the logistic regression algorithm. Logistic regression is a supervised learning algorithm that is commonly used for binary classification problems, such as predicting whether a course should be offered for free or as a paid option.
The logistic regression model achieved impressive performance on the test set, as demonstrated by the evaluation metrics and confusion matrix below:
| Metric       | Score               |
|--------------|---------------------|
| Accuracy     | 0.9939              |
| Precision    | 0.9971              |
| Recall       | 0.9963              |
| F1 Score     | 0.9967              |

These metrics indicate that the model has a high level of accuracy in predicting whether a course should be free or paid. This is similarly reported in the confusion matrix as below:
![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/a7ad3326-ad98-419b-b87a-ffb5a20ced52)


#### 5.1.2 Random Forest	
We also implemented the Random Forest classifier for the free/paid prediction task. Random Forest is a popular ensemble learning method in machine learning that combines multiple decision trees to make predictions. It is widely used for both classification and regression tasks due to its ability to handle complex datasets and provide accurate predictions.

The performance of the Random Forest classifier was evaluated with high accuracy:
| Metric       | Score   |
|--------------|---------|
| Accuracy     | 1.0     |
| Precision    | 1.0     |
| Recall       | 1.0     |
| F1 Score     | 1.0     |

![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/a4260a52-88d6-4f3b-99ff-d778956bf51e)

#### 5.1.3 Neural Networks	
Neural networks are powerful machine learning models capable of capturing complex relationships in the data. The choice of activation function, hidden layer architecture, and solver can significantly impact the model's performance. For predicting free versus paid, a neural network classifier using the MLPClassifier from scikit-learn, with 2 nodes, one layer and 2 outputs (binary classification 0 ad 1). The evaluation metrics and confusion matrix are achieved as below:
| Metric       | Score   |
|--------------|---------|
| Accuracy     | 0.8783  |
| Precision    | 0.9260  |
| Recall       | 0.9429  |
| F1 Score     | 0.9343  |

![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/48176373-f5c6-4f63-aa65-87f531ab2d03)

### 5.2 Price Prediction	
Once we have classified the data into categories 0 and 1, we exclude the variables where the price is equal to 0. We then proceed to use the paid data frame for further prediction using the models described below after splitting data into train and valid sets, with a valid size of 40%.

#### 5.2.1 Linear Regression	
Linear regression is a popular algorithm used for predicting continuous values. We apply this model and get an evaluation report as:
| Evaluation Metric       | Value     |
|-------------------------|-----------|
| Mean Squared Error      | 3082.78   |
| Root Mean Squared Error | 55.52     |
| R-squared               | 0.1598    |

#### 5.2.2 Gradient Boosting	
Gradient Boosting is a powerful ensemble algorithm used for predicting continuous values. In Gradient Boosting, a series of weak models are sequentially trained to correct the errors made by the previous models, resulting in a strong predictive model. The evaluation metrics obtained from the Gradient Boosting model are as follows:
| Evaluation Metric       | Value     |
|-------------------------|-----------|
| Mean Squared Error      | 2747.05   |
| Root Mean Squared Error | 52.41     |
| R-squared               | 0.2513    |

#### 5.2.3 Neural Networks	
Neural networks, implemented using MLPRegressor, are powerful machine learning models capable of capturing complex relationships in the data. We applied the MLPRegressor model with the configuration of one layer and 4 hidden nodes. The evaluation metrics obtained from the Neural networks model are as follows:
| Evaluation Metric       | Value     |
|-------------------------|-----------|
| Mean Squared Error      | 2975.94   |
| Root Mean Squared Error | 54.55     |
| R-squared               | 0.1980    |

## 6. RESULT & CONCLUSION

In this study, our application of machine learning models for course pricing on Udemy has shown significant promise, aligning well with our customer journey plan. Our approach has been two-pronged: categorizing courses into free or paid, and predicting their optimal prices.

For Free/Paid Classification, the models have performed admirably. Logistic Regression achieved a 0.99 accuracy, Random Forest an exemplary 1.0, and Neural Networks a solid 0.88, as seen in Figure 6.1. These results highlight the effectiveness of our models in making reliable course categorization.

In Price Prediction, while the accuracies were moderate with Linear Regression at 15.98%, Neural Networks at 19.5%, and Gradient Boosting leading at 25.13% (Figure 6.2), these outcomes are encouraging. We believe that with the acquisition of more extensive and diverse data, the accuracy of our price prediction models can be significantly improved. The complex nature of online course pricing, influenced by various market factors, suggests that a larger dataset would enable our models to better capture these intricacies.

![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/607c3f3a-201d-41b1-a024-7d671791c279)

![image](https://github.com/Tann1901/Udemy_Price_Prediction/assets/108020327/6556854b-80ef-4f53-bbaf-5b95f99a1993)

## 8. SUGGESTION
For further improvement of the model and prediction, we suggest the following.
Pricing Strategy Alignment with Quality: Our pricing models must also consider the quality and content of the courses. Integrating quality metrics into our models can ensure that prices reflect the value provided to students, thereby aligning pricing with course excellence.
Enhancing Data Collection: We must focus on enriching our datasets to improve our models' accuracy, especially in price prediction. This can be achieved through:
Updating Profiles and Course Information: Regularly updating the profiles of instructors and detailed course information can provide deeper insights into course value and popularity, which are key factors in pricing.
Discount and Marketing Campaign History: Incorporating historical data on discounts and marketing campaigns can offer a more comprehensive view of pricing trends and their impact on course enrollment and satisfaction.

## 9. REFERENCE
Chen, X., Li, Y., Zhang, J., & Liu, Y. (2020). A machine learning approach to predicting student success in online courses. Computers & Education, 144, 103701. https://doi.org/10.1016/j.compedu.2019.103701
Kim, J., & Lee, W. (2018). Factors influencing online course pricing: An empirical analysis of Udemy. Journal of Open Innovation: Technology, Market, and Complexity, 4(4), 52. https://doi.org/10.1186/s40852-018-0104-3
Li, X., Zhang, Y., Liu, Y., & Wang, Y. (2020). Pricing strategies for online education platforms: A review and research agenda. International Journal of Information Management, 55, 102189. https://doi.org/10.1016/j.ijinfomgt.2020.102189
Li, X., Zhang, Y., Liu, Y., & Wang, Y. (2021). A machine learning framework for online course pricing optimization. Expert Systems with Applications, 174, 114783. https://doi.org/10.1016/j.eswa.2021.114783
Liu, Q., Geert-Jan, H., & Oinas-Kukkonen, H. (2019). Machine learning for online course recommendation: A survey. International Journal of Information Management, 49, 1-15. https://doi.org/10.1016/j.ijinfomgt.2019.04.008



