Introduction: 

AI for Health: Predicting Health Outcomes through BRFSS and U.S. Health Rankings through Machine Learning: 

This project explores how predictive analytics can be used to understand and forecast population health in the United States using data from the 2023 Behavioral Risk Factor Surveillance System (BRFSS). The BRFSS is the nation’s largest health-related survey, capturing information on health behaviors, chronic conditions, and access to care. By leveraging both individual-level responses and state-level prevalence data, this project aims to build a model capable of predicting general health outcomes and estimating overall health rankings across all 50 states. The goal is to demonstrate how large-scale public health data can be transformed into meaningful insights that support data driven decision making in the health care industry.

Data sources: 

| Source # | Name/Source | URL              | Type | List of Fields | Format | Estimated Data Size, number of data points you plan to use |
| :--------:| :-----------: | :--------------: | :-----: | :--------------: | :------: |  :----------------------------------------------------------: |
| 1        | BRFSS ‘23   | https://www.kaggle.com/datasets/isuruprabath/brfss-2023-csv-dataset?select=BRFSS2023.csv | File | Healthcare | CSV | 25,000 |
| 2        | HealthData.gov / BRFSS Prevalence Data | https://healthdata.gov/CDC/Behavioral-Risk-Factor-Surveillance-System-BRFSS-P/khk6-anfn/about_data | File | Healthcare | CSV | 25,000 | 
| 3        | America’s Health Rankings / United Healthcare Foundation | https://api.americashealthrankings.org/graphql | API | Healthcare | JSON | 25,000 |

Analysis: 

In this project, I analyzed the 2023 BRFSS dataset to identify how behavioral and health related factors contribute to individuals’ general or overall health outcomes. I examined key predictors including obesity status, smoking behavior, and chronic disease indicators, using both individual level and prevalence based categorical variables to capture broader health patterns. These predictors were then integrated into a predictive modeling framework such as a Random Forest Classifier and I also aggregated the state level data to assess the overall health trends across the United States creating a Random Forest Regressor predictive model. This analytical approach allowed me to link personal health characteristics with population-evel patterns, forming the foundation for generating state level health estimates.


Summary of the results: 

I built a predictive model to estimate individuals’ general health based on behavioral, and health-related variables. By incorporating categorical prevalence measures such as obesity rates, smoking status, chronic disease prevalence, and other variables from the BRFSS prevalence dataset, the model captured broader population-level health patterns that influence self-reported health. I then aggregated these predictors to the state level to generate estimates of overall health for all 50 states, producing results that aligned with known U.S. health rankings. Overall, the project demonstrates how BRFSS data can be used to predict both individual general health outcomes and larger state-level health trends within the United States.

How to run: 
- describe how to run your pipeline and reproduce results of your work/analysis, including fetching the data. We should be able to reproduce data 



Loading/fetching and data processing: 

I was able to gather my API data from America's Health Ranking by United Healthcare Foundation, I created an account from the website and through the website I was able to get a free API key once the email is confirmed. In order to gather the data from the website, you have to follow specific instuctions because the API is an GraphQL and this API ony allows clients to request data that they need. In my case I needed population data in regard to U.S. Health rankings, so I first had to gather information from the GetReports and see what kind of Datar reports they had acquired on the population level. Upon loooking at this section of data I looked at the meausureID and utilized the measureIDs of the following from the GetReports list: (measureID: "16465") = Risk factors (annual); (measureID: "16535") = Behaviors; (measureID: "17679") = Behavioral health & (measureID: "16540" = Overall health. I wanted to gather data that was relevant to behavior's and overall general health across the population within the U.S. After that I created a query that encompassed all the measuredIDs across the population by utilizing the GetMeasurePopulationData for all 50 states and if the predictive model could predict the overall health utilizing data from the API.  
