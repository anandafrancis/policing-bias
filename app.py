# import necessary libraries
import pandas as pd
import streamlit as st
from visuals import heatmap, barChart, scatter2D, scatter3D, bubbleChart
from visuals import customBarChart, pieChart, customPieChart, dropdrownPieChart
from chatbot import chatResponse, detectBias
from models import decisionTree, svm_model
from featureEng import featureSelector



@st.cache_data
def load_data(file):
    '''Load csv files into pandas'''
    df = pd.read_csv(file)
    return df

# save all data files for analysis
shooting = load_data('../policing-bias/data/shootingComp.csv')
crimes = load_data('../policing-bias/data/crimesComp.csv')
field = load_data('../policing-bias/data/fieldComp.csv')
summary = load_data('../policing-bias/data/summary.csv')

shootingOH = load_data('../policing-bias/data/shootingOH.csv')
crimesOH = load_data('../policing-bias/data/crimesOH.csv')
fieldOH = load_data('../policing-bias/data/fieldOH.csv')

# Create a list of X variables and y variables
shotsX = shooting.columns.tolist()
shotsX.remove('district')
shotsY = 'district'

crimesX = crimes.columns.tolist()
crimesX.remove('district')
crimesY = 'district'

fieldX = list(field.columns)
fieldX.remove('district')
fieldX.remove('city')
fieldX.remove('zip')
fieldX.remove('contact_officer_name')
fieldX.remove('supervisor_name')
fieldX.remove('contact_reason')
friskedX = fieldX
friskedX.remove('was_frisked')
fieldY1 = 'district'
fieldY2 = 'city'
fieldY3 = 'zip'
fieldY4 = 'was_frisked'
fieldY5 = 'contact_officer_name'
fieldY6 = 'supervisor_name'


def main():
    '''Create Web Application'''

    # define web pages
    page = st.sidebar.selectbox(
                          "Select a Page",
                          [
                            "Home",
                            "Correlation Heat Maps",
                            "Frequency Plots",
                            "Scatterplots",
                            'Pie Charts',
                            'Bias Buster: Best AI Detective Around',
                            'Feature Engineering',
                            'Decision Trees',
                            'Support Vector Machines',
                          ],)
    
    # create homepage
    if page == "Home":

        # include intro animation
        st.balloons()

        # create a title
        st.title("Equity In Action")
        st.write('\n\n')

        # add description of the project
        st.write("Welcome to my comprehensive policing data dashboard, designed to help you analyze and "+ 
                 "identify potential bias in policing using state-of-the-art visualizations, tree-based "+ 
                 "models, and natural language processing. Our dashboard is built around three carefully "+
                "curated datasets, all from 2022, each with its own unique set of analysis tools. On the left-hand side of "+ 
                "the page, you'll find our intuitive navigation menu, allowing you to easily switch between "+ 
                "the various web pages and explore the different forms of analysis available.")
        
        
        st.write('\n\n')


        # introduce datasets used for analysis
        st.subheader("Shooting Dataset")
        
        st.write(shooting.head())

        st.write('\n\n\n')
        
        st.write('The Shootings dashboard displays data on shooting incidents in Boston within the jurisdiction of the Boston Police Department. It includes incident details and victim demographics, but excludes self-inflicted and justifiable shootings. The information is updated by the Boston Regional Intelligence Center with a 7-day rolling delay. Below are the column names and their descriptions')

        st.write('\n\n\n')

        st.write(
            {'shooting_type_v2': 'describes if the shooting resulted in a casuality',
             'victim_gender': 'gender of the victim',
             'victim_race': 'race of the victim',
             'victim_ethnicity_nibrs': 'whether the victim was hispanic or latinx',
             'district': 'the policing district the shooting occured in',
             'victim_plural': 'describes if there were multiple victims',
             'month': 'the month the shooting took place',
             'day': 'the day the shooting took place',
             'hour': 'the hour the shooting took place'}
        )


        st.write('\n\n\n\n\n\n\n')
        st.write('\n\n\n\n\n\n\n')

        st.subheader("Crime Dataset")  
        
        st.write(crimes.head())

        st.write('\n\n\n')

        st.write("The Boston Police Department (BPD) provides crime incident reports that document initial details surrounding incidents responded to by BPD officers. The dataset includes records from the new crime incident report system that focus on capturing the incident's type, location, and time, with records starting from June 2015 onwards. It's a reduced set of fields compared to previous reports.")

        st.write('\n\n\n')
        
        st.write({
        "offense_description": 'type of crime',
        "district": 'the district the crime occured in',
        "shooting": 'if there was a shooting (0 for False, 1 for True)',
        "month": 'the month the crime took place in',
        "weekend": 'if the crime occurred during the weekend (Fri-Sun)',
        "hour": 'the hour the crime occured in',
        "day": 'the day of the month the crime occured in'
        })

        st.write('\n\n\n\n\n\n\n')
        st.write('\n\n\n\n\n\n\n')

        st.subheader("Police Field Interactions Dataset")
        
        st.write(field.head())

        st.write('\n\n\n')

        st.write("The FIO program, which covers various interactions between the Boston Police Department (BPD) and private individuals, is now recorded in two tables: FieldContact and FieldContact_Name. These tables are compiled from the BPD's new Records Management System (RMS), with the data provided here being a static representation of the Field Interaction and/or Observation that occurred in 2019. However, it's worth noting that the FIOs are maintained in a live database, and the information related to each individual may change over time, with NULL indicating no entry for an optional field.")

        st.write('\n\n\n')
        
        st.write(
        {
        "sex": 'the sex of the civilian that interacted with the police',
        "race": 'the race of the civilian that interacted with the police',
        "build": 'the build of the civilian that interacted with the police',
        "hair_style": 'the hair style the civilian wore during the interaction with the police',
        "ethnicity": 'the ethnicity of the civilian that interacted with the police',
        "skin_tone": 'the skin tone of the civilian that interacted with the police',
        "fc_num": 'the cas number of the interaction between police and civilian',
        "age": 'the age of the civilian that interacted with the police',
        "was_frisked": 'if the civilian was frisker',
        "contacthour": 'the hour the interaction took place',
        "circumstance": 'describes if the interaction was from observation, a stop or an encounter',
        "basis": "describes the reason for the interaction",
        "vehicle_model": 'the vehicle model of the civilian that interacted with the police',
        "vehicle_style": 'the vehicle style of the civilian that interacted with the police',
        "vehicle_year": 'the year of the vehicle of the civilian that interacted with the police',
        "vehicle_type": 'the vehicle type of the civilian that interacted with the police',
        "weather": 'the weather during the interaction',
        "vehicle_state": 'the state the vehicle is registered in of the civilian that interacted with the police',
        "contact_officer_name": 'the name of the officer than interacted with the civilian',
        "supervisor_name": 'the name of the supervisor of the officer',
        "city": 'the city in interaction took place in',
        "zip": 'the zip in interaction took place in',
        "stop_duration": 'the length of the interaction ',
        "month": 'the month in interaction took place in',
        "day": 'the day in interaction took place on',
        "hour": 'the hour in interaction took place in',
        "district": 'the district in interaction took place in',
        "contact_reason": 'notes from police about the interaction explaining event and cause'
        }
        )

    
    # create new page
    elif page == "Correlation Heat Maps":
        
        st.title("Correlation Heat Maps")

        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write('\n\n\n')
        st.write("I created code that generates a heat map to visualize the results of a chi-squared test. The chi-squared test was used to determine whether there is a significant relationship between each X variable and the Y variable: policing district. I focus on policing district to see if there could be any bias in policing based on neighborhood and how it connects to specific features. Further investigation may be needed to understand the nature of these associations and to develop interventions that address these biases.")
        
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader("Correlation Between Features and Policing District from Shooting Dataset")
        heatmap(shooting, shotsX, shotsY)
        st.write('Based on the p-values calculated, the race of the victim and the month, day, and hour of the shooting had the strongest associations with the target variable. The race of the victim being an influential factor in predicting policing distribution could be due to different neighborhoods have different races that are more populated than others. As for the temporal correlation detected, time series analysis could be incorporated in the future to look at trends between what months, days and hours shootings are more common in specific neighborhoods and how can police prevent these shootings without implementing bias.')


        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader("Correlation Between Features and Police District from Crimes Dataset")
        heatmap(crimes, crimesX, crimesY)
        st.write("Based on the p-values and the heat map, it seems that the type of crime, the presence of a shooting, whether it was a weekend, and the month and hour of the crime were most strongly associated with the policing district. The correlation between crime types and police districts could potentially reveal bias in policing, as overpolicing in certain neighborhoods known for certain crimes could result in an overrepresentation of those crimes in the data. The higher occurrence of crimes on weekends may be due to an actual increase in crime during those times or police forces becoming more active in looking for crime during the weekends. Further analysis would be needed to determine whether the observed trends are the result of bias or reflect actual patterns in civilian crime occurrence.")

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader("Correlation Between Features and Neighborhoods from Field Contact Dataset")
        heatmap(field, fieldX, fieldY1)
        heatmap(field, fieldX, fieldY2)
        heatmap(field, fieldX, fieldY3)

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader("Correlation Between Features and Officers from Field Contact Dataset")
        heatmap(field, fieldX, fieldY5)
        heatmap(field, fieldX, fieldY6)
        heatmap(field, friskedX, fieldY4)

    # create new page
    elif page == "Frequency Plots":
        
        st.title("Frequency Plots")

        st.subheader('Methodology')
        st.write("I organized the original datasets used for analysis by district to examine the occurrence frequency of shootings, field police interactions, crimes, and the specific types of crimes, based on policing district. The aim was to detect patterns associated with policing district, which could either corroborate or contradict the notion of bias in policing in Boston based on neighborhoods and other characteristics.")

        st.write('\n\n\n')
        st.write('\n\n\n')
        
        st.subheader("Frequency Plots of Incidents By District")
        barChart(summary, 'district', 'shootings', 
                       'Shootings in 2022 By Policing District', 'District', 'Frequency')
        
        barChart(summary, 'district', 'crimes', 
                       'Field Interactions in 2022 By Policing District', 'District', 'Frequency')
        
        barChart(summary, 'district', 'fieldInt', 
                       'Crimes Committed in 2022 By Policing District', 'District', 'Frequency')
        
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader("Customizable Frequency Plot of Specific Crimes By District")        
        customBarChart('offense_description', 'district', 'hour', crimes)

    # create new page
    elif page == "Scatterplots":

        st.title("Scatterplots")
        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write("I utilized Plotly to create both 3D and 2D scatter plots and bubble charts to visualize the relationships between the number of shootings, field police interactions, and crimes that occurred during 2022. The scatter3d object was created using the go.Scatter3d() function, which takes data from a Pandas dataframe. This visualization aims to identify trends and potential correlations between these variables, which could indicate if policing is done based on crime occurrence and prevention or other factors and how it differs by district.")
        st.write("The bubble charts also visualize the frequency of these incidents and allow for correlation analysis to be conducted. The size of each bubble represents the frequency of the corresponding incident, while the color indicates the policing district being analyzed. This allows for easy identification of patterns and relationships between the datasets. Overall, these visualizations help to provide insights into the potential presence of bias in policing based on the occurrence of crimes and interactions in different neighborhoods.")
        
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('3D Scatterplot of Count Data By District')
        scatter3D(summary, 'shootings', 'crimes', 'fieldInt', 'district')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Bubble Charts')
        bubbleChart(summary, 'shootings', 'fieldInt', 'crimes', 'district')
        bubbleChart(summary, 'shootings', 'crimes', 'fieldInt', 'district')
        bubbleChart(summary, 'crimes', 'fieldInt', 'shootings', 'district')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        #st.subheader('2 Variable Scatterplots')
        #scatter2D(summary, 'shootings', 'crimes', 'district')
        #scatter2D(summary, 'shootings', 'fieldInt', 'district')
        #scatter2D(summary, 'crimes', 'fieldInt', 'district')

        

    # create new page
    elif page == 'Pie Charts':

        st.title("Pie Charts")
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write('I created pie charts to display the distribution of each feature based on a group variable. I have also developed customizable charts where you can group data based on various factors such as the police officer, their supervisor, the district, zip code, city, and whether a civilian was frisked or not, to show the percentage breakdown for each of these groups with respect to the race of the civilian and whether they were frisked.')
        st.write('In addition, I included other pie charts that show the percentage breakdown of field interactions, shootings, and crimes that occurred in Boston by district. This helped identify the districts with the highest and lowest percentages, similar to a frequency plot. Lastly, I created pie charts that show demographic information for shooting victims by race and gender and field interactions by race, build and hairstyle.')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        
        st.subheader('Customizable Pie Charts')
        fieldOptions = ['contact_officer_name', 'supervisor_name', 'district',
                        'zip', 'city', 'was_frisked']
        dropdrownPieChart('hour', 'race', field, fieldOptions, 'abc', 'def')
        dropdrownPieChart('hour', 'was_frisked', field, fieldOptions, 'ghi', 'jkl')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        
        st.subheader('Pie Charts from Summary Data')
        pieChart('crimes', 'district', summary)
        pieChart('shootings', 'district', summary)
        pieChart('fieldInt', 'district', summary)

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        
        st.subheader('Pie Charts from Shootings Data')
        customPieChart('hour', 'victim_race', shooting, 'SHOOTINGS')
        customPieChart('hour', 'victim_gender', shooting, 'SHOOTINGS')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        
        st.subheader('Pie Chart from Field Data')
        customPieChart('hour', 'race', field, 'FIELD INTERACTIONS')
        customPieChart('hour', 'build', field, 'FIELD INTERACTIONS')
        customPieChart('hour', 'hair_style', field, 'FIELD INTERACTIONS')

    # create new page
    elif page == 'Bias Buster: Best AI Detective Around':
        
        st.title("Bias Buster: Best AI Detective Around")

        st.subheader('Methodology')
        st.write("Bias Buster is an AI-powered chat bot designed to detect potential biases in police interactions with civilians. Utilizing OpenAI's API, it analyzes the notes taken by police officers during these interactions to identify any underlying biases. With Bias Buster Knows All, users can engage in discourse with the chat bot by asking questions and receiving responses. Additionally, Find the Bias allows users to input a specific case number, which serves as the primary key for a field police interaction, and view the corresponding notes along with Bias Buster's analysis of whether bias was present or not.")

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Bias Buster Knows All')
        st.write('Ask any questions or may any statement regarding policing, bias, AI, ANYTHING!')
        
        # Add a text input field for the user's message
        user_input = st.text_input("Enter your message")

        if user_input:
            chatResponse(user_input)

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Find the Bias')
        detectBias(field)

    # create new page
    elif page == 'Feature Engineering':

        st.title('Feature Engineering')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write(' I explored different packages, such as SelectPercentile, f_classif, mutual_info_classif, chi2, and SelectKBestModels from sci-kit learn to determine the best features to use. The user can choose which model they would like to use to find the features most closely related to the target variable: policing district. Despite comprehensive feature selection, the model performance was still low. Given more time, I would explore boosting and stacking to improve model outputs. However, for now, I found a way to demonstrate their value through feature importance and visualizations.')
        
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.header('Feature Selection for Shooting Data')
        # remove non x variables
        xColSD = list(shootingOH.columns)
        xColSD.remove('district')

        featureSelector(shootingOH, xColSD, 'district', 'SD')

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.header('Feature Selection for Crimes Data')
        # remove non x variables
        xColCD = list(crimesOH.columns)
        xColCD.remove('district')


        featureSelector(crimesOH, xColCD, 'district', 'CD')


        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.header('Feature Selection for Field Interactions Data')
        # remove non x variables
        xColFD = list(fieldOH.columns)
        xColFD.remove('district')
        xColFD.remove('fc_num')
        xColFD.remove('city')
        xColFD.remove('zip')
        xColFD.remove('contact_officer_name')
        xColFD.remove('supervisor_name')
        xColFD.remove('contact_reason')
    
        featureSelector(fieldOH, xColFD, 'district', 'FD')


    # create new page
    elif page == 'Decision Trees':

        st.title('Decision Tree Models')
        
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write("A decision tree is a predictive model that uses a tree-like structure to make predictions based on input features. Each internal node represents a decision based on a feature, and each leaf node represents a prediction. To interpret the decision tree visualization, follow the path from the root node to a leaf node, which represents a decision path that leads to the predicted outcome or class variable.")
        st.write("I used the sklearn library to build a decision tree model that identifies the most important features for predicting policing districts. Identifying feature importance is critical to detecting and addressing bias in policing, as it can reveal which characteristics are most influential in determining how policing is carried out in different neighborhoods. To visualize the feature importance, I created a bar graph where the height of each bar represents how influential a particular feature is in predicting the policing district.")
        st.write("You can create different decision trees based on any of the three datasets below, each of which uses different variables to predict policing districts. With each decision model there is an output of the labelled tree, feature importance bar graph and model metrics: accuracy, recall, f1 and precision.")
        
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Decision Tree Model for Shootings Dataset')
        shotsXInput = st.multiselect(
            'Select Variables to Use in Decision Tree Model To Predict Policing District:',
              shootingOH.columns)
        button = st.button('Submit Variables', key='shotsButton')
        if button:
            decisionTree(shootingOH, shotsXInput, ['district'])

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader('Decision Tree Model for Field Interactions Dataset')
        fieldXInput = st.multiselect(
            'Select Variables to Use in Decision Tree Model To Predict Policing District:',
              fieldOH.columns)
        button = st.button('Submit Variables', key='fieldButton')
        if button:
            decisionTree(fieldOH, fieldXInput, ['district'])

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader('Decision Tree Model for Crimes Dataset')
        crimesXInput = st.multiselect(
            'Select Variables to Use in Decision Tree Model To Predict Policing District:',
              crimesOH.columns)
        button = st.button('Submit Variables', key='crimesButton')
        if button:
            decisionTree(crimesOH, crimesXInput, ['district'])

    
            
    # create new page
    elif page == 'Support Vector Machines':

        st.title('Support Vector Machine Models')

        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Methodology')
        st.write("A multiclass SVM is a machine learning algorithm that is specifically designed to classify data into more than two classes. It works by identifying the optimal hyperplane that can best separate different classes of data. In this particular use case, I have utilized the algorithm to categorize data into different policing districts based on input features. A linear kernel function has been used to generate coefficients for each feature.")
        st.write("There are three datasets available, each with different variables used to classify policing districts. You can create various SVM algorithms based on these datasets. Each model generates a bar graph that shows the coefficient values for each feature. If the coefficient value is 0, then no bar will be displayed. Additionally, the model's performance metrics, such as accuracy, recall, f1 and precision, will be outputted.")

        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')

        st.subheader('Support Vector Machine Algorithm for Shootings Data')
        shotsXInput = st.multiselect(
            'Select Variables to Use in Support Vector Machine Model To Predict Policing District:',
              shootingOH.columns)
        button = st.button('Submit Variables', key='shotsButton')
        if button:
            svm_model(shootingOH, shotsXInput, ['district'])  

  
        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader('Support Vector Machine Algorithm for Field Interactions Data')
        fieldXInput = st.multiselect(
            'Select Variables to Use in Support Vector Machine Model To Predict Policing District:',
              fieldOH.columns)
        button = st.button('Submit Variables', key='fieldButton')
        if button:
            svm_model(fieldOH, fieldXInput, ['district'])


        st.write('\n\n\n')
        st.write('\n\n\n')
        st.write('\n\n\n')


        st.subheader('Support Vector Machine Algorithm for Crimes Data')
        crimesXInput = st.multiselect(
            'Select Variables to Use in Support Vector Machine Model To Predict Policing District:',
              crimesOH.columns)
        button = st.button('Submit Variables', key='crimesButton')
        if button:
            svm_model(crimesOH, crimesXInput, ['district'])      

        

    
        
if __name__ == "__main__":
    main()