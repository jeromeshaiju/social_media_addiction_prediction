from sklearn.tree import DecisionTreeRegressor
import pandas as pd

scocialmediadata = pd.read_csv('social media addiction\social_media_productivity_6000.csv')


# 50 is for medium 100 for high and 0 for low


modal = DecisionTreeRegressor(random_state=5)
scocialmediadata = scocialmediadata.dropna(axis=0)
X = scocialmediadata[["age","daily_screen_time","social_media_hours","study_hours","sleep_hours","notifications_per_day","focus_score","productivity_score"]]
y= scocialmediadata.addiction_level
modal.fit(X, y)


print("Enter values for prediction:")
age = float(input("Enter age: "))
daily_screen_time = float(input("Enter daily screen time (hours): "))
social_media_hours = float(input("Enter social media hours: "))
study_hours = float(input("Enter study hours: "))
sleep_hours = float(input("Enter sleep hours: "))
notifications_per_day = float(input("Enter notifications per day: "))
focus_score = float(input("Enter focus score(out of 100): "))
productivity_score = float(input("Enter productivity score(out of 100): "))

new_data = pd.DataFrame({
    "age": [age],
    "daily_screen_time": [daily_screen_time],
    "social_media_hours": [social_media_hours],
    "study_hours": [study_hours],
    "sleep_hours": [sleep_hours],
    "notifications_per_day": [notifications_per_day],
    "focus_score": [focus_score],
    "productivity_score": [productivity_score]
})



predict=modal.predict(new_data)
if predict[0]==100:
    print("HIGH")
elif predict[0]==50:
    print("MEDIUM")
elif predict[0]==0:
    print("LOW")