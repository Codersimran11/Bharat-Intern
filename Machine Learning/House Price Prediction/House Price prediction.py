import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

file_path = r"Other's Code\Simran's Code\Bharat Intern\Machine Learning\House Price Prediction\House Details.csv"
DataFrame = pd.read_csv(file_path)

X = DataFrame.drop(columns=["price"])
Y = DataFrame["price"]

Label={}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    Label[col] = le
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

Model = LinearRegression()
Model.fit(X, Y)

y_pred = Model.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
      
Area = int(input("Enter the area of House: "))
Bedroom = int(input("Enter the number of Bedrooms: "))
Bathroom = int(input("Enter the Number Of Bathrooms: "))
Stories = int(input("Enter the Number Of Floor: "))
MainRoad = input("Is House connected to Mainroad (yes/no): ")
Guestroom = input("Do House have a Guestroom (yes/no): ")
Basement = input("Do House have a Basement (yes/no): ")
HotWaterHeating = input("Do House have a Hot Water Heating System (yes/no): ")
AC = input("Do House have a AC (yes/no): ")
Parking = int(input("Enter the Number Of Parking Slots: "))
FurnishingStatus = input("House's Furnishing Status (furnished/semi-furnished/unfurnished): ")

user_input = pd.DataFrame(
    {
        "area": [Area],
        "bedrooms": [Bedroom],
        "bathrooms": [Bathroom],
        "stories": [Stories],
        "mainroad": [MainRoad],
        "guestroom": [Guestroom],
        "basement": [Basement],
        "hotwaterheating": [HotWaterHeating],
        "airconditioning": [AC],
        "parking": [Parking],
        "furnishingstatus": [FurnishingStatus],
    }
)
for col, le in Label.items():
    user_input[col] = le.transform(user_input[col])

predicted_price = Model.predict(user_input)
print("The Price must be near about Rs.", predicted_price[0])
