import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
file_path = 'C:\\Users\\wursterkj09\\Downloads\\Restaurant Revenue.xlsx'
data = pd.read_excel(file_path)

# Extracting independent and dependent variables
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend',
          'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform multiple linear regression using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Print out the coefficients of the model
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Use statsmodels for detailed regression summary
X_train_with_const = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_with_const).fit()
print(model_sm.summary())
