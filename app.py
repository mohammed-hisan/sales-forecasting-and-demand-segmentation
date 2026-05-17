
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("Sales Forecasting and Demand Segmentation")
st.write("Machine Learning Deployment Project")

# -------------------------------------------------
# DATASET SELECTION
# -------------------------------------------------

option = st.selectbox(
    "Choose Dataset",
    (
        "SuperMarket Analysis Dataset",
        "Holiday Events Dataset"
    )
)

# =================================================
# DATASET 1 : SUPERMARKET ANALYSIS
# =================================================

if option == "SuperMarket Analysis Dataset":

    st.header("SuperMarket Analysis")

    df = pd.read_csv("data/SuperMarket Analysis.csv")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Columns")
    st.write(df.columns)

    # ---------------------------------------------
    # CLEANING
    # ---------------------------------------------

    df = df.dropna()
    df.columns = df.columns.str.strip()

    # ---------------------------------------------
    # ALGORITHM SELECTION
    # ---------------------------------------------

    algo = st.selectbox(
        "Choose Algorithm",
        (
            "Linear Regression",
            "Decision Tree",
            "K-Means Clustering"
        )
    )

    # =============================================
    # LINEAR REGRESSION
    # =============================================

    if algo == "Linear Regression":

        st.subheader("Sales Forecasting using Linear Regression")

        X = df[['Unit price', 'Quantity', 'Tax 5%', 'cogs']]
        y = df['gross income']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)

        st.write(f"Accuracy (R2 Score): {accuracy:.2f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")

        # Graph
        fig, ax = plt.subplots(figsize=(8,5))

        ax.plot(y_test.values[:50], label="Actual")
        ax.plot(prediction[:50], label="Predicted")

        ax.set_title("Actual vs Predicted Sales")
        ax.legend()

        st.pyplot(fig)

    # =============================================
    # DECISION TREE
    # =============================================

    elif algo == "Decision Tree":

        st.subheader("Sales Forecasting using Decision Tree")

        X = df[['Unit price', 'Quantity', 'Tax 5%', 'cogs']]
        y = df['gross income']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)

        st.write(f"Accuracy (R2 Score): {accuracy:.2f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")

        # Graph
        fig, ax = plt.subplots(figsize=(8,5))

        ax.scatter(y_test, prediction)

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Decision Tree Prediction")

        st.pyplot(fig)

    # =============================================
    # KMEANS CLUSTERING
    # =============================================

    elif algo == "K-Means Clustering":

        st.subheader("Demand Segmentation using K-Means")

        cluster_data = df[['gross income', 'Quantity', 'Rating']]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        kmeans = KMeans(n_clusters=3, random_state=42)

        df['Cluster'] = kmeans.fit_predict(scaled_data)

        st.write(df[['gross income', 'Quantity', 'Rating', 'Cluster']].head())

        # Graph
        fig2, ax2 = plt.subplots(figsize=(8,5))

        ax2.scatter(
            df['gross income'],
            df['Quantity'],
            c=df['Cluster']
        )

        ax2.set_xlabel("Gross Income")
        ax2.set_ylabel("Quantity")
        ax2.set_title("Demand Segmentation")

        st.pyplot(fig2)

        st.success("K-Means Clustering Completed")

# =================================================
# DATASET 2 : HOLIDAY EVENTS
# =================================================

elif option == "Holiday Events Dataset":

    st.header("Holiday Events Dataset")

    holiday_df = pd.read_csv("data/holidays_events.csv")

    st.subheader("Dataset Preview")
    st.write(holiday_df.head())

    # ---------------------------------------------
    # CLEANING
    # ---------------------------------------------

    holiday_df = holiday_df.dropna()

    # Convert categorical values into numbers

    from sklearn.preprocessing import LabelEncoder

    le_locale = LabelEncoder()
    le_type = LabelEncoder()
    le_transferred = LabelEncoder()

    holiday_df['locale_encoded'] = le_locale.fit_transform(
        holiday_df['locale']
    )

    holiday_df['transferred_encoded'] = le_transferred.fit_transform(
        holiday_df['transferred']
    )

    holiday_df['type_encoded'] = le_type.fit_transform(
        holiday_df['type']
    )

    # ---------------------------------------------
    # FEATURES AND TARGET
    # ---------------------------------------------

    X = holiday_df[['locale_encoded', 'transferred_encoded']]
    y = holiday_df['type_encoded']

    # ---------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------
    # DECISION TREE
    # ---------------------------------------------

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    st.subheader("Decision Tree Classification")

    st.write(f"Accuracy: {accuracy:.2f}")

    # ---------------------------------------------
    # GRAPH
    # ---------------------------------------------

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(y_test.values[:50], label="Actual")
    ax.plot(prediction[:50], label="Predicted")

    ax.set_title("Holiday Event Prediction")

    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------------
    # HOLIDAY TYPE DISTRIBUTION
    # ---------------------------------------------

    st.subheader("Holiday Type Distribution")

    holiday_count = holiday_df['type'].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8,5))

    holiday_count.plot(kind='bar', ax=ax2)

    ax2.set_title("Holiday Types")

    st.pyplot(fig2)

    st.success("Holiday Dataset Decision Tree Completed")