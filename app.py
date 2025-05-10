#!/usr/bin/env python
# MandiMithra Backend System
# AI-powered agricultural marketplace platform
# Data source: https://www.data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi#api

import os
import json
import time
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import schedule
import threading
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mandimithra.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = os.environ.get("DATA_GOV_API_KEY", "your_api_key_here")  # Replace with actual API key

# Database setup
DB_PATH = "mandimithra.db"

def initialize_database():
    """Create SQLite database tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS commodity_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        state TEXT,
        district TEXT,
        market TEXT,
        commodity TEXT,
        variety TEXT,
        arrival_date TEXT,
        min_price REAL,
        max_price REAL,
        modal_price REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS price_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commodity TEXT,
        market TEXT,
        state TEXT,
        predicted_price REAL,
        confidence_score REAL,
        prediction_date TEXT,
        prediction_for_date TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        user_type TEXT,
        state TEXT,
        district TEXT,
        preferred_market TEXT,
        preferred_commodities TEXT,
        registration_date TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def fetch_data_from_api(offset=0, limit=1000, filters=None):
    """
    Fetch commodity price data from the Data.gov.in API
    
    Args:
        offset (int): Starting record
        limit (int): Number of records to fetch
        filters (dict): Optional filters for the API
        
    Returns:
        dict: API response data
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "offset": offset,
        "limit": limit
    }
    
    # Add filters if provided
    if filters:
        for key, value in filters.items():
            params[key] = value
    
    try:
        logger.info(f"Fetching data from API with params: {params}")
        response = requests.get(API_BASE_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        logger.info(f"Successfully fetched {len(data.get('records', []))} records")
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"records": []}

def save_data_to_db(records):
    """
    Save fetched records to the database
    
    Args:
        records (list): List of records to save
    """
    if not records:
        logger.warning("No records to save")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        for record in records:
            cursor.execute('''
            INSERT INTO commodity_prices 
            (timestamp, state, district, market, commodity, variety, arrival_date, min_price, max_price, modal_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                record.get('state', ''),
                record.get('district', ''),
                record.get('market', ''),
                record.get('commodity', ''),
                record.get('variety', ''),
                record.get('arrival_date', ''),
                float(record.get('min_price', 0)),
                float(record.get('max_price', 0)),
                float(record.get('modal_price', 0))
            ))
        
        conn.commit()
        logger.info(f"Successfully saved {len(records)} records to database")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save records to database: {e}")
    
    finally:
        conn.close()

def update_database():
    """Scheduled task to update the database with latest price data"""
    logger.info("Starting scheduled database update")
    
    # Get data for the last 7 days
    today = datetime.datetime.now()
    week_ago = today - datetime.datetime.timedelta(days=7)
    
    filters = {
        "from-date": week_ago.strftime("%d-%m-%Y"),
        "to-date": today.strftime("%d-%m-%Y")
    }
    
    # Fetch and save data
    data = fetch_data_from_api(filters=filters, limit=10000)
    if data and "records" in data:
        save_data_to_db(data["records"])

def get_commodity_data_from_db(commodity=None, state=None, market=None, days=30):
    """
    Retrieve commodity price data from the database
    
    Args:
        commodity (str): Filter by commodity name
        state (str): Filter by state
        market (str): Filter by market
        days (int): Number of days of data to retrieve
        
    Returns:
        pd.DataFrame: DataFrame containing the requested data
    """
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM commodity_prices WHERE 1=1"
    params = []
    
    if commodity:
        query += " AND commodity = ?"
        params.append(commodity)
    
    if state:
        query += " AND state = ?"
        params.append(state)
        
    if market:
        query += " AND market = ?"
        params.append(market)
    
    # Add date filter
    if days:
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        query += " AND timestamp > ?"
        params.append(cutoff_date)
    
    query += " ORDER BY arrival_date DESC"
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Retrieved {len(df)} records from database")
        return df
    
    except Exception as e:
        logger.error(f"Failed to retrieve data from database: {e}")
        return pd.DataFrame()
    
    finally:
        conn.close()

def prepare_data_for_model(df):
    """
    Prepare data for model training
    
    Args:
        df (pd.DataFrame): Raw data from database
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, encoders, scaler
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for model training")
        return None, None, None, None, None, None
    
    # Convert dates to datetime and extract features
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['day_of_week'] = df['arrival_date'].dt.dayofweek
    df['month'] = df['arrival_date'].dt.month
    df['year'] = df['arrival_date'].dt.year
    df['day'] = df['arrival_date'].dt.day
    
    # Handle missing values
    df = df.dropna()
    
    # Extract features and target
    features = ['day_of_week', 'month', 'year', 'day', 'state', 'market', 'commodity', 'variety']
    X = df[features].copy()
    y = df['modal_price']
    
    # Encode categorical variables
    encoders = {}
    for col in ['state', 'market', 'commodity', 'variety']:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoders, scaler

def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        model: Trained Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=20,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_neural_network_model(X_train, y_train):
    """
    Train a neural network regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        model: Trained neural network model
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics

def make_price_prediction(model, commodity, market, state, variety, encoders, scaler, days_ahead=7):
    """
    Make price predictions for a specific commodity in a market
    
    Args:
        model: Trained model
        commodity (str): Commodity name
        market (str): Market name
        state (str): State name
        variety (str): Variety name
        encoders (dict): Dictionary of LabelEncoders
        scaler: StandardScaler
        days_ahead (int): Number of days to predict ahead
        
    Returns:
        list: List of predictions for each day
    """
    today = datetime.datetime.now()
    predictions = []
    
    for i in range(1, days_ahead + 1):
        future_date = today + datetime.timedelta(days=i)
        
        # Create a sample for prediction
        X_pred = pd.DataFrame({
            'day_of_week': [future_date.weekday()],
            'month': [future_date.month],
            'year': [future_date.year],
            'day': [future_date.day],
            'state': [state],
            'market': [market],
            'commodity': [commodity],
            'variety': [variety]
        })
        
        # Encode categorical variables
        for col in ['state', 'market', 'commodity', 'variety']:
            if col in X_pred.columns and col in encoders:
                try:
                    X_pred[col] = encoders[col].transform(X_pred[col])
                except ValueError:
                    # Handle unseen categories
                    X_pred[col] = 0
        
        # Scale features
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        price = float(model.predict(X_pred_scaled)[0])
        
        # Calculate confidence score (simplified)
        confidence = 0.8 - (i * 0.05)  # Confidence decreases as we predict further ahead
        
        predictions.append({
            'date': future_date.strftime("%Y-%m-%d"),
            'predicted_price': round(price, 2),
            'confidence': round(confidence, 2)
        })
        
        # Save prediction to database
        save_prediction(commodity, market, state, price, confidence, future_date)
    
    return predictions

def save_prediction(commodity, market, state, price, confidence, prediction_date):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO price_predictions 
        (commodity, market, state, predicted_price, confidence_score, prediction_date, prediction_for_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            commodity,
            market,
            state,
            price,
            confidence,
            datetime.datetime.now().isoformat(),
            prediction_date.isoformat()
        ))
        
        conn.commit()
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save prediction to database: {e}")
    
    finally:
        conn.close()

def get_market_insights(commodity, market=None, state=None, days=90):
    """
    Generate market insights for a commodity
    
    Args:
        commodity (str): Commodity name
        market (str): Optional market filter
        state (str): Optional state filter
        days (int): Number of days of historical data to analyze
        
    Returns:
        dict: Dictionary containing market insights
    """
    df = get_commodity_data_from_db(commodity=commodity, market=market, state=state, days=days)
    
    if df.empty:
        return {
            "status": "error",
            "message": "No data available for the specified parameters"
        }
    
    # Calculate price trends
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df = df.sort_values('arrival_date')
    
    # Calculate rolling statistics
    df['rolling_avg_7d'] = df['modal_price'].rolling(window=7, min_periods=1).mean()
    df['rolling_avg_30d'] = df['modal_price'].rolling(window=30, min_periods=1).mean()
    
    # Get current, min, max prices
    current_price = df['modal_price'].iloc[-1] if not df.empty else 0
    min_price = df['modal_price'].min()
    max_price = df['modal_price'].max()
    
    # Calculate price volatility
    price_std = df['modal_price'].std()
    price_volatility = (price_std / df['modal_price'].mean()) * 100 if df['modal_price'].mean() > 0 else 0
    
    # Determine current trend
    if len(df) >= 7:
        recent_slope = (df['modal_price'].iloc[-1] - df['modal_price'].iloc[-7]) / 7
        if recent_slope > 0.01:
            trend = "rising"
        elif recent_slope < -0.01:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "insufficient data"
    
    # Get top markets for this commodity
    top_markets_df = df.groupby('market')['modal_price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    top_markets = top_markets_df.head(5).reset_index().to_dict('records')
    
    # Get price forecasts
    # (We would use model predictions here in a real implementation)
    
    insights = {
        "commodity": commodity,
        "current_price": round(current_price, 2),
        "min_price_90d": round(min_price, 2),
        "max_price_90d": round(max_price, 2),
        "price_volatility": round(price_volatility, 2),
        "trend": trend,
        "top_markets": top_markets,
        "price_recommendation": {
            "sell_now": trend == "rising" and current_price > df['rolling_avg_30d'].iloc[-1],
            "wait": trend == "rising" and current_price <= df['rolling_avg_30d'].iloc[-1],
            "confidence": 0.7  # Simplified confidence score
        }
    }
    
    return insights

def get_market_comparison(commodity, markets, days=30):
    """
    Compare prices for a commodity across different markets
    
    Args:
        commodity (str): Commodity name
        markets (list): List of markets to compare
        days (int): Number of days of historical data to analyze
        
    Returns:
        dict: Dictionary containing market comparison data
    """
    market_data = {}
    
    for market in markets:
        df = get_commodity_data_from_db(commodity=commodity, market=market, days=days)
        if not df.empty:
            market_data[market] = {
                "avg_price": round(df['modal_price'].mean(), 2),
                "current_price": round(df['modal_price'].iloc[-1], 2) if not df.empty else 0,
                "min_price": round(df['modal_price'].min(), 2),
                "max_price": round(df['modal_price'].max(), 2)
            }
    
    # Find the best market based on current price
    best_market = max(market_data.items(), key=lambda x: x[1]["current_price"])[0] if market_data else None
    
    return {
        "commodity": commodity,
        "markets": market_data,
        "best_market": best_market,
        "price_difference": round(max([m["current_price"] for m in market_data.values()]) - 
                              min([m["current_price"] for m in market_data.values()]), 2) if market_data else 0
    }

def get_seasonal_analysis(commodity, state=None, years=3):
    """
    Analyze seasonal price patterns for a commodity
    
    Args:
        commodity (str): Commodity name
        state (str): Optional state filter
        years (int): Number of years of historical data to analyze
        
    Returns:
        dict: Dictionary containing seasonal analysis
    """
    # Get data for multiple years
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=365*years)).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM commodity_prices WHERE commodity = ?"
    params = [commodity]
    
    if state:
        query += " AND state = ?"
        params.append(state)
    
    query += " AND timestamp > ? ORDER BY arrival_date"
    params.append(cutoff_date)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {
            "status": "error",
            "message": "Insufficient historical data"
        }
    
    # Convert dates and extract month
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['month'] = df['arrival_date'].dt.month
    df['year'] = df['arrival_date'].dt.year
    
    # Calculate monthly averages
    monthly_avg = df.groupby(['year', 'month'])['modal_price'].mean().reset_index()
    
    # Calculate overall monthly averages
    overall_monthly_avg = df.groupby('month')['modal_price'].mean().reset_index()
    
    # Find best and worst months to sell
    best_month = overall_monthly_avg.loc[overall_monthly_avg['modal_price'].idxmax()]
    worst_month = overall_monthly_avg.loc[overall_monthly_avg['modal_price'].idxmin()]
    
    # Calculate percentage difference between best and worst month
    price_diff_pct = ((best_month['modal_price'] - worst_month['modal_price']) / worst_month['modal_price']) * 100
    
    # Map month numbers to names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # Create monthly data
    monthly_data = []
    for _, row in overall_monthly_avg.iterrows():
        monthly_data.append({
            "month": month_names[row['month']],
            "avg_price": round(row['modal_price'], 2)
        })
    
    return {
        "commodity": commodity,
        "best_month": {
            "month": month_names[int(best_month['month'])],
            "avg_price": round(best_month['modal_price'], 2)
        },
        "worst_month": {
            "month": month_names[int(worst_month['month'])],
            "avg_price": round(worst_month['modal_price'], 2)
        },
        "price_difference_percent": round(price_diff_pct, 2),
        "monthly_data": monthly_data
    }

def get_trending_commodities():
    """
    Get trending commodities based on price changes
    
    Returns:
        list: List of trending commodities
    """
    # Get data for the last 30 days
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT commodity, 
           AVG(modal_price) as avg_price,
           COUNT(*) as data_points
    FROM commodity_prices 
    WHERE timestamp > ?
    GROUP BY commodity
    HAVING COUNT(*) >= 10
    ORDER BY avg_price DESC
    """
    
    df = pd.read_sql_query(query, conn, params=[cutoff_date])
    conn.close()
    
    # Calculate price changes for each commodity
    trending = []
    for _, row in df.iterrows():
        commodity = row['commodity']
        
        # Get current price and price from 30 days ago
        conn = sqlite3.connect(DB_PATH)
        recent_query = """
        SELECT modal_price, arrival_date
        FROM commodity_prices
        WHERE commodity = ?
        ORDER BY arrival_date DESC
        LIMIT 1
        """
        
        old_query = """
        SELECT modal_price, arrival_date
        FROM commodity_prices
        WHERE commodity = ? AND timestamp > ?
        ORDER BY arrival_date ASC
        LIMIT 1
        """
        
        recent_price = pd.read_sql_query(recent_query, conn, params=[commodity])
        old_price = pd.read_sql_query(old_query, conn, params=[commodity, cutoff_date])
        conn.close()
        
        if not recent_price.empty and not old_price.empty:
            price_change = ((recent_price['modal_price'].iloc[0] - old_price['modal_price'].iloc[0]) / 
                            old_price['modal_price'].iloc[0]) * 100
            
            trending.append({
                "commodity": commodity,
                "current_price": round(recent_price['modal_price'].iloc[0], 2),
                "price_change_percent": round(price_change, 2),
                "trend": "up" if price_change > 0 else "down"
            })
    
    # Sort by absolute price change
    trending.sort(key=lambda x: abs(x['price_change_percent']), reverse=True)
    
    return trending[:10]  # Return top 10 trending commodities

# Flask API routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/commodities', methods=['GET'])
def get_commodities():
    """API endpoint to get list of available commodities"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT commodity FROM commodity_prices ORDER BY commodity")
    commodities = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({"commodities": commodities})

@app.route('/api/markets', methods=['GET'])
def get_markets():
    """API endpoint to get list of available markets"""
    state = request.args.get('state')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if state:
        cursor.execute("SELECT DISTINCT market FROM commodity_prices WHERE state = ? ORDER BY market", (state,))
    else:
        cursor.execute("SELECT DISTINCT market FROM commodity_prices ORDER BY market")
        
    markets = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({"markets": markets})

@app.route('/api/states', methods=['GET'])
def get_states():
    """API endpoint to get list of available states"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT state FROM commodity_prices ORDER BY state")
    states = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({"states": states})

@app.route('/api/varieties', methods=['GET'])
def get_varieties():
    """API endpoint to get list of varieties for a commodity"""
    commodity = request.args.get('commodity')
    
    if not commodity:
        return jsonify({"error": "Commodity parameter is required"}), 400
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT variety FROM commodity_prices WHERE commodity = ? ORDER BY variety", (commodity,))
    varieties = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({"varieties": varieties})

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """API endpoint to get price data"""
    commodity = request.args.get('commodity')
    state = request.args.get('state')
    market = request.args.get('market')
    days = request.args.get('days', 30, type=int)
    
    if not commodity:
        return jsonify({"error": "Commodity parameter is required"}), 400
    
    df = get_commodity_data_from_db(commodity=commodity, state=state, market=market, days=days)
    
    if df.empty:
        return jsonify({"error": "No data available for the specified parameters"}), 404
    
    # Convert to JSON format
    prices = []
    for _, row in df.iterrows():
        prices.append({
            "date": row['arrival_date'],
            "market": row['market'],
            "state": row['state'],
            "min_price": float(row['min_price']),
            "max_price": float(row['max_price']),
            "modal_price": float(row['modal_price'])
        })
    
    return jsonify({"prices": prices})

@app.route('/api/predict', methods=['GET'])
def predict_prices():
    """API endpoint to predict prices"""
    commodity = request.args.get('commodity')
    market = request.args.get('market')
    state = request.args.get('state')
    variety = request.args.get('variety', 'Other')
    days_ahead = request.args.get('days_ahead', 7, type=int)
    
    if not commodity or not market or not state:
        return jsonify({"error": "Commodity, market, and state parameters are required"}), 400
    
    # Get data for model training
    df = get_commodity_data_from_db(commodity=commodity, market=market, state=state, days=365)
    
    if len(df) < 30:
        return jsonify({
            "error": "Insufficient data for prediction",
            "message": "Need at least 30 data points for accurate prediction"
        }), 404
    
    # Train model
    X_train, X_test, y_train, y_test, encoders, scaler = prepare_data_for_model(df)
    
    if X_train is None:
        return jsonify({"error": "Failed to prepare data for model training"}), 500
    
    model = train_random_forest_model(X_train, y_train)
    
    # Make predictions
    predictions = make_price_prediction(
        model, commodity, market, state, variety, 
        encoders, scaler, days_ahead=days_ahead
    )
    
    # Get market insights
    insights = get_market_insights(commodity, market, state)
    
    return jsonify({
        "commodity": commodity,
        "market": market,
        "state": state,
        "predictions": predictions,
        "insights": insights
    })

@app.route('/api/insights', methods=['GET'])
def market_insights():
    """API endpoint to get market insights"""
    commodity = request.args.get('commodity')
    market = request.args.get('market')
    state = request.args.get('state')
    
    if not commodity:
        return jsonify({"error": "Commodity parameter is required"}), 400
