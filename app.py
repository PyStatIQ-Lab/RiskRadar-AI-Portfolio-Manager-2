import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import openpyxl
import os
import plotly.express as px
from datetime import datetime, timedelta
import requests
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import cvxpy as cp
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Portfolio Manager Pro+",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary: #4f8bf9;
        --secondary: #6c757d;
        --success: #28a745;
        --danger: #dc3545;
        --warning: #fd7e14;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #343a40;
    }
    
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid var(--primary);
    }
    
    .risk-high { color: var(--danger); font-weight: bold; }
    .risk-medium { color: var(--warning); font-weight: bold; }
    .risk-low { color: var(--success); font-weight: bold; }
    
    .health-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    
    .score-excellent { color: var(--success); }
    .score-good { color: #7CB342; }
    .score-fair { color: var(--warning); }
    .score-poor { color: var(--danger); }
    
    .news-card {
        border-left: 4px solid var(--info);
        padding: 10px 15px;
        margin-bottom: 10px;
        background: white;
        border-radius: 5px;
    }
    
    .var-critical { background-color: #ffcccc; }
    .var-warning { background-color: #fff3cd; }
    .var-safe { background-color: #d4edda; }
    
    .factor-box {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .efficient-frontier-container {
        height: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_stock_list(file_path="stocks.xlsx"):
    """Load stock symbols from Excel file"""
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path, engine='openpyxl')
            if 'Symbol' in df.columns:
                return df['Symbol'].dropna().unique().tolist()
        return []
    except Exception as e:
        st.error(f"Error loading stock list: {str(e)}")
        return []

def get_stock_data(ticker):
    """Fetch comprehensive stock data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        hist_3mo = stock.history(period="3mo")
        
        if hist.empty:
            return None
            
        # Calculate additional metrics
        info['daily_volatility'] = hist['Close'].pct_change().std() * np.sqrt(252)
        info['weekly_volatility'] = hist['Close'].resample('W').last().pct_change().std() * np.sqrt(52)
        info['monthly_volatility'] = hist['Close'].resample('M').last().pct_change().std() * np.sqrt(12)
        
        return {
            'info': info,
            'history': hist,
            'history_3mo': hist_3mo
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators from historical data"""
    if df.empty:
        return df
        
    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    # RSI
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan
    
    # Bollinger Bands
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['MA_20'] + (2 * df['Close'].rolling(window=20).std())
    df['Lower_Band'] = df['MA_20'] - (2 * df['Close'].rolling(window=20).std())
    
    return df

def calculate_portfolio_metrics(portfolio):
    """Calculate portfolio-level metrics"""
    metrics = {
        'total_value': 0,
        'total_beta': 0,
        'total_pe': 0,
        'total_debt_to_equity': 0,
        'total_roe': 0,
        'sector_exposure': {},
        'correlation_matrix': None,
        'individual_weights': {},
        'volatility_metrics': {},
        'health_score': 0,
        'returns_data': None,
        'cov_matrix': None
    }
    
    close_prices = pd.DataFrame()
    total_value = sum([h['value'] for h in portfolio.values() if h['data'] is not None])
    
    for ticker, holding in portfolio.items():
        if holding['data'] is None:
            continue
            
        weight = holding['value'] / total_value if total_value > 0 else 0
        metrics['individual_weights'][ticker] = weight
        
        # Aggregate portfolio metrics (weighted)
        info = holding['data']['info']
        metrics['total_beta'] += info.get('beta', 0) * weight
        metrics['total_pe'] += info.get('trailingPE', 0) * weight
        metrics['total_debt_to_equity'] += info.get('debtToEquity', 0) * weight
        metrics['total_roe'] += info.get('returnOnEquity', 0) * weight
        
        # Track sector exposure
        sector = info.get('sector', 'Unknown')
        metrics['sector_exposure'][sector] = metrics['sector_exposure'].get(sector, 0) + weight
        
        # Add to correlation matrix
        close_prices[ticker] = holding['data']['history']['Close']
        
        # Volatility metrics
        metrics['volatility_metrics'][ticker] = {
            'daily': info.get('daily_volatility', 0),
            'weekly': info.get('weekly_volatility', 0),
            'monthly': info.get('monthly_volatility', 0)
        }
    
    metrics['total_value'] = total_value
    
    # Calculate correlation matrix and covariance
    if not close_prices.empty:
        returns = close_prices.pct_change().dropna()
        metrics['returns_data'] = returns
        metrics['correlation_matrix'] = returns.corr()
        metrics['cov_matrix'] = returns.cov()
    
    # Calculate portfolio health score (0-100)
    health_score = 100
    # Deduct for high beta
    if metrics['total_beta'] > 1.2: health_score -= 15
    elif metrics['total_beta'] < 0.8: health_score -= 5
    # Deduct for high P/E
    if metrics['total_pe'] > 25: health_score -= 10
    # Deduct for sector concentration
    if len(metrics['sector_exposure']) < 3: health_score -= 10
    # Deduct for high correlation
    if metrics['correlation_matrix'] is not None:
        avg_corr = metrics['correlation_matrix'].values.mean()
        if avg_corr > 0.7: health_score -= 10
    metrics['health_score'] = max(0, health_score)
    
    return metrics

def calculate_var(returns, method='historical', confidence_level=0.95, days=1):
    """Calculate Value at Risk using different methods"""
    if returns is None or len(returns) == 0:
        return None
    
    if method == 'historical':
        # Historical VaR
        return np.percentile(returns, 100 * (1 - confidence_level)) * np.sqrt(days)
    
    elif method == 'parametric':
        # Parametric (Normal distribution) VaR
        mean = np.mean(returns)
        std_dev = np.std(returns)
        return (mean - std_dev * norm.ppf(confidence_level)) * np.sqrt(days)
    
    elif method == 'monte_carlo':
        # Monte Carlo VaR
        mean = np.mean(returns)
        std_dev = np.std(returns)
        simulations = np.random.normal(mean, std_dev, 10000)
        return np.percentile(simulations, 100 * (1 - confidence_level)) * np.sqrt(days)
    
    return None

def calculate_cvar(returns, confidence_level=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if returns is None or len(returns) == 0:
        return None
    
    var = calculate_var(returns, 'historical', confidence_level)
    return np.mean(returns[returns <= var])

def stress_test(returns, scenario='2008'):
    """Apply stress test scenarios to portfolio returns"""
    if returns is None:
        return None
    
    if scenario == '2008':
        # 2008 financial crisis scenario (approximate)
        stress_factor = -0.40  # 40% drop
    elif scenario == 'covid':
        # COVID-19 scenario
        stress_factor = -0.30  # 30% drop
    elif scenario == 'dotcom':
        # Dot-com bubble
        stress_factor = -0.45  # 45% drop
    else:
        stress_factor = -0.20  # Default 20% drop
    
    stressed_returns = returns * (1 + stress_factor)
    return stressed_returns.mean()

def calculate_hhi(weights):
    """Calculate Herfindahl-Hirschman Index for concentration"""
    return np.sum(np.square(weights)) * 10000

def calculate_factor_exposure(returns, factors):
    """Calculate factor exposures using regression"""
    if returns is None or factors is None:
        return None
    
    try:
        factors = sm.add_constant(factors)
        model = sm.OLS(returns, factors).fit()
        return model.params
    except:
        return None

def mean_variance_optimization(returns, cov_matrix, target_return=None, risk_free_rate=0.0):
    """Perform mean-variance optimization"""
    n_assets = len(returns)
    
    def portfolio_stats(weights):
        port_return = np.sum(returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol
        return port_return, port_vol, sharpe
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(n_assets))
    
    if target_return is not None:
        # Target return optimization
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * returns) - target_return}
        )
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                         n_assets * [1. / n_assets],
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    else:
        # Max Sharpe optimization
        result = minimize(lambda x: -portfolio_stats(x)[2],
                         n_assets * [1. / n_assets],
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    
    return result.x if result.success else None

def risk_parity_allocation(cov_matrix):
    """Calculate risk parity allocation"""
    n = cov_matrix.shape[0]
    weights = cp.Variable(n)
    risk_contributions = []
    
    for i in range(n):
        rc = weights[i] * (cov_matrix @ weights)[i] / cp.quad_form(weights, cov_matrix)
        risk_contributions.append(rc)
    
    objective = cp.Minimize(cp.sum_squares(cp.hstack(risk_contributions) - 1/n))
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return weights.value if problem.status == cp.OPTIMAL else None

def hierarchical_risk_parity(cov_matrix):
    """Hierarchical Risk Parity allocation"""
    # Step 1: Hierarchical clustering
    corr_matrix = cov_to_corr(cov_matrix)
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    np.fill_diagonal(dist_matrix, 0)
    
    # Step 2: Quasi-diagonalization
    linkage = sch.linkage(dist_matrix, method='single')
    sort_idx = sch.dendrogram(linkage, no_plot=True)['leaves']
    sorted_corr = corr_matrix.iloc[sort_idx, sort_idx]
    
    # Step 3: Recursive bisection
    weights = pd.Series(1, index=sorted_corr.index)
    clusters = [weights.index]
    
    while len(clusters) > 0:
        cluster = clusters.pop(0)
        if len(cluster) == 1:
            continue
            
        # Split cluster into two sub-clusters
        sub_cluster1 = cluster[:len(cluster)//2]
        sub_cluster2 = cluster[len(cluster)//2:]
        
        # Allocate weights based on inverse variance
        var1 = cov_matrix.loc[sub_cluster1, sub_cluster1].mean().mean()
        var2 = cov_matrix.loc[sub_cluster2, sub_cluster2].mean().mean()
        
        total_var = var1 + var2
        alpha = 1 - var1 / total_var
        
        weights[sub_cluster1] *= alpha
        weights[sub_cluster2] *= (1 - alpha)
        
        clusters += [sub_cluster1, sub_cluster2]
    
    return weights / weights.sum()

def cov_to_corr(cov_matrix):
    """Convert covariance matrix to correlation matrix"""
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.columns)

def black_litterman(returns, cov_matrix, tau=0.05, views=None, P=None, Q=None):
    """Black-Litterman model for incorporating views"""
    if views is None:
        return returns
    
    # Market equilibrium returns (CAPM)
    pi = returns
    
    # Omega - uncertainty in views (proportional to variance)
    omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))
    
    # Black-Litterman formula
    try:
        first_term = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
        second_term = np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q
        new_returns = first_term @ second_term
    except:
        new_returns = pi
    
    return new_returns

def generate_ai_insights(portfolio, portfolio_metrics):
    """Generate AI-powered insights based on the data"""
    insights = []
    warnings = []
    suggestions = []
    report_cards = {}
    
    # Portfolio-level insights
    if portfolio_metrics['total_beta'] > 1.2:
        insights.append("🔴 Your portfolio has higher-than-market risk (Beta = {:.2f}). Consider adding defensive stocks.".format(portfolio_metrics['total_beta']))
    elif portfolio_metrics['total_beta'] < 0.8:
        insights.append("🟢 Your portfolio has lower-than-market risk (Beta = {:.2f}). You may be under-exposed to market upside.".format(portfolio_metrics['total_beta']))
    
    if portfolio_metrics['total_pe'] > 25:
        warnings.append("⚠️ Portfolio appears overvalued (Avg P/E = {:.1f}). Look for value opportunities.".format(portfolio_metrics['total_pe']))
    
    # Sector concentration warning
    if len(portfolio_metrics['sector_exposure']) < 3:
        main_sector = max(portfolio_metrics['sector_exposure'], key=portfolio_metrics['sector_exposure'].get)
        warnings.append(f"⚠️ High concentration in {main_sector} sector ({portfolio_metrics['sector_exposure'][main_sector]*100:.0f}%). Consider diversifying.")
    
    # Stock-specific analysis
    for ticker, holding in portfolio.items():
        if holding['data'] is None:
            warnings.append(f"⚠️ Could not fetch data for {ticker}")
            continue
            
        info = holding['data']['info']
        hist = holding['data']['history']
        report_card = {
            'valuation': {},
            'profitability': {},
            'risk': {},
            'financial_health': {},
            'cash_flow': {},
            'dividends': {},
            'technical': {}
        }
        
        # Valuation Metrics
        pe = info.get('trailingPE', 0)
        report_card['valuation']['P/E'] = {'value': pe, 'status': 'high' if pe > 25 else 'medium' if pe > 15 else 'low'}
        
        pb = info.get('priceToBook', 0)
        report_card['valuation']['P/B'] = {'value': pb, 'status': 'high' if pb > 3 else 'medium' if pb > 1.5 else 'low'}
        
        ps = info.get('priceToSalesTrailing12Months', 0)
        report_card['valuation']['P/S'] = {'value': ps, 'status': 'high' if ps > 5 else 'medium' if ps > 2 else 'low'}
        
        # Profitability Metrics
        roe = info.get('returnOnEquity', 0)
        report_card['profitability']['ROE'] = {'value': roe, 'status': 'high' if roe > 0.15 else 'medium' if roe > 0.1 else 'low'}
        
        profit_margin = info.get('profitMargins', 0)
        report_card['profitability']['Profit Margin'] = {'value': profit_margin, 'status': 'high' if profit_margin > 0.15 else 'medium' if profit_margin > 0.1 else 'low'}
        
        # Risk Metrics
        beta = info.get('beta', 0)
        report_card['risk']['Beta'] = {'value': beta, 'status': 'high' if beta > 1.2 else 'low' if beta < 0.8 else 'medium'}
        
        volatility = info.get('daily_volatility', 0)
        report_card['risk']['Volatility'] = {'value': volatility, 'status': 'high' if volatility > 0.3 else 'medium' if volatility > 0.2 else 'low'}
        
        # Financial Health
        de = info.get('debtToEquity', 0)
        report_card['financial_health']['Debt/Equity'] = {'value': de, 'status': 'high' if de > 1.5 else 'medium' if de > 0.5 else 'low'}
        
        current_ratio = info.get('currentRatio', 0)
        report_card['financial_health']['Current Ratio'] = {'value': current_ratio, 'status': 'high' if current_ratio > 2 else 'medium' if current_ratio > 1 else 'low'}
        
        # Cash Flow & Earnings
        fcf = info.get('freeCashflow', 0)
        report_card['cash_flow']['Free Cash Flow'] = {'value': fcf, 'status': 'high' if fcf > 1e9 else 'medium' if fcf > 5e8 else 'low'}
        
        # Dividends
        div_yield = info.get('dividendYield', 0)
        report_card['dividends']['Dividend Yield'] = {'value': div_yield, 'status': 'high' if div_yield > 0.04 else 'medium' if div_yield > 0.02 else 'low'}
        
        # Technical Analysis
        if 'RSI' in hist.columns and not pd.isna(hist['RSI'].iloc[-1]):
            last_rsi = hist['RSI'].iloc[-1]
            report_card['technical']['RSI'] = {'value': last_rsi, 'status': 'high' if last_rsi > 70 else 'low' if last_rsi < 30 else 'medium'}
        
        report_cards[ticker] = report_card
        
        # Generate warnings and suggestions based on report card
        if pe > 30 and pe > info.get('industryPE', 100):
            warnings.append(f"⚠️ {ticker}: High P/E ratio ({pe:.1f}) compared to industry")
        
        if de > 1.5:
            warnings.append(f"⚠️ {ticker}: High debt-to-equity ratio ({de:.2f})")
        
        if 'RSI' in report_card['technical'] and report_card['technical']['RSI']['status'] == 'high':
            warnings.append(f"⚠️ {ticker}: Overbought (RSI = {report_card['technical']['RSI']['value']:.1f}) - Consider profit booking")
        
        if 'MA_50' in hist.columns and 'MA_200' in hist.columns:
            if len(hist) >= 2:
                if hist['MA_50'].iloc[-1] < hist['MA_200'].iloc[-1] and hist['MA_50'].iloc[-2] >= hist['MA_200'].iloc[-2]:
                    suggestions.append(f"🔴 Consider exiting {ticker} - Death Cross detected (50MA crossed below 200MA)")
    
    return {
        'insights': insights,
        'warnings': warnings,
        'suggestions': suggestions,
        'report_cards': report_cards
    }

def get_news_sentiment(ticker):
    """Optional: Get news sentiment for a stock"""
    # This is a placeholder - you would integrate with a news API
    return {
        'sentiment': 'neutral',
        'summary': 'No major news events recently'
    }

def display_risk_analytics(portfolio_metrics):
    """Display advanced risk analytics section"""
    st.subheader("📉 Advanced Risk Analytics", divider="blue")
    
    if portfolio_metrics['returns_data'] is None:
        st.warning("Insufficient data for advanced risk analysis")
        return
    
    returns = portfolio_metrics['returns_data']
    cov_matrix = portfolio_metrics['cov_matrix']
    weights = np.array(list(portfolio_metrics['individual_weights'].values()))
    
    # Value at Risk Analysis
    st.markdown("### Value-at-Risk (VaR) Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Historical VaR")
        var_95 = calculate_var(returns.mean(axis=1), 'historical', 0.95)
        var_99 = calculate_var(returns.mean(axis=1), 'historical', 0.99)
        st.metric("1-day 95% VaR", f"{var_95*100:.2f}%", delta_color="inverse")
        st.metric("1-day 99% VaR", f"{var_99*100:.2f}%", delta_color="inverse")
    
    with col2:
        st.markdown("#### Parametric VaR")
        pvar_95 = calculate_var(returns.mean(axis=1), 'parametric', 0.95)
        pvar_99 = calculate_var(returns.mean(axis=1), 'parametric', 0.99)
        st.metric("1-day 95% VaR", f"{pvar_95*100:.2f}%", delta_color="inverse")
        st.metric("1-day 99% VaR", f"{pvar_99*100:.2f}%", delta_color="inverse")
    
    with col3:
        st.markdown("#### Expected Shortfall (CVaR)")
        cvar_95 = calculate_cvar(returns.mean(axis=1), 0.95)
        cvar_99 = calculate_cvar(returns.mean(axis=1), 0.99)
        st.metric("1-day 95% CVaR", f"{cvar_95*100:.2f}%", delta_color="inverse")
        st.metric("1-day 99% CVaR", f"{cvar_99*100:.2f}%", delta_color="inverse")
    
    # Stress Testing
    st.markdown("### Stress Testing Scenarios")
    stress_cols = st.columns(4)
    
    with stress_cols[0]:
        st.markdown("#### 2008 Crisis")
        stress_return = stress_test(returns.mean(axis=1), '2008')
        st.metric("Expected Return", f"{stress_return*100:.2f}%", delta_color="inverse")
    
    with stress_cols[1]:
        st.markdown("#### COVID-19")
        stress_return = stress_test(returns.mean(axis=1), 'covid')
        st.metric("Expected Return", f"{stress_return*100:.2f}%", delta_color="inverse")
    
    with stress_cols[2]:
        st.markdown("#### Dot-com Bubble")
        stress_return = stress_test(returns.mean(axis=1), 'dotcom')
        st.metric("Expected Return", f"{stress_return*100:.2f}%", delta_color="inverse")
    
    with stress_cols[3]:
        st.markdown("#### Custom Scenario")
        custom_shock = st.slider("Shock Percentage", -50, 0, -20, key="custom_shock") / 100
        stress_return = returns.mean(axis=1).mean() * (1 + custom_shock)
        st.metric("Expected Return", f"{stress_return*100:.2f}%", delta_color="inverse")
    
    # Factor Analysis
    st.markdown("### Factor Risk Analysis")
    
    # Simulate some factors (in a real app, you'd get these from a data provider)
    dates = returns.index
    factors = pd.DataFrame({
        'Market': np.random.normal(0.0005, 0.01, len(dates)),
        'Size': np.random.normal(0.0002, 0.005, len(dates)),
        'Value': np.random.normal(0.0003, 0.007, len(dates)),
        'Momentum': np.random.normal(0.0004, 0.008, len(dates))
    }, index=dates)
    
    factor_exposures = calculate_factor_exposure(returns.mean(axis=1), factors)
    if factor_exposures is not None:
        fig = px.bar(x=factor_exposures.index[1:], y=factor_exposures.values[1:], 
                     labels={'x': 'Factor', 'y': 'Exposure'},
                     title="Factor Exposures")
        st.plotly_chart(fig, use_container_width=True)
    
    # Liquidity Analysis
    st.markdown("### Liquidity Risk Metrics")
    liq_cols = st.columns(3)
    
    with liq_cols[0]:
        st.metric("Portfolio Turnover", "0.5%", "Low")
    
    with liq_cols[1]:
        st.metric("Avg Bid-Ask Spread", "0.2%", "Low")
    
    with liq_cols[2]:
        st.metric("Market Impact Cost", "0.3%", "Medium")
    
    # Concentration Analysis
    st.markdown("### Concentration Risk")
    conc_cols = st.columns(2)
    
    with conc_cols[0]:
        hhi = calculate_hhi(weights)
        st.metric("Herfindahl-Hirschman Index", f"{hhi:.0f}", 
                 "High" if hhi > 1500 else "Medium" if hhi > 1000 else "Low")
    
    with conc_cols[1]:
        top3_weight = sum(sorted(weights, reverse=True)[:3])
        st.metric("Top 3 Holdings Weight", f"{top3_weight*100:.1f}%", 
                 "High" if top3_weight > 0.5 else "Medium" if top3_weight > 0.3 else "Low")

def display_portfolio_optimization(portfolio_metrics):
    """Display portfolio optimization tools"""
    st.subheader("⚙️ Portfolio Optimization", divider="blue")
    
    if portfolio_metrics['returns_data'] is None:
        st.warning("Insufficient data for portfolio optimization")
        return
    
    returns = portfolio_metrics['returns_data']
    mean_returns = returns.mean()
    cov_matrix = portfolio_metrics['cov_matrix']
    tickers = list(portfolio_metrics['individual_weights'].keys())
    
    tab1, tab2, tab3, tab4 = st.tabs(["Mean-Variance", "Black-Litterman", "Risk Parity", "Hierarchical Risk Parity"])
    
    with tab1:
        st.markdown("#### Mean-Variance Optimization")
        
        target_return = st.slider("Target Annual Return (%)", 
                                min_value=0.0, 
                                max_value=50.0, 
                                value=10.0, 
                                step=0.5) / 100
        
        opt_weights = mean_variance_optimization(mean_returns, cov_matrix, target_return)
        
        if opt_weights is not None:
            opt_df = pd.DataFrame({
                'Ticker': tickers,
                'Current Weight': [portfolio_metrics['individual_weights'][t] for t in tickers],
                'Optimal Weight': opt_weights
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=opt_df['Ticker'],
                y=opt_df['Current Weight'],
                name='Current Weight',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=opt_df['Ticker'],
                y=opt_df['Optimal Weight'],
                name='Optimal Weight',
                marker_color='royalblue'
            ))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Efficient Frontier
            st.markdown("#### Efficient Frontier")
            
            # Generate random portfolios
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                port_return = np.sum(mean_returns * weights) * 252
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                results[0,i] = port_return
                results[1,i] = port_vol
                results[2,i] = (port_return - 0.02) / port_vol  # Sharpe ratio
            
            # Create DataFrame
            results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])
            
            # Plot efficient frontier
            fig = px.scatter(results_df, x='Volatility', y='Return', color='Sharpe',
                           title='Efficient Frontier',
                           labels={'Volatility': 'Annualized Volatility', 
                                  'Return': 'Annualized Return'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Black-Litterman Model")
        st.info("Incorporate your views to adjust market equilibrium returns")
        
        # Create views matrix
        st.markdown("##### Express Your Views")
        view_tickers = st.multiselect("Select tickers for views", tickers)
        
        P = np.zeros((len(view_tickers), len(tickers)))
        Q = np.zeros(len(view_tickers))
        
        for i, ticker in enumerate(view_tickers):
            col1, col2 = st.columns([1, 3])
            with col1:
                relative_return = st.number_input(f"View on {ticker} (%)", 
                                                min_value=-20.0, 
                                                max_value=20.0, 
                                                value=2.0, 
                                                step=0.5,
                                                key=f"view_{ticker}")
                Q[i] = relative_return / 100
            with col2:
                st.write(f"Example: '{ticker} will outperform by {relative_return}%'")
            
            P[i, tickers.index(ticker)] = 1
        
        if st.button("Optimize with Views"):
            bl_returns = black_litterman(mean_returns, cov_matrix, views=view_tickers, P=P, Q=Q)
            opt_weights = mean_variance_optimization(bl_returns, cov_matrix)
            
            if opt_weights is not None:
                opt_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Current Weight': [portfolio_metrics['individual_weights'][t] for t in tickers],
                    'Optimal Weight': opt_weights
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=opt_df['Ticker'],
                    y=opt_df['Current Weight'],
                    name='Current Weight',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=opt_df['Ticker'],
                    y=opt_df['Optimal Weight'],
                    name='Optimal Weight',
                    marker_color='royalblue'
                ))
                fig.update_layout(barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Risk Parity Allocation")
        rp_weights = risk_parity_allocation(cov_matrix)
        
        if rp_weights is not None:
            rp_df = pd.DataFrame({
                'Ticker': tickers,
                'Current Weight': [portfolio_metrics['individual_weights'][t] for t in tickers],
                'Risk Parity Weight': rp_weights
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rp_df['Ticker'],
                y=rp_df['Current Weight'],
                name='Current Weight',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=rp_df['Ticker'],
                y=rp_df['Risk Parity Weight'],
                name='Risk Parity Weight',
                marker_color='royalblue'
            ))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Hierarchical Risk Parity")
        hrp_weights = hierarchical_risk_parity(cov_matrix)
        
        if hrp_weights is not None:
            hrp_df = pd.DataFrame({
                'Ticker': tickers,
                'Current Weight': [portfolio_metrics['individual_weights'][t] for t in tickers],
                'HRP Weight': hrp_weights
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hrp_df['Ticker'],
                y=hrp_df['Current Weight'],
                name='Current Weight',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=hrp_df['Ticker'],
                y=hrp_df['HRP Weight'],
                name='HRP Weight',
                marker_color='royalblue'
            ))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

def display_advanced_analytics(portfolio_metrics):
    """Display advanced analytics section"""
    st.subheader("🔍 Advanced Analytics", divider="blue")
    
    if portfolio_metrics['returns_data'] is None:
        st.warning("Insufficient data for advanced analytics")
        return
    
    returns = portfolio_metrics['returns_data']
    mean_returns = returns.mean()
    cov_matrix = portfolio_metrics['cov_matrix']
    tickers = list(portfolio_metrics['individual_weights'].keys())
    
    tab1, tab2, tab3, tab4 = st.tabs(["Regime Detection", "Tail Risk Hedging", "Transaction Cost Modeling", "Tax Optimization"])
    
    with tab1:
        st.markdown("#### Market Regime Detection")
        
        # Simple regime detection based on volatility
        rolling_vol = returns.mean(axis=1).rolling(20).std()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=rolling_vol,
            name='20-day Rolling Volatility',
            line=dict(color='royalblue')
        ))
        
        # Add regime thresholds
        high_vol = rolling_vol.quantile(0.75)
        low_vol = rolling_vol.quantile(0.25)
        
        fig.add_hline(y=high_vol, line_dash="dot", 
                     annotation_text="High Volatility Regime", 
                     annotation_position="bottom right",
                     line_color="red")
        fig.add_hline(y=low_vol, line_dash="dot", 
                     annotation_text="Low Volatility Regime", 
                     annotation_position="top right",
                     line_color="green")
        
        fig.update_layout(title="Volatility Regime Detection",
                         yaxis_title="Volatility",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Regime-Based Strategy Suggestions:**
        - High Volatility: Reduce risk exposure, increase cash or hedges
        - Low Volatility: Consider leveraging or option strategies
        """)
    
    with tab2:
        st.markdown("#### Tail Risk Hedging Simulator")
        
        hedge_type = st.selectbox("Hedge Type", 
                                ["Put Options", "VIX Futures", "Gold", "Long Volatility ETFs"])
        
        hedge_cost = st.slider("Hedge Cost (% of portfolio)", 
                             min_value=0.1, 
                             max_value=5.0, 
                             value=1.0, 
                             step=0.1)
        
        protection_level = st.slider("Protection Level (% drop)", 
                                   min_value=5, 
                                   max_value=30, 
                                   value=15, 
                                   step=1)
        
        # Simulate hedge effectiveness
        stress_return = stress_test(returns.mean(axis=1), '2008')
        hedge_payout = max(0, (-stress_return * 100) - protection_level) / 100
        net_effect = hedge_payout - (hedge_cost / 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Loss in Crisis", f"{-stress_return*100:.1f}%")
        with col2:
            st.metric("Hedge Payout", f"{hedge_payout*100:.1f}%")
        
        st.metric("Net Effect", f"{net_effect*100:.1f}%", 
                 delta_color="normal" if net_effect > 0 else "inverse")
    
    with tab3:
        st.markdown("#### Transaction Cost Modeling")
        
        trade_size = st.number_input("Trade Size (% of portfolio)", 
                                   min_value=0.1, 
                                   max_value=100.0, 
                                   value=5.0, 
                                   step=0.1)
        
        liquidity_tier = st.selectbox("Liquidity Tier", 
                                    ["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"])
        
        # Estimate costs
        if liquidity_tier == "Large Cap":
            impact_cost = 0.001 * trade_size
            spread_cost = 0.0005
        elif liquidity_tier == "Mid Cap":
            impact_cost = 0.002 * trade_size
            spread_cost = 0.001
        elif liquidity_tier == "Small Cap":
            impact_cost = 0.005 * trade_size
            spread_cost = 0.002
        else:
            impact_cost = 0.01 * trade_size
            spread_cost = 0.005
        
        total_cost = impact_cost + spread_cost
        
        st.metric("Estimated Market Impact", f"{impact_cost*100:.3f}%")
        st.metric("Estimated Spread Cost", f"{spread_cost*100:.3f}%")
        st.metric("Total Implementation Cost", f"{total_cost*100:.3f}%", delta_color="inverse")
    
    with tab4:
        st.markdown("#### Tax Optimization")
        
        holding_period = st.selectbox("Holding Period", 
                                    ["<1 year", "1-3 years", "3-5 years", "5+ years"])
        
        tax_rate = 0.2  # Default long-term rate
        if holding_period == "<1 year":
            tax_rate = 0.4
        elif holding_period == "1-3 years":
            tax_rate = 0.3
        
        unrealized_gain = st.number_input("Unrealized Gain (%)", 
                                        min_value=0.0, 
                                        max_value=500.0, 
                                        value=20.0, 
                                        step=0.1)
        
        tax_drag = unrealized_gain * tax_rate / 100
        
        st.metric("Estimated Tax Rate", f"{tax_rate*100:.0f}%")
        st.metric("Tax Drag on Returns", f"{tax_drag:.1f}%", delta_color="inverse")
        
        st.info("""
        **Tax Optimization Strategies:**
        - Harvest tax losses to offset gains
        - Consider holding periods to qualify for long-term rates
        - Use tax-advantaged accounts where possible
        """)

# Main app
def main():
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}
    
    # Load stock list from Excel
    stock_list = load_stock_list()
    
    # Sidebar for user input
    with st.sidebar:
        st.image("https://via.placeholder.com/200x50?text=Portfolio+Pro", use_column_width=True)
        st.header("Portfolio Setup")
        
        # Stock selection and quantity input
        if stock_list:
            selected_stock = st.selectbox("Select Stock", stock_list)
            quantity = st.number_input("Quantity", min_value=1, value=100)
            
            if st.button("Add to Portfolio"):
                with st.spinner(f"Fetching data for {selected_stock}..."):
                    stock_data = get_stock_data(selected_stock)
                    if stock_data:
                        current_price = stock_data['info'].get('currentPrice', stock_data['info'].get('regularMarketPrice', 0))
                        value = current_price * quantity
                        st.session_state.portfolio[selected_stock] = {
                            'quantity': quantity,
                            'value': value,
                            'data': stock_data
                        }
                        st.success(f"Added {quantity} shares of {selected_stock} to portfolio")
                    else:
                        st.error(f"Failed to fetch data for {selected_stock}")
        
        # Portfolio summary in sidebar
        if st.session_state.portfolio:
            st.subheader("Your Portfolio")
            total_value = sum([h['value'] for h in st.session_state.portfolio.values() if h['data'] is not None])
            st.metric("Total Value", f"₹{total_value:,.2f}" if total_value > 0 else "₹0")
            
            for ticker, holding in st.session_state.portfolio.items():
                if holding['data'] is not None:
                    st.markdown(f"**{ticker}**: {holding['quantity']} shares (₹{holding['value']:,.2f})")

    # Main content area
    st.title("📊 AI Portfolio Manager Pro+")
    st.caption("Advanced portfolio analysis with comprehensive risk assessment")
    
    if not st.session_state.portfolio:
        st.info("💡 Add stocks to your portfolio using the sidebar to begin analysis")
        return
    
    # Calculate portfolio metrics
    with st.spinner("Analyzing your portfolio..."):
        portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio)
        ai_output = generate_ai_insights(st.session_state.portfolio, portfolio_metrics)
    
    # Portfolio Health Score
    st.subheader("🏆 Portfolio Health Score", divider="blue")
    health_score = portfolio_metrics['health_score']
    score_class = "score-excellent" if health_score >= 85 else "score-good" if health_score >= 70 else "score-fair" if health_score >= 50 else "score-poor"
    st.markdown(f"<div class='health-score {score_class}'>{health_score:.0f}/100</div>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Diversification", "Good" if len(portfolio_metrics['sector_exposure']) >= 3 else "Needs Improvement")
    with cols[1]:
        st.metric("Risk Profile", "Aggressive" if portfolio_metrics['total_beta'] > 1.2 else "Defensive" if portfolio_metrics['total_beta'] < 0.8 else "Moderate")
    with cols[2]:
        st.metric("Valuation", "Overvalued" if portfolio_metrics['total_pe'] > 25 else "Undervalued" if portfolio_metrics['total_pe'] < 15 else "Fair")
    
    # Risk Analysis Section
    st.subheader("⚠️ Risk Analysis", divider="blue")
    
    if ai_output['warnings']:
        st.warning("### Immediate Risk Warnings")
        for warning in ai_output['warnings']:
            st.write(f"- {warning}")
    else:
        st.success("No critical risk warnings detected")
    
    # AI Insights Section
    st.subheader("🤖 AI Insights", divider="blue")
    
    if ai_output['insights'] or ai_output['suggestions']:
        tab1, tab2 = st.tabs(["Market Insights", "Actionable Suggestions"])
        
        with tab1:
            for insight in ai_output['insights']:
                st.info(insight)
        
        with tab2:
            for suggestion in ai_output['suggestions']:
                st.success(suggestion)
    else:
        st.info("No specific insights or suggestions at this time")
    
    # Portfolio Composition
    st.subheader("🧩 Portfolio Composition", divider="blue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sector Exposure")
        if portfolio_metrics['sector_exposure']:
            fig = go.Figure(go.Pie(
                labels=list(portfolio_metrics['sector_exposure'].keys()),
                values=list(portfolio_metrics['sector_exposure'].values()),
                hole=0.3,
                marker_colors=px.colors.qualitative.Pastel
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Stock Allocation")
        weights_df = pd.DataFrame.from_dict(portfolio_metrics['individual_weights'], 
                                          orient='index', columns=['Weight'])
        fig = go.Figure(go.Pie(
            labels=weights_df.index,
            values=weights_df['Weight'],
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display all the advanced analytics sections
    display_risk_analytics(portfolio_metrics)
    display_portfolio_optimization(portfolio_metrics)
    display_advanced_analytics(portfolio_metrics)
    
    # Stock Report Cards
    st.subheader("📋 Stock Report Cards", divider="blue")
    
    for ticker, report_card in ai_output['report_cards'].items():
        with st.expander(f"📌 {ticker} - Detailed Analysis"):
            tabs = st.tabs(["Valuation", "Profitability", "Risk", "Financial Health", "Cash Flow", "Dividends", "Technical"])
            
            with tabs[0]:
                st.markdown("#### Valuation Metrics")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['valuation'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"{data['value']:.2f}",
                            data['status'].capitalize(),
                            delta_color="inverse" if data['status'] == 'high' else "off"
                        )
            
            with tabs[1]:
                st.markdown("#### Profitability Metrics")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['profitability'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"{data['value']:.2%}",
                            data['status'].capitalize()
                        )
            
            with tabs[2]:
                st.markdown("#### Risk Metrics")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['risk'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"{data['value']:.2f}",
                            data['status'].capitalize(),
                            delta_color="inverse" if data['status'] == 'high' else "off"
                        )
            
            with tabs[3]:
                st.markdown("#### Financial Health")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['financial_health'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"{data['value']:.2f}",
                            data['status'].capitalize(),
                            delta_color="inverse" if metric == 'Debt/Equity' and data['status'] == 'high' else "normal"
                        )
            
            with tabs[4]:
                st.markdown("#### Cash Flow & Earnings")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['cash_flow'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"₹{data['value']/1e6:,.1f}M" if data['value'] > 1e6 else f"₹{data['value']:,.0f}",
                            data['status'].capitalize()
                        )
            
            with tabs[5]:
                st.markdown("#### Dividend Metrics")
                cols = st.columns(3)
                for i, (metric, data) in enumerate(report_card['dividends'].items()):
                    with cols[i % 3]:
                        st.metric(
                            metric,
                            f"{data['value']:.2%}",
                            data['status'].capitalize()
                        )
            
            with tabs[6]:
                st.markdown("#### Technical Analysis")
                if report_card['technical']:
                    cols = st.columns(3)
                    for i, (metric, data) in enumerate(report_card['technical'].items()):
                        with cols[i % 3]:
                            st.metric(
                                metric,
                                f"{data['value']:.1f}",
                                data['status'].capitalize(),
                                delta_color="inverse" if data['status'] == 'high' else "normal"
                            )
                else:
                    st.info("No technical indicators available")
                
                # Price chart
                hist = st.session_state.portfolio[ticker]['data']['history']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'))
                if 'MA_50' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_50'], name='50-Day MA'))
                if 'MA_200' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_200'], name='200-Day MA'))
                if 'Upper_Band' in hist.columns and 'Lower_Band' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Upper_Band'], name='Upper Bollinger Band', line=dict(color='rgba(255,0,0,0.3)')))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Lower_Band'], name='Lower Bollinger Band', line=dict(color='rgba(0,255,0,0.3)', fill='tonexty')))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Optional News Sentiment
            if st.checkbox("Show News Sentiment (Simulated)", key=f"news_{ticker}"):
                sentiment = get_news_sentiment(ticker)
                st.markdown(f"**Sentiment**: {sentiment['sentiment'].capitalize()}")
                st.markdown(f"**Summary**: {sentiment['summary']}")

if __name__ == "__main__":
    main()
