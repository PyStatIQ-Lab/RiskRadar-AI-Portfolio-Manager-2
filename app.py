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

# Configure page
st.set_page_config(
    page_title="AI Portfolio Manager Pro",
    layout="wide",
    page_icon="📈",
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
    
    .main {
        background-color: #f5f7fa;
    }
    
    .st-bw {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .header {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 4px solid var(--primary);
    }
    
    .risk-high {
        color: var(--danger);
        font-weight: bold;
    }
    
    .risk-medium {
        color: var(--warning);
        font-weight: bold;
    }
    
    .risk-low {
        color: var(--success);
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background-color: white;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    .stock-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
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
        
        if hist.empty:
            return None
            
        return {'info': info, 'history': hist}
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators from historical data"""
    if df.empty:
        return df
        
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    # Calculate RSI only if we have enough data
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = np.nan
    
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
        'individual_weights': {}
    }
    
    close_prices = pd.DataFrame()
    total_value = sum([h['value'] for h in portfolio.values()])
    
    for ticker, holding in portfolio.items():
        if holding['data'] is None:
            continue
            
        weight = holding['value'] / total_value if total_value > 0 else 0
        metrics['individual_weights'][ticker] = weight
        
        # Aggregate portfolio metrics (weighted)
        metrics['total_beta'] += holding['data']['info'].get('beta', 0) * weight
        metrics['total_pe'] += holding['data']['info'].get('trailingPE', 0) * weight
        metrics['total_debt_to_equity'] += holding['data']['info'].get('debtToEquity', 0) * weight
        metrics['total_roe'] += holding['data']['info'].get('returnOnEquity', 0) * weight
        
        # Track sector exposure
        sector = holding['data']['info'].get('sector', 'Unknown')
        metrics['sector_exposure'][sector] = metrics['sector_exposure'].get(sector, 0) + weight
        
        # Add to correlation matrix
        close_prices[ticker] = holding['data']['history']['Close']
    
    metrics['total_value'] = total_value
    
    # Calculate correlation matrix
    if not close_prices.empty:
        metrics['correlation_matrix'] = close_prices.corr()
    
    return metrics

def generate_ai_insights(portfolio, portfolio_metrics):
    """Generate AI-powered insights based on the data"""
    insights = []
    warnings = []
    suggestions = []
    
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
        
        # Valuation warnings
        if info.get('trailingPE', 0) > 30 and info.get('trailingPE', 0) > info.get('industryPE', 100):
            warnings.append(f"⚠️ {ticker}: High P/E ratio ({info.get('trailingPE', 0):.1f}) compared to industry")
        
        # Financial health warnings
        if info.get('debtToEquity', 0) > 1.5:
            warnings.append(f"⚠️ {ticker}: High debt-to-equity ratio ({info.get('debtToEquity', 0):.2f})")
        
        # Technical analysis signals (only if RSI exists)
        if 'RSI' in hist.columns and not pd.isna(hist['RSI'].iloc[-1]):
            last_rsi = hist['RSI'].iloc[-1]
            if last_rsi > 70:
                warnings.append(f"⚠️ {ticker}: Overbought (RSI = {last_rsi:.1f}) - Consider profit booking")
            elif last_rsi < 30:
                insights.append(f"🟢 {ticker}: Oversold (RSI = {last_rsi:.1f}) - Potential buying opportunity")
        
        # Exit signal based on moving averages
        if 'MA_50' in hist.columns and 'MA_200' in hist.columns:
            if len(hist) >= 2:
                if hist['MA_50'].iloc[-1] < hist['MA_200'].iloc[-1] and hist['MA_50'].iloc[-2] >= hist['MA_200'].iloc[-2]:
                    suggestions.append(f"🔴 Consider exiting {ticker} - Death Cross detected (50MA crossed below 200MA)")
    
    return {
        'insights': insights,
        'warnings': warnings,
        'suggestions': suggestions
    }

def detect_anomalies(portfolio):
    """Use Isolation Forest to detect anomalous behavior in stocks"""
    features = []
    tickers = []
    
    for ticker, holding in portfolio.items():
        if holding['data'] is None:
            continue
            
        info = holding['data']['info']
        hist = holding['data']['history']
        
        features.append([
            info.get('beta', 0),
            info.get('trailingPE', 0),
            info.get('debtToEquity', 0),
            info.get('returnOnEquity', 0),
            info.get('currentRatio', 0),
            info.get('quickRatio', 0),
            info.get('priceToBook', 0),
            hist['Close'].pct_change().std() * np.sqrt(252) if not hist.empty else 0
        ])
        tickers.append(ticker)
    
    if not features or len(features) < 2:
        return {}
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    # Train Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    preds = clf.predict(X)
    
    # Return anomalous stocks
    anomalies = {}
    for i, pred in enumerate(preds):
        if pred == -1:
            anomalies[tickers[i]] = "Anomalous behavior detected across multiple metrics"
    
    return anomalies

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
                        current_price = stock_data['info'].get('currentPrice', 
                                             stock_data['info'].get('regularMarketPrice', 0))
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
            total_value = sum([h['value'] for h in st.session_state.portfolio.values() 
                              if h['data'] is not None])
            st.metric("Total Value", f"${total_value:,.2f}" if total_value > 0 else "$0")
            
            for ticker, holding in st.session_state.portfolio.items():
                if holding['data'] is not None:
                    st.markdown(f"**{ticker}**: {holding['quantity']} shares (${holding['value']:,.2f})")

    # Main content area
    st.title("AI Portfolio Manager Pro")
    st.caption("Professional-grade portfolio analysis powered by AI")
    
    if not st.session_state.portfolio:
        st.info("💡 Add stocks to your portfolio using the sidebar to begin analysis")
        return
    
    # Calculate portfolio metrics
    with st.spinner("Analyzing your portfolio..."):
        portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio)
        ai_output = generate_ai_insights(st.session_state.portfolio, portfolio_metrics)
        anomalies = detect_anomalies(st.session_state.portfolio)
    
    # Dashboard Overview
    st.subheader("📊 Portfolio Overview", divider="blue")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${portfolio_metrics['total_value']:,.2f}")
    with col2:
        beta_color = "red" if portfolio_metrics['total_beta'] > 1.2 else "green" if portfolio_metrics['total_beta'] < 0.8 else "orange"
        st.metric("Portfolio Beta", f"{portfolio_metrics['total_beta']:.2f}", 
                 help="Measures sensitivity to market movements")
    with col3:
        st.metric("Avg P/E Ratio", f"{portfolio_metrics['total_pe']:.1f}")
    with col4:
        st.metric("Avg ROE", f"{portfolio_metrics['total_roe']:.2%}")
    
    # Risk Analysis Section
    st.subheader("⚠️ Risk Analysis", divider="blue")
    
    if ai_output['warnings'] or anomalies:
        tab1, tab2 = st.tabs(["Risk Warnings", "Anomaly Detection"])
        
        with tab1:
            for warning in ai_output['warnings']:
                st.warning(warning)
        
        with tab2:
            if anomalies:
                for ticker, message in anomalies.items():
                    st.error(f"**{ticker}**: {message}")
            else:
                st.success("No anomalies detected in your portfolio")
    else:
        st.success("No significant risk factors detected in your portfolio")
    
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
    
    # Correlation Matrix
    st.subheader("🔗 Correlation Matrix", divider="blue")
    if portfolio_metrics['correlation_matrix'] is not None:
        fig = go.Figure(go.Heatmap(
            z=portfolio_metrics['correlation_matrix'],
            x=portfolio_metrics['correlation_matrix'].columns,
            y=portfolio_metrics['correlation_matrix'].columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual Stock Analysis
    st.subheader("📈 Stock Analysis", divider="blue")
    
    selected_ticker = st.selectbox(
        "Select stock for detailed analysis",
        [t for t in st.session_state.portfolio.keys() 
         if st.session_state.portfolio[t]['data'] is not None],
        format_func=lambda x: f"{x} - {st.session_state.portfolio[x]['data']['info'].get('shortName', '')}"
    )
    
    if selected_ticker:
        holding = st.session_state.portfolio[selected_ticker]
        info = holding['data']['info']
        hist = holding['data']['history']
        
        # Stock header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {selected_ticker}")
            st.markdown(f"**{info.get('shortName', '')}**")
            st.caption(f"{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}")
        
        with col2:
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            st.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A")
        
        with col3:
            day_change = info.get('regularMarketChangePercent', 0)
            st.metric("Daily Change", f"{day_change:.2f}%", 
                     delta_color="inverse" if day_change else "off")
        
        # Valuation metrics
        st.markdown("#### Valuation Metrics")
        cols = st.columns(4)
        metrics = [
            ('P/E Ratio', 'trailingPE', None),
            ('Price/Book', 'priceToBook', None),
            ('Price/Sales', 'priceToSalesTrailing12Months', None),
            ('Dividend Yield', 'dividendYield', '{:.2%}'),
        ]
        
        for (col, (name, key, fmt)) in zip(cols, metrics):
            with col:
                value = info.get(key, 0)
                display = fmt.format(value) if fmt else f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                st.metric(name, display)
        
        # Financial health metrics
        st.markdown("#### Financial Health")
        cols = st.columns(4)
        metrics = [
            ('Debt/Equity', 'debtToEquity', None),
            ('Current Ratio', 'currentRatio', None),
            ('ROE', 'returnOnEquity', '{:.2%}'),
            ('Profit Margin', 'profitMargins', '{:.2%}'),
        ]
        
        for (col, (name, key, fmt)) in zip(cols, metrics):
            with col:
                value = info.get(key, 0)
                display = fmt.format(value) if fmt else f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                st.metric(name, display)
        
        # Price chart with technical indicators
        st.markdown("#### Price Analysis")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'], 
            name='Price', line=dict(color='#4f8bf9')), row=1, col=1)
        
        if 'MA_50' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['MA_50'], 
                name='50-Day MA', line=dict(color='#ff7f0e')), row=1, col=1)
        
        if 'MA_200' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['MA_200'], 
                name='200-Day MA', line=dict(color='#2ca02c')), row=1, col=1)
        
        # RSI if available
        if 'RSI' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['RSI'], 
                name='RSI', line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=600,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
