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
import requests  # For optional news API

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
        'health_score': 0
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
    
    # Calculate correlation matrix
    if not close_prices.empty:
        metrics['correlation_matrix'] = close_prices.corr()
    
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
