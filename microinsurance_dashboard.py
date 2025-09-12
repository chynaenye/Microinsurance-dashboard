
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Microinsurance Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .critical-insight {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Your SHAP Data (from your actual results)
@st.cache_data
def load_shap_insights():
    return {
        'top_features': [
            {'feature': 'Months_Since_Claim', 'importance': 1.1717, 'rank': 1},
            {'feature': 'Total_Claims', 'importance': 0.3125, 'rank': 2},
            {'feature': 'Age', 'importance': 0.2001, 'rank': 3},
            {'feature': 'Policy_Start_Date_ts', 'importance': 0.1956, 'rank': 4},
            {'feature': 'Clinic_Visits', 'importance': 0.1898, 'rank': 5},
            {'feature': 'Clinic_Access_Proxy', 'importance': 0.1593, 'rank': 6},
            {'feature': 'Month_month', 'importance': 0.0943, 'rank': 7},
            {'feature': 'Month_ts', 'importance': 0.0751, 'rank': 8},
            {'feature': 'Claim_Denial_Rate', 'importance': 0.0721, 'rank': 9},
            {'feature': 'Avg_Monthly_Balance_NGN', 'importance': 0.0679, 'rank': 10}
        ],
        'business_impact': {
            'annual_dropouts': 6909,
            'dropout_rate': 63.8,
            'annual_cost': 34.5,
            'potential_savings': 14.8,
            'roi_3year': 586
        }
    }

# Regional data (from your original analysis)
@st.cache_data  
def load_regional_data():
    return {
        'region': ['Lagos', 'Enugu', 'Kaduna', 'Kano', 'Abuja', 'Jos', 'Ibadan', 'Port Harcourt'],
        'dropout_rate': [66.2, 65.3, 64.8, 64.1, 63.7, 62.7, 62.5, 61.2],
        'beneficiaries': [1429, 1215, 1108, 1087, 1156, 987, 1203, 1644],
        'risk_level': ['🔴 HIGH', '🔴 HIGH', '🔴 HIGH', '🟡 MEDIUM-HIGH', '🟡 MEDIUM', '🟡 MEDIUM', '🟡 MEDIUM', '🟢 LOWEST']
    }

# Navigation Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose Page:",
    ["🏠 Overview", "🎯 AI Insights (SHAP)", "🌍 Regional Analysis", "📈 Business Impact", "⚡ Quick Actions"]
)

# MAIN PAGES
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">Microinsurance Dropout Risk Dashboard</h1>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Current Dropout Rate",
            value="63.8%",
            delta="-5.0% potential improvement"
        )
    
    with col2:
        st.metric(
            label="💰 Annual Cost",
            value="₦34.5M",
            delta="₦14.8M recoverable"
        )
    
    with col3:
        st.metric(
            label="🎯 Model Accuracy",
            value="95.6%",
            delta="Can catch 96% of dropouts"
        )
    
    with col4:
        st.metric(
            label="🏆 3-Year ROI",
            value="586%",
            delta="4.3 month payback"
        )
    
    st.markdown("---")
    
    # Project Summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Project Overview")
        st.markdown("""
        **Objective**: Predict microinsurance dropout risk in Sub-Saharan Africa
        
        **Dataset**: 10,829 beneficiaries across 4 integrated datasets:
        - Policy information
        - Mobile money transactions  
        - Weather data
        - Clinic access metrics
        
        **Key Achievement**: AI model identifies 95.6% of dropouts 1-6 months in advance
        
        **Business Impact**: ₦14.8M annual savings potential through targeted interventions
        """)
    
    with col2:
        st.subheader("🚨 Critical Discovery")
        st.error("""
        **"Months_Since_Claim"** is 3.7x more predictive of dropout than any other factor!
        
        This means beneficiaries who haven't claimed recently are at extreme risk.
        """)
        
        st.success("""
        **Immediate Action Required**:
        Target 2,500+ beneficiaries with 90+ days since last claim
        """)

elif page == "🎯 AI Insights (SHAP)":
    st.title("🎯 AI-Discovered Dropout Predictors")
    st.markdown("### Machine Learning Analysis of 10,829 Beneficiaries")
    
    shap_data = load_shap_insights()
    
    # Create DataFrame for visualization
    features_df = pd.DataFrame(shap_data['top_features'])
    
    # Feature Importance Chart
    fig = px.bar(
        features_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Dropout Risk Predictors - AI Analysis',
        labels={'importance': 'Predictive Impact Score', 'feature': 'Risk Factors'},
        color='importance',
        color_continuous_scale='Reds',
        height=600
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        title_font_size=16,
        font_size=12
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Critical Insight Box
    st.markdown('<div class="critical-insight">', unsafe_allow_html=True)
    st.markdown("""
    ## 🚨 CRITICAL BUSINESS INSIGHT
    
    **"Months_Since_Claim"** has an impact score of **1.17** - this is **3.7x higher** than the second most important factor (Total_Claims: 0.31).
    
    **Translation**: Beneficiaries who haven't filed claims recently are at extreme dropout risk.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Business Actions
    st.subheader("💡 Recommended Business Actions")
    
    tab1, tab2, tab3 = st.tabs(["🚨 Priority #1", "📊 Priority #2", "👥 Priority #3"])
    
    with tab1:
        st.markdown("""
        ### Months Since Last Claim (Impact: 1.17)
        
        **Why Critical**: 3.7x more predictive than any other factor
        
        **Immediate Actions**:
        - 🚨 Emergency outreach to beneficiaries with 90+ days since claim
        - 📱 Automated SMS after 60 days: "We miss you! Free health check available"  
        - 🏥 Mobile health screening campaigns for inactive beneficiaries
        - 👨‍⚕️ Deploy field agents for 90+ day cases
        
        **Expected Impact**: Prevent 1,200+ dropouts = ₦6M savings annually
        """)
    
    with tab2:
        st.markdown("""
        ### Total Claims (Impact: 0.31)
        
        **Why Important**: Heavy users surprisingly at dropout risk
        
        **Actions**:
        - 🌟 VIP support hotline for frequent claimants
        - 👤 Dedicated case managers for 5+ claims/year  
        - ⚡ Fast-track claim processing (24-hour resolution)
        - 📞 Personal check-ins with high-value customers
        
        **Expected Impact**: Prevent 400+ dropouts = ₦2M savings annually
        """)
    
    with tab3:
        st.markdown("""
        ### Age Demographics (Impact: 0.20)
        
        **Why Relevant**: Different age groups need different approaches
        
        **Actions**:
        - 👨‍💼 Youth programs (18-30): Social media engagement
        - 👨‍👩‍👧‍👦 Family packages (31-50): Workplace wellness  
        - 👴👵 Senior care (50+): Home visits, simplified forms
        - 📱 Age-appropriate communication channels
        
        **Expected Impact**: Prevent 380+ dropouts = ₦1.9M savings annually
        """)

elif page == "🌍 Regional Analysis":
    st.title("🌍 Regional Dropout Analysis")
    st.markdown("### 8 Regions Across Nigeria")
    
    regional_data = load_regional_data()
    regions_df = pd.DataFrame(regional_data)
    
    # Regional Chart
    fig = px.bar(
        regions_df,
        x='region',
        y='dropout_rate',
        title='Dropout Rates by Region',
        labels={'dropout_rate': 'Dropout Rate (%)', 'region': 'Region'},
        color='dropout_rate',
        color_continuous_scale='RdYlBu_r',
        height=500
    )
    
    fig.add_hline(y=63.8, line_dash="dash", line_color="red", 
                  annotation_text="National Average: 63.8%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional Strategy Table
    st.subheader("📋 Regional Strategy Matrix")
    
    strategy_df = regions_df.copy()
    strategy_df['Annual_Cost'] = (strategy_df['beneficiaries'] * strategy_df['dropout_rate'] / 100 * 5000).round(0).astype(int)
    strategy_df['Priority'] = ['URGENT', 'URGENT', 'HIGH', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'STUDY']
    
    st.dataframe(
        strategy_df[['region', 'dropout_rate', 'beneficiaries', 'risk_level', 'Priority', 'Annual_Cost']],
        column_config={
            'region': 'Region',
            'dropout_rate': st.column_config.NumberColumn('Dropout Rate (%)', format="%.1f%%"),
            'beneficiaries': st.column_config.NumberColumn('Beneficiaries', format="%d"),
            'risk_level': 'Risk Level',
            'Priority': 'Action Priority',
            'Annual_Cost': st.column_config.NumberColumn('Annual Cost (₦)', format="₦%d")
        }
    )
    
    # Key Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🔴 URGENT INTERVENTION**
        - Lagos: 66.2% dropout (1,429 beneficiaries)
        - Enugu: 65.3% dropout (1,215 beneficiaries)
        - Combined: 2,644 high-risk beneficiaries
        """)
    
    with col2:
        st.success("""
        **🟢 BEST PRACTICE STUDY**
        - Port Harcourt: 61.2% dropout (best performance)
        - 5% better than Lagos
        - Study success factors for replication
        """)

elif page == "📈 Business Impact":
    st.title("📈 Business Impact & ROI Analysis")
    
    impact_data = load_shap_insights()['business_impact']
    
    # Financial Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Annual Cost", f"₦{impact_data['annual_cost']}M", "Customer churn loss")
        st.metric("Wasted Subsidies", "₦17.3M", "50% government funded")
    
    with col2:
        st.metric("Potential Savings", f"₦{impact_data['potential_savings']}M", "With AI intervention")
        st.metric("Implementation Cost", "₦6.5M", "Year 1 investment")
    
    with col3:
        st.metric("3-Year ROI", f"{impact_data['roi_3year']}%", "586% return")
        st.metric("Payback Period", "4.3 months", "Break-even timeline")
    
    # ROI Breakdown Chart
    st.subheader("💰 3-Year Financial Projection")
    
    years = ['Year 1', 'Year 2', 'Year 3']
    investment = [6.5, 2.0, 2.0]
    savings = [14.8, 14.8, 14.8]
    net_benefit = [8.3, 12.8, 12.8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Investment', x=years, y=investment, marker_color='red'))
    fig.add_trace(go.Bar(name='Savings', x=years, y=savings, marker_color='green'))
    fig.add_trace(go.Bar(name='Net Benefit', x=years, y=net_benefit, marker_color='blue'))
    
    fig.update_layout(
        title='Financial Impact Over 3 Years (₦ Millions)',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource Allocation
    st.subheader("🎯 Recommended Resource Allocation")
    
    allocation_data = {
        'Intervention': ['Claim Inactivity (90+ days)', 'High Claimants Support', 'Age-Based Programs', 'Regional Variations'],
        'SHAP_Priority': ['Priority #1', 'Priority #2', 'Priority #3', 'Supporting'],
        'Budget_Percent': [60, 20, 15, 5],
        'Annual_Budget': [3.9, 1.3, 1.0, 0.3],
        'Expected_Impact': ['1,200 dropouts prevented', '400 dropouts prevented', '300 dropouts prevented', '82 dropouts prevented']
    }
    
    allocation_df = pd.DataFrame(allocation_data)
    
    st.dataframe(
        allocation_df,
        column_config={
            'Budget_Percent': st.column_config.NumberColumn('Budget %', format="%d%%"),
            'Annual_Budget': st.column_config.NumberColumn('Annual Budget (₦M)', format="₦%.1fM")
        }
    )

elif page == "⚡ Quick Actions":
    st.title("⚡ Immediate Action Plan")
    st.markdown("### What to Do Right Now")
    
    # Emergency Actions
    st.subheader("🚨 Emergency Actions (Next 7 Days)")
    
    with st.expander("Day 1-2: Identify Critical Cases", expanded=True):
        st.markdown("""
        **Task**: Extract all beneficiaries with 90+ days since last claim
        - Expected count: ~2,500 beneficiaries
        - Priority regions: Lagos, Enugu, Kaduna
        - Create contact list with phone numbers
        """)
        
        if st.button("📊 Download Critical Cases Template"):
            st.success("Template would be downloaded in real implementation")
    
    with st.expander("Day 3-4: SMS Campaign", expanded=False):
        st.markdown("""
        **SMS Template**: "We miss you! Your health matters. Reply YES for a free health check. - [Company]"
        
        **Targets**:
        - All 90+ day inactive beneficiaries
        - Send between 9 AM - 5 PM
        - Track response rates by region
        """)
    
    with st.expander("Day 5-7: Field Agent Calls", expanded=False):
        st.markdown("""
        **Call Script**: 
        1. "Hi [Name], this is [Agent] from [Company]"
        2. "We noticed you haven't used your insurance recently"  
        3. "Is everything okay with your health?"
        4. "Would you like to schedule a free health check?"
        
        **Priority**: Non-responders to SMS campaign
        """)
    
    # 30-Day Plan
    st.subheader("📅 30-Day Implementation Plan")
    
    timeline_data = {
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'Focus': ['Emergency Outreach', 'System Setup', 'Process Optimization', 'Results Analysis'],
        'Key_Activities': [
            'Contact 2,500 critical cases',
            'Deploy automated SMS system',  
            'Train field agents on SHAP insights',
            'Measure intervention success rate'
        ],
        'Success_Metric': [
            '20% response rate to outreach',
            'SMS system processing 1000/day',
            'Agents trained on top 5 factors', 
            '15% reduction in critical cases'
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)
    
    # Budget Request
    st.subheader("💰 Budget Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Emergency Budget Needed**: ₦1.0M
        
        **Breakdown**:
        - SMS campaigns: ₦200K
        - Field agent overtime: ₦500K  
        - Mobile health screenings: ₦300K
        """)
    
    with col2:
        st.markdown("""
        **Expected ROI**: 
        - Prevent 200 dropouts in 30 days
        - Save ₦1.0M in acquisition costs
        - 100% ROI in first month
        """)
    
    # Call to Action
    st.error("""
    ## 🎯 DECISION REQUIRED
    
    **The Data Shows**: 2,500+ beneficiaries are at critical dropout risk RIGHT NOW
    
    **The Cost**: ₦1M emergency intervention vs ₦12.5M if they all dropout
    
    **The Ask**: Approve ₦1M emergency budget to start interventions immediately
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    📊 Microinsurance Dropout Risk Dashboard | Built with Streamlit | Data Science Team
</div>
""", unsafe_allow_html=True)
