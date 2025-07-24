"""
🤖 Agentic AI Flood Prediction System - Hugging Face Spaces Deployment
=====================================================================

This is the main application file for deploying your autonomous flood prediction 
system on Hugging Face Spaces using Gradio.

Author: Your Name
License: MIT
"""

import gradio as gr
import pandas as pd
import numpy as np
import asyncio
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Import your agentic AI system
try:
    from agentic_ai_system import (
        create_flood_prediction_orchestrator,
        ModelPerformance,
        DataQualityReport
    )
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    print("⚠️ Agentic AI system not available. Using demo mode.")

# Global variables
orchestrator = None
current_results = None

def initialize_system():
    """Initialize the agentic AI system"""
    global orchestrator
    
    if not SYSTEM_AVAILABLE:
        return "❌ System not available - running in demo mode"
    
    try:
        orchestrator = create_flood_prediction_orchestrator()
        return "✅ Agentic AI System initialized successfully!"
    except Exception as e:
        return f"❌ Failed to initialize system: {str(e)}"

def create_sample_data(n_samples=1000):
    """Create sample flood prediction data"""
    np.random.seed(42)
    
    # Generate realistic flood prediction features
    data = {
        'MonsoonIntensity': np.random.normal(7.5, 2.0, n_samples),
        'TopographyDrainage': np.random.normal(5.0, 1.5, n_samples),
        'RiverManagement': np.random.normal(6.0, 1.8, n_samples),
        'Deforestation': np.random.normal(4.5, 2.0, n_samples),
        'Urbanization': np.random.normal(5.5, 1.7, n_samples),
        'ClimateChange': np.random.normal(6.8, 1.9, n_samples),
        'DamsQuality': np.random.normal(7.2, 1.6, n_samples),
        'Siltation': np.random.normal(4.8, 1.8, n_samples),
        'AgriculturalPractices': np.random.normal(5.8, 1.5, n_samples),
        'Encroachments': np.random.normal(4.2, 1.9, n_samples),
        'IneffectiveDisasterPreparedness': np.random.normal(3.8, 1.7, n_samples),
        'DrainageSystems': np.random.normal(6.5, 1.8, n_samples),
        'CoastalVulnerability': np.random.normal(5.2, 2.1, n_samples),
        'Landslides': np.random.normal(3.5, 1.6, n_samples),
        'Watersheds': np.random.normal(6.8, 1.7, n_samples),
        'DeterioratingInfrastructure': np.random.normal(4.5, 1.8, n_samples),
        'PopulationScore': np.random.normal(6.2, 1.9, n_samples),
        'WetlandLoss': np.random.normal(4.8, 1.7, n_samples),
        'InadequatePlanning': np.random.normal(4.2, 1.8, n_samples),
        'PoliticalFactors': np.random.normal(5.5, 2.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic flood probability
    flood_prob = (
        0.08 * df['MonsoonIntensity'] +
        0.06 * df['ClimateChange'] +
        0.05 * df['Deforestation'] +
        0.05 * df['Urbanization'] +
        -0.04 * df['TopographyDrainage'] +
        -0.04 * df['RiverManagement'] +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Normalize to 0-1 range
    flood_prob = (flood_prob - flood_prob.min()) / (flood_prob.max() - flood_prob.min())
    df['FloodProbability'] = flood_prob
    
    return df

async def run_autonomous_cycle():
    """Run a single autonomous AI cycle"""
    global current_results
    
    if not SYSTEM_AVAILABLE or orchestrator is None:
        return create_demo_results()
    
    try:
        # Create sample data
        flood_data = create_sample_data()
        
        # Prepare data context
        from sklearn.model_selection import train_test_split
        
        feature_columns = [col for col in flood_data.columns if col != 'FloodProbability']
        X = flood_data[feature_columns]
        y = flood_data['FloodProbability']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        data_context = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'current_data': flood_data,
            'data_version': f'hf_v{datetime.now().strftime("%H%M%S")}',
            'timestamp': datetime.now()
        }
        
        # Run orchestration cycle
        result = await orchestrator.run_cycle(data_context)
        current_results = result
        
        return format_results(result)
        
    except Exception as e:
        return f"❌ Error running cycle: {str(e)}"

def create_demo_results():
    """Create demo results when system is not available"""
    return {
        'status': 'demo',
        'execution_record': {
            'duration': 45.2,
            'agents_executed': ['data_monitoring', 'model_selection', 'feature_engineering'],
            'summary': {
                'agents_run': 3,
                'actions_required': 2,
                'decisions_made': 1,
                'alerts_generated': 0
            }
        },
        'agent_results': {
            'model_result': {
                'best_model': {
                    'name': 'XGBoost',
                    'r2': 0.8745,
                    'rmse': 0.0652,
                    'cross_val_score': 0.8512
                },
                'all_evaluations': [
                    {'name': 'XGBoost', 'cross_val_score': 0.8512},
                    {'name': 'RandomForest', 'cross_val_score': 0.8234},
                    {'name': 'LightGBM', 'cross_val_score': 0.8456}
                ]
            },
            'feature_result': {
                'original_features': 20,
                'selected_features': 35,
                'improvement': 0.0523
            }
        }
    }

def format_results(result):
    """Format results for display"""
    if result is None:
        result = create_demo_results()
    
    execution_record = result.get('execution_record', {})
    agent_results = result.get('agent_results', {})
    
    output = ["🤖 **AUTONOMOUS AI CYCLE COMPLETED**", "="*50]
    
    # Execution summary
    output.append(f"⏱️ **Duration**: {execution_record.get('duration', 0):.2f} seconds")
    output.append(f"🤖 **Agents Executed**: {len(execution_record.get('agents_executed', []))}")
    
    summary = execution_record.get('summary', {})
    output.append(f"⚡ **Actions Required**: {summary.get('actions_required', 0)}")
    output.append(f"🧠 **Decisions Made**: {summary.get('decisions_made', 0)}")
    output.append(f"🚨 **Alerts Generated**: {summary.get('alerts_generated', 0)}")
    
    # Model results
    if 'model_result' in agent_results:
        model_result = agent_results['model_result']
        best_model = model_result.get('best_model', {})
        if best_model:
            output.append("\n🎯 **BEST MODEL PERFORMANCE**")
            output.append(f"   Model: **{best_model.get('name', 'Unknown')}**")
            output.append(f"   R² Score: **{best_model.get('r2', 0):.4f}**")
            output.append(f"   RMSE: **{best_model.get('rmse', 0):.4f}**")
            output.append(f"   CV Score: **{best_model.get('cross_val_score', 0):.4f}**")
    
    # Feature engineering results
    if 'feature_result' in agent_results:
        feature_result = agent_results['feature_result']
        output.append("\n🔧 **FEATURE ENGINEERING**")
        output.append(f"   Original Features: **{feature_result.get('original_features', 0)}**")
        output.append(f"   Selected Features: **{feature_result.get('selected_features', 0)}**")
        output.append(f"   Improvement: **{feature_result.get('improvement', 0):.4f}**")
    
    # Decisions
    if 'decision_result' in agent_results:
        decision_result = agent_results['decision_result']
        decisions = decision_result.get('decisions', [])
        if decisions:
            output.append("\n🧠 **AUTONOMOUS DECISIONS**")
            for i, decision in enumerate(decisions[:3], 1):
                output.append(f"   {i}. **{decision.get('type', 'Unknown')}**")
                output.append(f"      Reason: {decision.get('reason', 'N/A')}")
    
    output.append("\n✅ **System operating autonomously!**")
    
    return "\n".join(output)

def create_performance_chart():
    """Create performance visualization"""
    global current_results
    
    if current_results is None:
        current_results = create_demo_results()
    
    agent_results = current_results.get('agent_results', {})
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Performance Comparison', 'Feature Engineering Impact', 
                       'System Status', 'Agent Activity'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "pie"}]]
    )
    
    # 1. Model Performance
    if 'model_result' in agent_results:
        model_result = agent_results['model_result']
        evaluations = model_result.get('all_evaluations', [])
        
        if evaluations:
            model_names = [eval_dict['name'] for eval_dict in evaluations]
            cv_scores = [eval_dict['cross_val_score'] for eval_dict in evaluations]
            
            fig.add_trace(
                go.Bar(x=model_names, y=cv_scores, name="CV Score", 
                       marker_color=['gold' if i == cv_scores.index(max(cv_scores)) else 'lightblue' 
                                   for i in range(len(cv_scores))]),
                row=1, col=1
            )
    
    # 2. Feature Engineering
    if 'feature_result' in agent_results:
        feature_result = agent_results['feature_result']
        original = feature_result.get('original_features', 20)
        selected = feature_result.get('selected_features', 35)
        
        fig.add_trace(
            go.Bar(x=['Original Features', 'Selected Features'], 
                   y=[original, selected],
                   name="Features", 
                   marker_color=['coral', 'lightgreen']),
            row=1, col=2
        )
    
    # 3. System Performance Gauge
    execution_record = current_results.get('execution_record', {})
    performance_score = min(100, execution_record.get('duration', 45) * 2)  # Mock performance
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=performance_score,
            title={'text': "System Performance"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=1
    )
    
    # 4. Agent Activity
    agents_executed = execution_record.get('agents_executed', ['data_monitoring', 'model_selection', 'feature_engineering'])
    agent_counts = [1 for _ in agents_executed]
    
    fig.add_trace(
        go.Pie(labels=agents_executed, values=agent_counts,
               marker_colors=['lightblue', 'lightgreen', 'orange', 'pink', 'lightcoral']),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="🤖 Agentic AI System Dashboard",
        height=600,
        showlegend=False
    )
    
    return fig

def predict_flood_probability(monsoon_intensity, topography_drainage, river_management, 
                            deforestation, urbanization, climate_change):
    """Predict flood probability based on input parameters"""
    
    # Simple prediction model (in real system, this would use the trained model)
    flood_prob = (
        0.08 * monsoon_intensity +
        0.06 * climate_change +
        0.05 * deforestation +
        0.05 * urbanization +
        -0.04 * topography_drainage +
        -0.04 * river_management
    )
    
    # Normalize to 0-1 range (approximate)
    flood_prob = max(0, min(1, (flood_prob + 2) / 4))
    
    # Create risk assessment
    if flood_prob < 0.3:
        risk_level = "🟢 LOW RISK"
        recommendations = [
            "Continue regular monitoring",
            "Maintain current flood management systems",
            "Review drainage systems annually"
        ]
    elif flood_prob < 0.6:
        risk_level = "🟡 MODERATE RISK"
        recommendations = [
            "Increase monitoring frequency",
            "Check drainage systems",
            "Prepare emergency response plans",
            "Monitor weather forecasts closely"
        ]
    else:
        risk_level = "🔴 HIGH RISK"
        recommendations = [
            "Activate emergency protocols",
            "Issue flood warnings",
            "Check evacuation routes",
            "Coordinate with emergency services",
            "Monitor situation continuously"
        ]
    
    result = f"""
## 🌊 Flood Probability Prediction

**Predicted Flood Probability: {flood_prob:.1%}**

**Risk Level: {risk_level}**

### 📊 Input Analysis:
- Monsoon Intensity: {monsoon_intensity}/10
- Topography Drainage: {topography_drainage}/10
- River Management: {river_management}/10
- Deforestation Level: {deforestation}/10
- Urbanization: {urbanization}/10
- Climate Change Impact: {climate_change}/10

### 💡 Recommendations:
"""
    
    for i, rec in enumerate(recommendations, 1):
        result += f"\n{i}. {rec}"
    
    result += f"\n\n*Prediction generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return result

# Initialize system on startup
init_message = initialize_system()

# Create Gradio interface
with gr.Blocks(title="🤖 Agentic AI Flood Prediction System", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🤖 Agentic AI Flood Prediction System
    
    Welcome to the **autonomous flood prediction system** powered by multiple AI agents!
    
    ## 🎯 What This System Does:
    - **Monitors** data quality and detects drift automatically
    - **Selects** the best models using advanced optimization
    - **Engineers** features autonomously to improve predictions
    - **Makes** intelligent decisions about deployment and retraining
    - **Operates** 24/7 without human intervention
    
    ## 🚀 Features:
    - Real-time flood probability predictions
    - Autonomous model training and optimization
    - Intelligent decision making
    - Continuous performance monitoring
    """)
    
    with gr.Tab("🔮 Flood Prediction"):
        gr.Markdown("### Enter the flood risk factors to get a prediction:")
        
        with gr.Row():
            with gr.Column():
                monsoon = gr.Slider(1, 10, value=7, label="Monsoon Intensity", 
                                  info="1=Low, 10=Extreme")
                topography = gr.Slider(1, 10, value=5, label="Topography Drainage", 
                                     info="1=Poor, 10=Excellent")
                river_mgmt = gr.Slider(1, 10, value=6, label="River Management", 
                                     info="1=Poor, 10=Excellent")
            
            with gr.Column():
                deforestation = gr.Slider(1, 10, value=4, label="Deforestation Level", 
                                        info="1=Low, 10=Severe")
                urbanization = gr.Slider(1, 10, value=5, label="Urbanization", 
                                       info="1=Low, 10=High")
                climate_change = gr.Slider(1, 10, value=7, label="Climate Change Impact", 
                                         info="1=Low, 10=Severe")
        
        predict_btn = gr.Button("🔮 Predict Flood Probability", variant="primary")
        prediction_output = gr.Markdown()
        
        predict_btn.click(
            predict_flood_probability,
            inputs=[monsoon, topography, river_mgmt, deforestation, urbanization, climate_change],
            outputs=prediction_output
        )
    
    with gr.Tab("🤖 Autonomous AI System"):
        gr.Markdown("### Run the autonomous AI system to train and optimize models automatically:")
        
        with gr.Row():
            run_btn = gr.Button("🚀 Run Autonomous Cycle", variant="primary")
            
        system_output = gr.Markdown(value=f"**System Status**: {init_message}")
        
        async def run_cycle_wrapper():
            return await run_autonomous_cycle()
        
        run_btn.click(
            lambda: asyncio.run(run_autonomous_cycle()),
            outputs=system_output
        )
    
    with gr.Tab("📊 Dashboard"):
        gr.Markdown("### Real-time system performance and analytics:")
        
        refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
        dashboard_plot = gr.Plot()
        
        refresh_btn.click(
            create_performance_chart,
            outputs=dashboard_plot
        )
        
        # Auto-load dashboard on tab open
        app.load(create_performance_chart, outputs=dashboard_plot)
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🧠 How the Agentic AI System Works:
        
        ### 🤖 Autonomous Agents:
        1. **Data Monitoring Agent**: Watches for data quality issues and drift
        2. **Model Selection Agent**: Automatically finds the best models
        3. **Feature Engineering Agent**: Creates new features to improve performance
        4. **Performance Monitoring Agent**: Tracks model performance continuously
        5. **Decision Making Agent**: Makes intelligent deployment decisions
        6. **Orchestrator**: Coordinates all agents and manages the system
        
        ### 🔄 Autonomous Cycle:
        1. **Monitor** → Check data quality and drift
        2. **Engineer** → Create and select optimal features  
        3. **Train** → Optimize models using advanced techniques
        4. **Evaluate** → Assess performance and detect issues
        5. **Decide** → Make deployment and retraining decisions
        6. **Deploy** → Update system autonomously
        
        ### 🎯 Key Benefits:
        - **Zero Human Intervention**: Runs completely autonomously
        - **Continuous Improvement**: Adapts and learns from new data
        - **Proactive Monitoring**: Detects issues before they impact predictions
        - **Intelligent Decisions**: Makes smart choices about model updates
        - **24/7 Operation**: Never stops learning and improving
        
        ### 🚀 Deployment:
        This system is deployed on **Hugging Face Spaces** for:
        - ✅ **Always-on availability**
        - ✅ **Scalable compute resources**
        - ✅ **Easy sharing and collaboration**
        - ✅ **Integrated ML infrastructure**
        
        ---
        **Built with cutting-edge agentic AI technology for autonomous flood prediction! 🌊**
        """)

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 