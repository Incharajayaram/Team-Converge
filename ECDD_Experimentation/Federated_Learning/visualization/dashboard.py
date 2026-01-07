"""
Interactive Dashboard for Federated Drift Monitoring.

Real-time visualization of:
- Client activity and status
- Hub aggregation progress
- Global drift detection
- Privacy budget consumption
- System metrics

Uses Plotly Dash for interactive web interface.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class FederatedMonitoringDashboard:
    """
    Interactive dashboard for federated drift monitoring.
    """
    
    def __init__(self, port: int = 8050):
        """
        Initialize dashboard.
        
        Args:
            port: Port for web server
        """
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.port = port
        
        # Data storage
        self.data = {
            'rounds': [],
            'drift_detected': [],
            'drift_scores': {'ks': [], 'psi': [], 'js': []},
            'client_status': {},
            'hub_status': {},
            'privacy_budget': [],
            'threshold_history': [],
            'anomalous_hubs': []
        }
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Federated Drift Detection Dashboard',
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P('Real-time monitoring of hierarchical federated learning',
                      style={'textAlign': 'center', 'color': '#7f8c8d'}),
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),
            
            # Main content
            html.Div([
                # Row 1: System Overview
                html.Div([
                    # Current Status
                    html.Div([
                        html.H3('System Status', style={'color': '#2c3e50'}),
                        html.Div(id='status-cards')
                    ], className='four columns', style={'padding': '10px'}),
                    
                    # Drift Detection
                    html.Div([
                        html.H3('Drift Detection', style={'color': '#2c3e50'}),
                        dcc.Graph(id='drift-timeline')
                    ], className='eight columns', style={'padding': '10px'}),
                ], className='row'),
                
                # Row 2: Drift Scores
                html.Div([
                    html.Div([
                        html.H3('Drift Scores (Ensemble)', style={'color': '#2c3e50'}),
                        dcc.Graph(id='drift-scores')
                    ], className='twelve columns', style={'padding': '10px'}),
                ], className='row'),
                
                # Row 3: Client & Hub Status
                html.Div([
                    html.Div([
                        html.H3('Client Activity', style={'color': '#2c3e50'}),
                        dcc.Graph(id='client-heatmap')
                    ], className='six columns', style={'padding': '10px'}),
                    
                    html.Div([
                        html.H3('Hub Status', style={'color': '#2c3e50'}),
                        dcc.Graph(id='hub-status')
                    ], className='six columns', style={'padding': '10px'}),
                ], className='row'),
                
                # Row 4: Privacy & Threshold
                html.Div([
                    html.Div([
                        html.H3('Privacy Budget', style={'color': '#2c3e50'}),
                        dcc.Graph(id='privacy-budget')
                    ], className='six columns', style={'padding': '10px'}),
                    
                    html.Div([
                        html.H3('Threshold Evolution', style={'color': '#2c3e50'}),
                        dcc.Graph(id='threshold-history')
                    ], className='six columns', style={'padding': '10px'}),
                ], className='row'),
                
            ], style={'padding': '20px'}),
            
            # Data storage
            dcc.Store(id='dashboard-data'),
            
        ], style={'fontFamily': 'Arial, sans-serif'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            [Output('status-cards', 'children'),
             Output('drift-timeline', 'figure'),
             Output('drift-scores', 'figure'),
             Output('client-heatmap', 'figure'),
             Output('hub-status', 'figure'),
             Output('privacy-budget', 'figure'),
             Output('threshold-history', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components."""
            
            # Status cards
            status_cards = self.create_status_cards()
            
            # Drift timeline
            drift_timeline = self.create_drift_timeline()
            
            # Drift scores
            drift_scores = self.create_drift_scores()
            
            # Client heatmap
            client_heatmap = self.create_client_heatmap()
            
            # Hub status
            hub_status = self.create_hub_status()
            
            # Privacy budget
            privacy_budget = self.create_privacy_budget()
            
            # Threshold history
            threshold_history = self.create_threshold_history()
            
            return (status_cards, drift_timeline, drift_scores,
                   client_heatmap, hub_status, privacy_budget, threshold_history)
    
    def create_status_cards(self):
        """Create status overview cards."""
        current_round = len(self.data['rounds'])
        drift_detected = self.data['drift_detected'][-1] if self.data['drift_detected'] else False
        active_clients = sum(1 for status in self.data['client_status'].values() if status)
        
        cards = html.Div([
            # Round
            html.Div([
                html.H4('Current Round', style={'color': '#7f8c8d'}),
                html.H2(str(current_round), style={'color': '#3498db'})
            ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'}),
            
            # Drift Status
            html.Div([
                html.H4('Drift Status', style={'color': '#7f8c8d'}),
                html.H2('⚠️ DETECTED' if drift_detected else '✓ NORMAL',
                       style={'color': '#e74c3c' if drift_detected else '#2ecc71'})
            ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'}),
            
            # Active Clients
            html.Div([
                html.H4('Active Clients', style={'color': '#7f8c8d'}),
                html.H2(f'{active_clients}/15', style={'color': '#9b59b6'})
            ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
        ])
        
        return cards
    
    def create_drift_timeline(self):
        """Create drift detection timeline."""
        if not self.data['rounds']:
            return go.Figure()
        
        fig = go.Figure()
        
        # Drift detection line
        fig.add_trace(go.Scatter(
            x=self.data['rounds'],
            y=self.data['drift_detected'],
            mode='lines+markers',
            name='Drift Detected',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title='Round',
            yaxis_title='Drift Detected',
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']),
            height=250,
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_drift_scores(self):
        """Create drift scores plot (KS, PSI, JS)."""
        if not self.data['rounds']:
            return go.Figure()
        
        fig = go.Figure()
        
        # KS p-value
        fig.add_trace(go.Scatter(
            x=self.data['rounds'],
            y=self.data['drift_scores']['ks'],
            mode='lines',
            name='KS p-value',
            line=dict(color='#3498db', width=2)
        ))
        
        # PSI
        fig.add_trace(go.Scatter(
            x=self.data['rounds'],
            y=self.data['drift_scores']['psi'],
            mode='lines',
            name='PSI',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # JS divergence
        fig.add_trace(go.Scatter(
            x=self.data['rounds'],
            y=self.data['drift_scores']['js'],
            mode='lines',
            name='JS Divergence',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Thresholds
        fig.add_hline(y=0.01, line_dash='dash', line_color='gray', annotation_text='KS threshold')
        fig.add_hline(y=0.1, line_dash='dash', line_color='gray', annotation_text='PSI/JS threshold')
        
        fig.update_layout(
            xaxis_title='Round',
            yaxis_title='Score',
            yaxis_type='log',
            height=300,
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white',
            legend=dict(x=0.01, y=0.99, bordercolor='gray', borderwidth=1)
        )
        
        return fig
    
    def create_client_heatmap(self):
        """Create client activity heatmap."""
        # Create synthetic heatmap data
        num_clients = 15
        num_rounds = min(len(self.data['rounds']), 50)
        
        if num_rounds == 0:
            return go.Figure()
        
        # Generate activity matrix (1 = active, 0 = inactive)
        activity_matrix = np.random.binomial(1, 0.8, (num_clients, num_rounds))
        
        fig = go.Figure(data=go.Heatmap(
            z=activity_matrix,
            x=list(range(max(0, len(self.data['rounds']) - 50), len(self.data['rounds']))),
            y=[f'Client {i}' for i in range(num_clients)],
            colorscale=[[0, '#ecf0f1'], [1, '#2ecc71']],
            showscale=False
        ))
        
        fig.update_layout(
            xaxis_title='Round',
            height=350,
            margin=dict(l=80, r=20, t=20, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_hub_status(self):
        """Create hub status bar chart."""
        hub_ids = [0, 1, 2]
        sketches_received = [5, 4, 5]  # Mock data
        anomalous = [False, True, False]  # Mock data
        
        colors = ['#e74c3c' if anom else '#2ecc71' for anom in anomalous]
        
        fig = go.Figure(data=[
            go.Bar(
                x=hub_ids,
                y=sketches_received,
                marker_color=colors,
                text=sketches_received,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            xaxis_title='Hub ID',
            yaxis_title='Sketches Received',
            xaxis=dict(tickmode='array', tickvals=hub_ids, ticktext=[f'Hub {i}' for i in hub_ids]),
            height=350,
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_privacy_budget(self):
        """Create privacy budget consumption plot."""
        if not self.data['rounds']:
            # Initial state
            rounds = [0]
            consumed = [0]
            total = [10]
        else:
            rounds = self.data['rounds']
            consumed = self.data['privacy_budget']
            total = [10] * len(rounds)
        
        fig = go.Figure()
        
        # Total budget
        fig.add_trace(go.Scatter(
            x=rounds,
            y=total,
            mode='lines',
            name='Total Budget',
            line=dict(color='#95a5a6', width=2, dash='dash')
        ))
        
        # Consumed budget
        fig.add_trace(go.Scatter(
            x=rounds,
            y=consumed,
            mode='lines',
            name='Consumed',
            line=dict(color='#e74c3c', width=3),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title='Round',
            yaxis_title='Privacy Budget (ε)',
            height=300,
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white',
            legend=dict(x=0.01, y=0.99, bordercolor='gray', borderwidth=1)
        )
        
        return fig
    
    def create_threshold_history(self):
        """Create threshold evolution plot."""
        if not self.data['rounds']:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['rounds'],
            y=self.data['threshold_history'],
            mode='lines+markers',
            name='Threshold',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            xaxis_title='Round',
            yaxis_title='Classification Threshold',
            yaxis=dict(range=[0, 1]),
            height=300,
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white'
        )
        
        return fig
    
    def update_data(self, round_num: int, round_log: dict):
        """
        Update dashboard data with new round information.
        
        Args:
            round_num: Round number
            round_log: Round log dictionary
        """
        self.data['rounds'].append(round_num)
        self.data['drift_detected'].append(int(round_log.get('drift_detected', False)))
        
        # Drift scores (mock for now - should come from actual detector)
        self.data['drift_scores']['ks'].append(np.random.uniform(0.01, 0.5))
        self.data['drift_scores']['psi'].append(np.random.uniform(0.05, 0.3))
        self.data['drift_scores']['js'].append(np.random.uniform(0.05, 0.3))
        
        # Privacy budget (cumulative)
        current_budget = self.data['privacy_budget'][-1] if self.data['privacy_budget'] else 0
        self.data['privacy_budget'].append(current_budget + 0.1)
        
        # Threshold
        self.data['threshold_history'].append(round_log.get('current_threshold', 0.5))
    
    def load_from_results(self, results_file: str):
        """
        Load data from experiment results file.
        
        Args:
            results_file: Path to JSON results file
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if 'round_logs' in results:
            for log in results['round_logs']:
                self.update_data(log['round'], log)
    
    def run(self, debug: bool = True):
        """
        Run dashboard server.
        
        Args:
            debug: Debug mode
        """
        print(f"\n{'='*70}")
        print("Starting Federated Monitoring Dashboard")
        print(f"{'='*70}")
        print(f"Open browser to: http://localhost:{self.port}/")
        print("Press Ctrl+C to stop")
        print(f"{'='*70}\n")
        
        self.app.run_server(debug=debug, port=self.port)


def create_static_dashboard(results_file: str, save_path: str = 'dashboard.html'):
    """
    Create static HTML dashboard from results.
    
    Args:
        results_file: Path to experiment results JSON
        save_path: Path to save HTML file
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Drift Detection Timeline', 'Drift Scores',
                       'Privacy Budget', 'Threshold Evolution',
                       'Detection Metrics', 'Communication Overhead'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "bar"}, {"secondary_y": False}]]
    )
    
    rounds = [log['round'] for log in results['round_logs']]
    drift_detected = [log['drift_detected'] for log in results['round_logs']]
    
    # Row 1: Drift timeline
    fig.add_trace(
        go.Scatter(x=rounds, y=drift_detected, mode='lines+markers',
                  name='Drift', line=dict(color='red')),
        row=1, col=1
    )
    
    # Row 1: Mock drift scores
    fig.add_trace(
        go.Scatter(x=rounds, y=np.random.uniform(0.01, 0.5, len(rounds)),
                  name='KS', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(height=900, showlegend=True,
                     title_text="Federated Drift Detection Dashboard")
    
    # Save
    fig.write_html(save_path)
    print(f"Static dashboard saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = FederatedMonitoringDashboard(port=8050)
    
    # Option 1: Run with mock data
    print("Starting dashboard with mock data...")
    print("In production, connect to live simulation")
    
    # Add some mock data
    for i in range(50):
        mock_log = {
            'round': i,
            'drift_detected': i > 30,  # Drift after round 30
            'current_threshold': 0.5 + 0.01 * (i - 30) if i > 30 else 0.5
        }
        dashboard.update_data(i, mock_log)
    
    # Run dashboard
    dashboard.run(debug=True)
    
    # Option 2: Load from results file
    # dashboard.load_from_results('results/exp1_baseline/run_0.json')
    # dashboard.run()
