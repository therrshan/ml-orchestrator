#!/usr/bin/env python3
"""
Generate HTML report for MNIST classification results.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def load_results(results_dir):
    """Load evaluation results."""
    results_dir = Path(results_dir)
    
    # Load evaluation results
    with open(results_dir / 'evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Load predictions if available
    predictions_file = results_dir / 'predictions.npz'
    predictions_data = None
    if predictions_file.exists():
        predictions_data = np.load(predictions_file)
    
    return results, predictions_data


def generate_confusion_matrix_html(conf_matrix):
    """Generate HTML for confusion matrix visualization."""
    conf_matrix = np.array(conf_matrix)
    
    html = '<div class="confusion-matrix">\n'
    html += '<table class="cm-table">\n'
    
    # Header row
    html += '<tr><th></th>'
    for i in range(10):
        html += f'<th>Pred {i}</th>'
    html += '</tr>\n'
    
    # Data rows
    for i in range(10):
        html += f'<tr><th>True {i}</th>'
        for j in range(10):
            value = conf_matrix[i, j]
            # Color intensity based on value
            max_val = conf_matrix.max()
            intensity = value / max_val if max_val > 0 else 0
            
            if i == j:  # Diagonal (correct predictions)
                color = f'background-color: rgba(40, 167, 69, {intensity * 0.8 + 0.2});'
            else:  # Off-diagonal (errors)
                color = f'background-color: rgba(220, 53, 69, {intensity * 0.8 + 0.1});'
            
            html += f'<td style="{color}">{value}</td>'
        html += '</tr>\n'
    
    html += '</table>\n'
    html += '</div>\n'
    
    return html


def generate_class_performance_html(per_class_acc, class_report):
    """Generate HTML for per-class performance."""
    html = '<div class="class-performance">\n'
    
    for i in range(10):
        acc = per_class_acc[i]
        class_data = class_report[str(i)]
        
        # Progress bar for accuracy
        bar_width = acc * 100
        color = '#28a745' if acc > 0.9 else '#ffc107' if acc > 0.8 else '#dc3545'
        
        html += f'''
        <div class="class-item">
            <div class="class-header">
                <span class="digit-label">Digit {i}</span>
                <span class="accuracy-value">{acc:.3f}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {bar_width}%; background-color: {color};"></div>
            </div>
            <div class="class-metrics">
                <span>Precision: {class_data['precision']:.3f}</span>
                <span>Recall: {class_data['recall']:.3f}</span>
                <span>F1: {class_data['f1-score']:.3f}</span>
                <span>Support: {class_data['support']}</span>
            </div>
        </div>
        '''
    
    html += '</div>\n'
    return html


def generate_training_history_html(model_info):
    """Generate HTML for training history if available."""
    if 'training_history' not in model_info:
        return '<p>Training history not available.</p>'
    
    history = model_info['training_history']
    epochs = len(history['train_acc'])
    
    html = '<div class="training-history">\n'
    html += '<h3>Training Progress</h3>\n'
    
    # Simple text-based progress chart
    html += '<div class="progress-chart">\n'
    for epoch in range(epochs):
        train_acc = history['train_acc'][epoch]
        val_acc = history['val_acc'][epoch]
        
        html += f'''
        <div class="epoch-row">
            <span class="epoch-label">Epoch {epoch + 1}</span>
            <div class="acc-bars">
                <div class="train-bar">
                    <div class="bar-fill" style="width: {train_acc}%; background: #007bff;"></div>
                    <span class="bar-label">Train: {train_acc:.1f}%</span>
                </div>
                <div class="val-bar">
                    <div class="bar-fill" style="width: {val_acc}%; background: #28a745;"></div>
                    <span class="bar-label">Val: {val_acc:.1f}%</span>
                </div>
            </div>
        </div>
        '''
    
    html += '</div>\n'
    html += '</div>\n'
    
    return html


def generate_html_report(results, output_file):
    """Generate complete HTML report."""
    
    model_info = results.get('model_info', {})
    class_report = results['classification_report']
    conf_matrix = results['confusion_matrix']
    per_class_acc = results['per_class_accuracy']
    
    # Generate HTML
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Classification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .confusion-matrix {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        .cm-table {{
            border-collapse: collapse;
            margin: 0 auto;
            background: white;
        }}
        .cm-table th, .cm-table td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: center;
            min-width: 40px;
        }}
        .cm-table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .class-performance {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .class-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        .class-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .digit-label {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .accuracy-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .progress-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .class-metrics {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #6c757d;
        }}
        .training-history {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        .progress-chart {{
            margin-top: 15px;
        }}
        .epoch-row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .epoch-label {{
            width: 80px;
            font-size: 0.9em;
            color: #6c757d;
        }}
        .acc-bars {{
            flex: 1;
            margin-left: 15px;
        }}
        .train-bar, .val-bar {{
            position: relative;
            height: 20px;
            background: #e9ecef;
            border-radius: 3px;
            margin-bottom: 5px;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        .bar-label {{
            position: absolute;
            right: 5px;
            top: 2px;
            font-size: 0.8em;
            color: white;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .info-table th, .info-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .info-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 MNIST Classification Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Overall Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{results['test_accuracy']:.3f}</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{class_report['macro avg']['precision']:.3f}</div>
                    <div class="metric-label">Avg Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{class_report['macro avg']['recall']:.3f}</div>
                    <div class="metric-label">Avg Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{class_report['macro avg']['f1-score']:.3f}</div>
                    <div class="metric-label">Avg F1-Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['test_samples']:,}</div>
                    <div class="metric-label">Test Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results['average_confidence']:.3f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔢 Per-Digit Performance</h2>
            {generate_class_performance_html(per_class_acc, class_report)}
        </div>

        <div class="section">
            <h2>🎯 Confusion Matrix</h2>
            <p>Darker colors indicate higher values. Green diagonal shows correct predictions, red off-diagonal shows misclassifications.</p>
            {generate_confusion_matrix_html(conf_matrix)}
        </div>

        <div class="section">
            <h2>🏋️ Training Information</h2>
            {generate_training_history_html(model_info)}
            
            <h3>Model Details</h3>
            <table class="info-table">
                <tr><th>Architecture</th><td>{model_info.get('architecture', 'N/A')}</td></tr>
                <tr><th>Parameters</th><td>{model_info.get('num_parameters', 'N/A'):,}</td></tr>
                <tr><th>Epochs Trained</th><td>{model_info.get('epochs_trained', 'N/A')}</td></tr>
                <tr><th>Batch Size</th><td>{model_info.get('batch_size', 'N/A')}</td></tr>
                <tr><th>Learning Rate</th><td>{model_info.get('learning_rate', 'N/A')}</td></tr>
                <tr><th>Device</th><td>{model_info.get('device', 'N/A')}</td></tr>
                <tr><th>Final Train Acc</th><td>{model_info.get('final_train_acc', 0):.2f}%</td></tr>
                <tr><th>Final Val Acc</th><td>{model_info.get('final_val_acc', 0):.2f}%</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>💡 Analysis</h2>
            <div class="highlight">
'''
    
    # Add analysis
    test_acc = results['test_accuracy']
    best_digit = np.argmax(per_class_acc)
    worst_digit = np.argmin(per_class_acc)
    
    if test_acc > 0.95:
        analysis = f"🎉 <strong>Excellent Performance!</strong> The model achieved {test_acc:.1%} accuracy on MNIST test set."
    elif test_acc > 0.90:
        analysis = f"✅ <strong>Good Performance!</strong> The model achieved {test_acc:.1%} accuracy, which is solid for MNIST."
    else:
        analysis = f"⚠️ <strong>Room for Improvement.</strong> The model achieved {test_acc:.1%} accuracy. Consider training longer or adjusting hyperparameters."
    
    html += f'''
                <p>{analysis}</p>
                <ul>
                    <li><strong>Best recognized digit:</strong> {best_digit} ({per_class_acc[best_digit]:.1%} accuracy)</li>
                    <li><strong>Most challenging digit:</strong> {worst_digit} ({per_class_acc[worst_digit]:.1%} accuracy)</li>
                    <li><strong>Model confidence:</strong> Average confidence of {results['average_confidence']:.1%}</li>
                    <li><strong>Training efficiency:</strong> Achieved {model_info.get('final_val_acc', 0):.1f}% validation accuracy in {model_info.get('epochs_trained', 'N/A')} epochs</li>
                </ul>
            </div>
        </div>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; text-align: center; color: #6c757d; font-size: 0.9em;">
            <p>Report generated by ML Pipeline Orchestrator - MNIST Classification Pipeline</p>
        </div>
    </div>
</body>
</html>
'''
    
    # Save HTML report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Generate MNIST classification report')
    parser.add_argument('--results', required=True, help='Directory with evaluation results')
    parser.add_argument('--output', required=True, help='Output path for HTML report')
    
    args = parser.parse_args()
    
    print("📊 Generating MNIST classification report...")
    
    try:
        # Load results
        results, predictions_data = load_results(args.results)
        
        # Generate report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        generate_html_report(results, output_path)
        
        print(f"🎉 Report generated successfully!")
        print(f"📄 HTML report: {output_path}")
        print(f"🌐 Open in browser: file://{output_path.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())