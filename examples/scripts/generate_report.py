#!/usr/bin/env python3
"""
Example script: Generate training report.
This creates a comprehensive HTML report of the training pipeline results.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def load_results(results_dir):
    """
    Load all available results from the results directory.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Dictionary containing all loaded results
    """
    results_dir = Path(results_dir)
    results = {}
    
    # Load evaluation results
    eval_file = results_dir / 'evaluation_results.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            results['evaluation'] = json.load(f)
    
    # Load predictions
    pred_file = results_dir / 'predictions.json'
    if pred_file.exists():
        with open(pred_file, 'r') as f:
            results['predictions'] = json.load(f)
    
    return results


def generate_html_report(results, output_file):
    """
    Generate HTML report from results.
    
    Args:
        results: Dictionary containing all results
        output_file: Path to save HTML report
    """
    evaluation = results.get('evaluation', {})
    predictions = results.get('predictions', {})
    
    # Extract key information
    model_info = evaluation.get('model_info', {})
    test_info = evaluation.get('test_info', {})
    metrics = evaluation.get('metrics', {})
    feature_importance = evaluation.get('feature_importance', [])
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Pipeline Training Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
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
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .header .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
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
            letter-spacing: 1px;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            max-width: 400px;
            margin: 20px auto;
        }}
        .cm-cell {{
            padding: 20px;
            text-align: center;
            border-radius: 5px;
            font-weight: bold;
        }}
        .cm-tp {{ background: #d4edda; color: #155724; }}
        .cm-tn {{ background: #d4edda; color: #155724; }}
        .cm-fp {{ background: #f8d7da; color: #721c24; }}
        .cm-fn {{ background: #f8d7da; color: #721c24; }}
        .feature-importance {{
            max-width: 600px;
            margin: 0 auto;
        }}
        .feature-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .feature-name {{
            width: 200px;
            text-align: right;
            padding-right: 10px;
            font-size: 0.9em;
        }}
        .feature-bar-fill {{
            height: 20px;
            background: linear-gradient(to right, #3498db, #2980b9);
            border-radius: 10px;
            min-width: 20px;
        }}
        .feature-value {{
            margin-left: 10px;
            font-size: 0.8em;
            color: #666;
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
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ML Pipeline Training Report</h1>
            <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="section">
            <h2>📊 Model Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('f1_score', 0):.3f}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            
            <h3>Confusion Matrix</h3>
            <div class="confusion-matrix">
                <div class="cm-cell cm-tp">
                    <div>True Positive</div>
                    <div>{metrics.get('confusion_matrix', {}).get('true_positive', 0)}</div>
                </div>
                <div class="cm-cell cm-fp">
                    <div>False Positive</div>
                    <div>{metrics.get('confusion_matrix', {}).get('false_positive', 0)}</div>
                </div>
                <div class="cm-cell cm-fn">
                    <div>False Negative</div>
                    <div>{metrics.get('confusion_matrix', {}).get('false_negative', 0)}</div>
                </div>
                <div class="cm-cell cm-tn">
                    <div>True Negative</div>
                    <div>{metrics.get('confusion_matrix', {}).get('true_negative', 0)}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Feature Importance</h2>
            <div class="feature-importance">
"""
    
    # Add feature importance bars
    if feature_importance:
        max_importance = max(imp[1] for imp in feature_importance[:10]) if feature_importance else 1
        
        for name, importance in feature_importance[:10]:
            bar_width = (importance / max_importance) * 300
            html_content += f"""
                <div class="feature-bar">
                    <div class="feature-name">{name}</div>
                    <div class="feature-bar-fill" style="width: {bar_width}px;"></div>
                    <div class="feature-value">{importance:.4f}</div>
                </div>
"""
    
    html_content += f"""
            </div>
        </div>

        <div class="section">
            <h2>ℹ️ Model Information</h2>
            <table class="info-table">
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Model Type</td>
                    <td>{model_info.get('model_type', 'Unknown')}</td>
                </tr>
                <tr>
                    <td>Training Accuracy</td>
                    <td>{model_info.get('training_accuracy', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Test Accuracy</td>
                    <td class="status-success">{metrics.get('accuracy', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Feature Count</td>
                    <td>{model_info.get('feature_count', 0)}</td>
                </tr>
                <tr>
                    <td>Test Samples</td>
                    <td>{test_info.get('test_samples', 0)}</td>
                </tr>
                <tr>
                    <td>Positive Samples</td>
                    <td>{test_info.get('positive_samples', 0)}</td>
                </tr>
                <tr>
                    <td>Negative Samples</td>
                    <td>{test_info.get('negative_samples', 0)}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>📈 Model Assessment</h2>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
"""
    
    # Add model assessment
    accuracy = metrics.get('accuracy', 0)
    f1_score = metrics.get('f1_score', 0)
    
    if accuracy >= 0.8 and f1_score >= 0.8:
        assessment = "🎉 <strong>Excellent Performance!</strong> The model shows strong performance across all metrics."
        status_class = "status-success"
    elif accuracy >= 0.7 and f1_score >= 0.7:
        assessment = "✅ <strong>Good Performance!</strong> The model performs well and is ready for deployment."
        status_class = "status-success"
    elif accuracy >= 0.6:
        assessment = "⚠️ <strong>Moderate Performance.</strong> Consider feature engineering or hyperparameter tuning."
        status_class = "status-warning"
    else:
        assessment = "❌ <strong>Poor Performance.</strong> Model needs significant improvement before deployment."
        status_class = "status-warning"
    
    html_content += f"""
                <p class="{status_class}">{assessment}</p>
                <ul>
                    <li><strong>Accuracy:</strong> {accuracy:.1%} - {'Excellent' if accuracy >= 0.8 else 'Good' if accuracy >= 0.7 else 'Needs improvement'}</li>
                    <li><strong>F1 Score:</strong> {f1_score:.1%} - {'Balanced performance' if f1_score >= 0.7 else 'May have precision/recall imbalance'}</li>
                    <li><strong>Data Quality:</strong> {test_info.get('test_samples', 0)} test samples available</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>Report generated by ML Pipeline Orchestrator</p>
            <p>Evaluation completed at {evaluation.get('evaluation_metadata', {}).get('evaluated_at', 'Unknown time')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate training report')
    parser.add_argument('--results', required=True, help='Directory with evaluation results')
    parser.add_argument('--output', required=True, help='Output path for HTML report')
    
    args = parser.parse_args()
    
    print(f"📊 Starting report generation script")
    print(f"📁 Results directory: {args.results}")
    print(f"📄 Output file: {args.output}")
    
    try:
        # Load results
        results = load_results(args.results)
        
        if not results:
            print("❌ No results found to generate report")
            return 1
        
        print(f"✅ Loaded results: {list(results.keys())}")
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        success = generate_html_report(results, output_path)
        
        if success:
            print(f"🎉 Report generated successfully!")
            print(f"📄 HTML report saved to: {output_path}")
            print(f"🌐 Open in browser: file://{output_path.absolute()}")
            return 0
        else:
            print("❌ Report generation failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())