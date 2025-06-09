from flask import Flask, render_template, send_file, abort
import os

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/index')
# def index():
#     return render_template('index.html')

@app.route('/download/dataset')
def download_dataset():
    """Redirect to GitHub raw file for dataset download"""
    from flask import redirect
    # Direct GitHub raw URL - browser will auto-download
    github_url = "https://raw.githubusercontent.com/LuthfiAditya01/TikTok_Comment_Sentiment_Analysis/main/indonesia%20gelap.csv"
    return redirect(github_url)

@app.route('/download/report')
def download_report():
    """Redirect to GitHub raw file for report txt download"""
    from flask import redirect
    github_url = "https://github.com/LuthfiAditya01/TikTok_Comment_Sentiment_Analysis/raw/main/static/model-report.zip"
    return redirect(github_url)

# For Vercel deployment
if __name__ == '__main__':
    app.run(debug=True)
