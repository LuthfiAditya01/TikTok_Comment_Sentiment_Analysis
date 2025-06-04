from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Cukup render halaman report.html; semua gambar ada di folder static/img
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
