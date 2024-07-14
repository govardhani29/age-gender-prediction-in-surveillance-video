from flask import Flask, render_template
from visualization import visualization_blueprint

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(visualization_blueprint, url_prefix='/visualization')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
