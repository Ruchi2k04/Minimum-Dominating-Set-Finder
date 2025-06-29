from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
)
import subprocess
import os

app = Flask(__name__)
app.secret_key = "super secret key"


# Function to run the quantum algorithm with arguments
def process_graph(num_vertices, edge_pairs):
    subprocess.run(["python", "minDomGrovers.py", str(num_vertices), edge_pairs])
    return


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Form submission to build graph and process
@app.route("/build_graph", methods=["POST"])
def build_graph():
    if request.method == "POST":
        num_vertices = request.form.get("vertices")
        edge_pairs = request.form.get("edges")  # Format: 0-1,1-2

        if not num_vertices or not edge_pairs:
            return "Missing input", 400

        process_graph(num_vertices, edge_pairs)

    return render_template("result.html")


@app.route("/img1")
def input1():
    return send_from_directory(os.path.dirname(__file__) + "/result/", "img1.png")


@app.route("/img2")
def input2():
    return send_from_directory(os.path.dirname(__file__) + "/result/", "img2.png")


@app.route("/img3")
def input3():
    return send_from_directory(os.path.dirname(__file__) + "/result/", "img3.png")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
