from flask import Flask, request, jsonify, render_template
from utils.text_processing import calculate_cosine_similarity, convert_similarity_to_score


# Flask app setup
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate_essay():
    data = request.json

    # Validasi input
    if 'reference_essay' not in data or 'student_essay' not in data:
        return jsonify({"error": "Both 'reference_essay' and 'student_essay' are required."}), 400

    reference_essay = data['reference_essay']
    student_essay = data['student_essay']

    # Menghitung cosine similarity
    similarity_score = calculate_cosine_similarity(
        reference_essay, student_essay)
    final_score = convert_similarity_to_score(similarity_score)

    # Mengembalikan hasil
    return jsonify({
        "cosine_similarity": similarity_score,
        "score": final_score
    })


if __name__ == '__main__':
    app.run(debug=True)
