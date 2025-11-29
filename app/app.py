import os
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
from math import sqrt, isclose
import numpy as np
import re
import traceback
from sklearn.linear_model import LinearRegression
import json
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
SECRET_KEY = os.getenv('SECRET_KEY')

# Initialize MongoDB client and database
client = MongoClient(MONGODB_URI)
db = client['PatternSolverDB']
users_collection = db['users']
patterns_collection = db['patterns']
comments_collection = db['comments']

# Flask app initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY

### --- OEIS integration --- ###
OEIS_CACHE_PATH = 'app/processed_oeis_data.json'

with open('app/processed_oeis_data.json', 'r', encoding='utf-8') as f:
    oeis_data = json.load(f)
print("OEIS loaded, keys:", list(oeis_data.keys())[:3])
first = list(oeis_data.items())[0]
print("First OEIS entry:", first)
print("OEIS entries count:", len(oeis_data))

try:
    with open('app/processed_oeis_data.json', 'r', encoding='utf-8') as f:
        oeis_data = json.load(f)
except Exception as e:
    print("Failed to load OEIS JSON:", e)
    oeis_data = {}

def load_oeis_cache(cache_path):
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None

oeis_data = load_oeis_cache(OEIS_CACHE_PATH)

def get_next_terms_from_data(full_sequence_list, input_sequence_list):
    len_input = len(input_sequence_list)
    len_full = len(full_sequence_list)
    for i in range(len_full - len_input + 1):
        if full_sequence_list[i : i + len_input] == input_sequence_list:
            start_index_next = i + len_input
            end_index_next = start_index_next + 3
            if end_index_next <= len_full:
                next_terms = full_sequence_list[start_index_next : end_index_next]
                return next_terms
            else:
                available_terms = full_sequence_list[start_index_next:]
                return available_terms
    return []

def search_oeis(user_input_str, processed_data):
    input_numbers_str = re.findall(r"-?\d+", user_input_str)
    if not input_numbers_str:
        return None, None
    input_numbers_list = [int(n) for n in input_numbers_str]
    search_pattern_str = "," + ",".join(input_numbers_str) + ","
    sorted_a_nums = sorted(processed_data.keys(), key=lambda x: int(x[1:]))
    for a_num in sorted_a_nums:
        entry = processed_data[a_num]
        seq_string = entry["sequence_str"]
        if search_pattern_str in ("," + seq_string):
            try:
                full_num_list = [int(n) for n in seq_string.rstrip(',').split(',') if n]
                next_terms = get_next_terms_from_data(full_num_list, input_numbers_list)
                if next_terms:
                    desc = f"OEIS: {entry['description']} ({a_num})"
                    return desc, next_terms
            except Exception:
                continue
    return None, None

# --- Utility Functions (for pattern) ---
def normalize_pattern_input(pattern_str):
    # Accept both comma and space separated, ignore extra whitespace
    # Accept input like "1,2,3 4, 5" or "1 2 3  4" etc.
    clean = re.sub(r'[,\s]+', ' ', pattern_str.strip())
    numbers = re.findall(r'-?\d+', clean)
    return [int(x) for x in numbers]

def is_perfect_square(n):
    return isclose(sqrt(n), round(sqrt(n))) and n >= 0

def is_perfect_cube(n):
    if n < 0:
        root = round(-abs(n) ** (1/3))
    else:
        root = round(n ** (1/3))
    return isclose(root ** 3, n)

def all_squares(lst):
    return all(is_perfect_square(x) for x in lst)

def all_cubes(lst):
    return all(is_perfect_cube(x) for x in lst)

def get_square_roots(lst):
    return [int(round(sqrt(x))) for x in lst]

def get_cube_roots(lst):
    def cube_root(n):
        if n < 0:
            return -round(abs(n) ** (1/3))
        else:
            return round(n ** (1/3))
    return [cube_root(x) for x in lst]

def is_arithmetic(numbers):
    if len(numbers) < 2:
        return False
    d = numbers[1] - numbers[0]
    return all(numbers[i+1] - numbers[i] == d for i in range(len(numbers)-1))

def is_geometric(numbers):
    if len(numbers) < 2 or 0 in numbers:
        return False
    r = numbers[1] / numbers[0]
    return all(isclose(numbers[i+1] / numbers[i], r) for i in range(len(numbers)-1))

def is_fibonacci(numbers):
    if len(numbers) < 3:
        return False
    return all(numbers[i] == numbers[i-1] + numbers[i-2] for i in range(2, len(numbers)))

def next_arith(numbers, n=3):
    d = numbers[1] - numbers[0]
    last = numbers[-1]
    return [last + d * (i+1) for i in range(n)]

def next_geo(numbers, n=3):
    r = numbers[1] / numbers[0]
    last = numbers[-1]
    return [last * (r ** (i+1)) for i in range(n)]

def next_fib(numbers, n=3):
    seq = numbers[:]
    for _ in range(n):
        seq.append(seq[-1] + seq[-2])
    return seq[-n:]

def is_triangular(numbers):
    def inv_tri(x):
        n = (-1 + sqrt(1 + 8*x)) / 2
        return isclose(n, round(n))
    return all(inv_tri(x) for x in numbers)

def next_triangular(numbers, n=3):
    start_n = int(round((-1 + sqrt(1 + 8*numbers[-1]))/2))
    return [int((start_n + i)*(start_n + i + 1)//2) for i in range(1, n+1)]

def identify_pattern_type(numbers):
    # 1. Arithmetic (first)
    if is_arithmetic(numbers):
        return "Arithmetic sequence", next_arith(numbers)
    # 2. Fibonacci
    if is_fibonacci(numbers):
        return "Fibonacci sequence", next_fib(numbers)
    # 3. Square (direct, increasing or decreasing)
    if all_squares(numbers):
        roots = [round(sqrt(x)) for x in numbers]
        diffs = [roots[i+1] - roots[i] for i in range(len(roots)-1)]
        if all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [roots[-1] + diffs[0]*(i+1) for i in range(3)]
            next_terms = [r**2 for r in next_roots]
            return f"Square numbers (n², {direction})", next_terms
    # 4. Cube (direct, increasing or decreasing)
    if all_cubes(numbers):
        def cube_root(n):
            if n < 0:
                return -round(abs(n) ** (1/3))
            else:
                return round(n ** (1/3))
        roots = [cube_root(x) for x in numbers]
        diffs = [roots[i+1] - roots[i] for i in range(len(roots)-1)]
        if all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [roots[-1] + diffs[0]*(i+1) for i in range(3)]
            next_terms = [r**3 for r in next_roots]
            return f"Cube numbers (n³, {direction})", next_terms
    # 5. Square root (direct, increasing or decreasing)
    if all(is_perfect_square(x) and x >= 0 for x in numbers):
        sqrt_seq = [int(round(sqrt(x))) for x in numbers]
        diffs = [sqrt_seq[i+1] - sqrt_seq[i] for i in range(len(sqrt_seq)-1)]
        if all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [sqrt_seq[-1] + diffs[0]*(i+1) for i in range(3)]
            next_terms = [r**2 for r in next_roots]
            return f"Square roots (√n, {direction})", next_terms
    # 6. Cube root (direct, increasing or decreasing)
    def cube_root(n):
        if n < 0:
            return -round(abs(n) ** (1/3))
        else:
            return round(n ** (1/3))
    if all(is_perfect_cube(x) for x in numbers):
        cube_seq = [cube_root(x) for x in numbers]
        diffs = [cube_seq[i+1] - cube_seq[i] for i in range(len(cube_seq)-1)]
        if all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [cube_seq[-1] + diffs[0]*(i+1) for i in range(3)]
            next_terms = [r**3 for r in next_roots]
            return f"Cube roots (³√n, {direction})", next_terms
    # 7. Geometric (direct)
    if is_geometric(numbers):
        return "Geometric sequence", next_geo(numbers)
    # 8. Triangular (direct)
    if is_triangular(numbers):
        return "Triangular numbers", next_triangular(numbers)
        # 9. OEIS
    if oeis_data:
        desc, next_terms = search_oeis(','.join(str(x) for x in numbers), oeis_data)
        if desc and next_terms:
            desc = re.sub(r'^OEIS:\s*', '', desc)
            desc = re.sub(r'\s*\(A\d+\)$', '', desc)
            return desc, next_terms
    # 10. Varying geometric
    ratios = []
    try:
        ratios = [numbers[i+1]/numbers[i] for i in range(len(numbers)-1) if numbers[i] != 0]
    except Exception:
        ratios = []
    if len(ratios) >= 2 and len(set(round(r, 6) for r in ratios)) > 1:
        return f"Geometric sequence with varying ratio (ratios: {[format(r, '.2f') for r in ratios]})", [
            round(numbers[-1] * ratios[-1], 2),
            round(numbers[-1] * ratios[-1] * (ratios[-2] if len(ratios) > 1 else ratios[-1]), 2),
            round(numbers[-1] * ratios[-1] * (ratios[-2] if len(ratios) > 1 else ratios[-1]) * (ratios[-3] if len(ratios) > 2 else ratios[-1]), 2),
        ]
    # 11. Polynomial regression
    degree = min(3, len(numbers)-1)
    x = np.arange(1, len(numbers)+1)
    y = np.array(numbers)
    try:
        coefs = np.polyfit(x, y, degree)
        poly = np.poly1d(coefs)
        next_terms = [float(poly(i)) for i in range(len(numbers)+1, len(numbers)+4)]
        return f"Polynomial (degree {degree}) regression", [round(x,2) for x in next_terms]
    except Exception:
        pass
    # 12. Machine learning regression fallback
    try:
        model = LinearRegression()
        model.fit(np.array(x).reshape(-1, 1), y)
        next_terms = [model.predict(np.array([[len(numbers)+i]]))[0] for i in range(1, 4)]
        return "Machine Learning (Linear Regression)", [round(x, 2) for x in next_terms]
    except Exception:
        pass
    # 13. Unknown
    return "Unknown pattern", []

# --- Password Hashing ---
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def format_number(num):
    if isinstance(num, int):
        return num
    if isinstance(num, float):
        if num.is_integer():
            return int(num)
        return round(num, 2)
    return num

# --- Flask routes (all unchanged except solver/game as requested) ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        user = None
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            bad_words = ["anal", "anus", "arse", "ass", "ballsack", "balls", "bastard", "bitch", "biatch", "bloody",
    "blowjob", "blow job", "bollock", "bollok", "boner", "boob", "bugger", "bum", "butt", "buttplug",
    "clitoris", "cock", "coon", "crap", "cunt", "damn", "dick", "dildo", "dyke", "fag", "feck",
    "fellate", "fellatio", "felching", "fuck", "f u c k", "fudgepacker", "fudge packer", "flange",
    "Goddamn", "God damn", "hell", "homo", "jerk", "jizz", "knobend", "knob end", "labia", "lmao",
    "lmfao", "muff", "nigger", "nigga", "omg", "penis", "piss", "poop", "prick", "pube", "pussy",
    "queer", "scrotum", "sex", "shit", "sh1t", "slut", "smegma", "spunk", "tit", "tosser",
    "turd", "twat", "vagina", "wank", "whore", "wtf"]
            for word in bad_words:
                if word.lower() in username.lower():
                    return render_template('register.html', error="Username contains inappropriate words.", user=user)

            if users_collection.find_one({'username': username}):
                return render_template('register.html', error="Username already taken", user=user)

            hashed_password = hash_password(password)
            users_collection.insert_one({
                'username': username,
                'password': hashed_password,
                'score': 0,
                'level': 1,
                'level_score':0,
                'patterns': [],
            })
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        return render_template('register.html', user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        user = None
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            user = users_collection.find_one({'username': username})
            if user and check_password(password, user['password']):
                session['user_id'] = str(user['_id'])
                return redirect(url_for('home'))
            flash('Invalid username or password')
        return render_template('login.html', user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/logout')
def logout():
    try:
        session.pop('user_id', None)
        return redirect(url_for('home'))
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/')
def home():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('home.html', user=user)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
   try:
        user = None
        patterns = None
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
            patterns = list(patterns_collection.find({'username': user['username']}).sort('timestamp', -1).limit(10))

        return render_template('settings.html', user=user, patterns=patterns)
   except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/game', methods=['GET', 'POST'])
def game():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    if request.method == 'POST':
        user_guess = request.form.get('guess')
        if not user_guess:
            session['game_error'] = "Please enter a guess."
            return redirect(url_for('game'))
        try:
            user_guess = float(user_guess)
        except ValueError:
            session['game_error'] = "Invalid guess format."
            return redirect(url_for('game'))

        if abs(user_guess - session.get('correct_answer', 0)) < 0.1:
            if user:
                new_score = user.get('score', 0) + 10
                old_level = user.get('level', 1)
                new_level = min(new_score // 50 + 1, 8)
                update_data = {'score': new_score}
                if new_level > old_level:
                    update_data['level'] = new_level
                    update_data['level_score'] = new_score
                users_collection.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update_data})
                user = users_collection.find_one({'_id': ObjectId(session['user_id'])}) # Refresh user object

            else:
                patterns_collection.insert_one({
                    'username': "not logged in",
                    'sequence': "game",
                    'solution': "game",
                    'timestamp': datetime.datetime.now()
                })
            session['tries'] = 0
            # Generate new pattern for next round
            session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(user.get('level', 1) if user else 1)
            session['game_success'] = f"Correct! Score: {user.get('score', 0) if user else 'Not logged in'}"
            return redirect(url_for('game'))
        else:
            session['tries'] = session.get('tries', 0) + 1
            if session['tries'] >= 3:
                session['tries'] = 0
                if user:
                    level_score = user.get('level_score', 0)
                    update_data = {'score': level_score}
                    users_collection.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update_data})
                    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})  # Refresh user object

                current_level = user.get('level', 1) if user else 1
                correct_answer = session['correct_answer']
                explanation = session['explanation']
                pattern_name = session['pattern_name']
                session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)
                session['game_error'] = f"Level reset! The answer was {correct_answer}. This was a {pattern_name} pattern: {explanation}"
                return redirect(url_for('game'))
            session['game_error'] = f"Wrong! Tries left: {3-session['tries']}"
            return redirect(url_for('game'))

    # GET
    current_level = user.get('level', 1) if user else 1
    if 'pattern' not in session or 'correct_answer' not in session:
        session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)
        session['tries'] = 0
    pattern = session['pattern']
    error = session.pop('game_error', None)
    success = session.pop('game_success', None)
    return render_template('game.html', pattern=pattern, error=error, success=success, user=user)

def generate_pattern(level):
    patterns = [
        (lambda x: x, "Linear"),
        (lambda x: x**2, "Square"),
        (lambda x: x**3, "Cube"),
        (lambda x: 2**x, "Exponential"),
        (lambda x: sqrt(x), "Square Root"),
        (lambda x: x**2 + x, "Quadratic"),
        (lambda x: x**3 - x**2, "Cubic"),
        (lambda n: fibonacci(n), "Fibonacci")
    ]

    pattern_func, pattern_name = patterns[min(level-1, len(patterns)-1)]
    length = np.random.randint(4, 7)
    start = np.random.randint(1, 11)
    is_ascending = np.random.choice([True, False])

    if is_ascending:
        sequence = [format_number(float(pattern_func(i))) for i in range(start, start + length)]
        next_term = format_number(float(pattern_func(start + length)))
    else:
        sequence = [format_number(float(pattern_func(i))) for i in range(start + length - 1, start - 1, -1)]
        next_term = format_number(float(pattern_func(start - 1)))
    if next_term <= 0:
        return generate_pattern(level)
    explanation = get_pattern_explanation(pattern_name)
    return sequence, next_term, explanation, pattern_name

def get_pattern_explanation(pattern_type):
    explanations = {
        "Linear": "Linear pattern: Each number increases by 1",
        "Square": "Square numbers: Each number is n²",
        "Cube": "Cube numbers: Each number is n³",
        "Exponential": "Powers of 2: Each number is 2ⁿ",
        "Square Root": "Square root pattern: Each number is √n",
        "Quadratic": "Quadratic pattern: Each number is n² + n",
        "Cubic": "Cubic pattern: Each number is n³ - n²",
        "Fibonacci": "Fibonacci sequence: Each number is the sum of the two preceding numbers"
    }
    return explanations.get(pattern_type, "Complex pattern: Try to spot the mathematical relationship")

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

@app.route('/solver', methods=['GET', 'POST'])
def solver():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    error = None
    pattern_type = None
    next_terms = None
    original_pattern = None
    pattern_input = ""

    if request.method == 'POST':
        pattern_input = request.form.get('pattern', '')
        try:
            numbers = normalize_pattern_input(pattern_input)
            if len(numbers) < 2:
                raise ValueError("Please enter at least two numbers.")
            pattern_type, next_terms = identify_pattern_type(numbers)
            original_pattern = numbers

            # Restore original behavior: save the solved pattern to MongoDB (for user's history/settings)
            if user:
                patterns_collection.insert_one({
                    'username': user['username'],
                    'sequence': pattern_input,
                    'solution': f"Type: {pattern_type}, Next terms: {next_terms}",
                    'timestamp': datetime.datetime.now()
                })
            else:
                patterns_collection.insert_one({
                    'username': "not logged in",
                    'sequence': pattern_input,
                    'solution': f"Type: {pattern_type}, Next terms: {next_terms}",
                    'timestamp': datetime.datetime.now()
                })
        except Exception as ex:
            error = f"Error: {str(ex)}"
    elif request.method == 'GET':
        pattern_input = request.args.get('pattern', '')

    return render_template('solver.html', error=error, pattern_type=pattern_type, next_terms=next_terms,
                           original_pattern=original_pattern, user=user, pattern_input=pattern_input)

# --- The rest of your routes remain unchanged! ---

@app.route('/about')
def about():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('about.html', user=user)

@app.route('/pattern_library')
def pattern_library():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('pattern_library.html', user=user)

@app.route('/blog')
def blog():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    blog_posts = {
        1: {"title": "The Beauty of Fibonacci Sequence", "date": "2025-02-12", "summary": "Discover the fascinating properties and applications of the Fibonacci sequence in nature and art."},
        2: {"title": "Unlocking the Secrets of Prime Numbers", "date": "2025-02-12", "summary": "Learn about the mysteries of prime numbers and their role in cryptography and computer science."},
        3: {"title": "The Power of Geometric Patterns", "date": "2025-02-12", "summary": "Explore the applications of geometric patterns in architecture, design, and mathematics."},
        4: {"title": "How to Identify Arithmetic Sequences", "date": "2025-02-12", "summary": "A guide to identifying and working with arithmetic sequences, with real-world examples."},
        5: {"title": "Polynomial Patterns in Data Analysis", "date": "2025-02-12", "summary": "How polynomial functions can be used to model and analyze data patterns."},
        6: {"title": "The Mathematics of Music: Unveiling Hidden Patterns", "date": "2025-02-12", "summary": "Explore the connections between music theory and mathematical principles."},
        7: {"title": "The Art of Problem Solving: Strategies and Techniques", "date": "2025-02-12", "summary": "Discover effective strategies and techniques for tackling complex problems in various fields."},
        8: {"title": "Mathematics in Nature: Beyond Fibonacci", "date": "2025-02-12", "summary": "Explore the mathematical principles that govern the natural world beyond just the Fibonacci sequence."}
    }
    return render_template('blog.html', user=user, blog_posts=blog_posts)

@app.route('/blog/<int:post_id>')
def blog_post(post_id):
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    blog_posts = {
        1: {"title": "The Beauty of Fibonacci Sequence", "date": "2025-02-12", "content": render_template('blog_content/fibonacci.html')},
        2: {"title": "Unlocking the Secrets of Prime Numbers", "date": "2025-02-12", "content": render_template('blog_content/prime_numbers.html')},
        3: {"title": "The Power of Geometric Patterns", "date": "2025-02-12", "content": render_template('blog_content/geometric_patterns.html')},
        4: {"title": "How to Identify Arithmetic Sequences", "date": "2025-02-12", "content": render_template('blog_content/arithmetic_sequences.html')},
        5: {"title": "Polynomial Patterns in Data Analysis", "date": "2025-02-12", "content": render_template('blog_content/polynomial_patterns.html')},
        6: {"title": "The Mathematics of Music: Unveiling Hidden Patterns", "date": "2025-02-12", "content": render_template('blog_content/mathematics_of_music.html')},
        7: {"title": "The Art of Problem Solving: Strategies and Techniques", "date": "2025-02-12", "content": render_template('blog_content/art_of_problem_solving.html')},
        8: {"title": "Mathematics in Nature: Beyond Fibonacci", "date": "2025-02-12", "content": render_template('blog_content/mathematics_in_nature.html')},
    }

    post = blog_posts.get(post_id)
    if post:
        comments = list(comments_collection.find({'post_id': post_id}).sort('timestamp', -1))
        return render_template('blog_post.html', title=post["title"], date=post["date"], content=post["content"], user=user, comments=comments, post_id=post_id)
    else:
        return "Blog post not found"

@app.route('/add_comment', methods=['POST'])
def add_comment():
    if 'user_id' not in session:
        flash('You must be logged in to comment.')
        return redirect(url_for('login'))

    post_id = int(request.form.get('post_id'))
    comment_text = request.form.get('comment')

    if not comment_text:
        flash('Comment cannot be empty.')
        return redirect(url_for('blog_post', post_id=post_id))

    user_id = ObjectId(session['user_id'])
    user = users_collection.find_one({'_id': user_id})
    username = user['username']

    timestamp = datetime.datetime.now()

    comments_collection.insert_one({
        'post_id': post_id,
        'username': username,
        'text': comment_text,
        'timestamp': timestamp
    })

    flash('Comment added successfully!')
    return redirect(url_for('blog_post', post_id=post_id))

@app.route('/google925b0ffc084ab8b1.html')
def serve_verification_file():
    try:
        return send_from_directory(app.static_folder, 'google925b0ffc084ab8b1.html')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/471cae572a7e45cfbcf4e59b54108d04.txt')
def serve_471cae572a7e45cfbcf4e59b54108d04_file():
    try:
        return send_from_directory(app.static_folder, '471cae572a7e45cfbcf4e59b54108d04.txt')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/sitemap.xml')
def serve_sitemap():
    try:
        return send_from_directory(app.static_folder, 'sitemap.xml')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/ads.txt')
def serve_ads_file():
    try:
        return send_from_directory(app.static_folder, 'Ads.txt')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/privacy-policy')
def privacy_policy():
    try:
        user = None
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        return render_template('privacy-policy.html', user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/cookie-consent')
def cookie_consent():
    try:
        user = None
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        return render_template('cookie_consent.html', user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/ezoic_scripts')
def inject_ezoic_scripts():
    try:
        user = None
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        return render_template('ezoic_scripts.html', user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/leaderboard')
def leaderboard():
    try:
        user = None
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        top_scores = list(users_collection.find().sort('score', -1).limit(10))
        return render_template('leaderboard.html', top_scores=top_scores, user=user)
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/robots.txt')
def serve_rob():
    try:
        return send_from_directory(app.static_folder, 'robots.txt')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/.well-known/pki-validation/7D3D3BA0B414B564BF03E299F5FCB2D3.txt')
def serve_val():
    try:
        return send_from_directory(app.static_folder, '7D3D3BA0B414B564BF03E299F5FCB2D3.txt')
    except Exception as e:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        if not feedback_text:
            flash('Feedback cannot be empty.')
            return render_template('feedback.html', user=user)

        db.feedback.insert_one({
            'username': user['username'],
            'feedback': feedback_text,
            'timestamp': datetime.datetime.now()
        })
        flash('Thank you for your feedback!')
        return redirect(url_for('home'))

    return render_template('feedback.html', user=user)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
