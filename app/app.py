import os
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
from math import sqrt
import numpy as np
import re
import traceback
from sklearn.linear_model import LinearRegression

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

def linear(x):
    return x

def square(x):
    return x**2

def cube(x):
    return x**3

def exponential(x):
    return 2**x

def square_root(x):
    return sqrt(x)

def quadratic(x):
    return x**2 + x

def cubic_minus(x):
    return x**3 - x**2

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

def is_fibonacci(numbers):
    if len(numbers) < 3:
        return False
    return all(abs(numbers[i] - (numbers[i-1] + numbers[i-2])) < 0.1
              for i in range(2, len(numbers)))

def next_fibonacci_terms(numbers, count=3):
    sequence = list(numbers)
    for _ in range(count):
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    return [round(x, 2) for x in sequence[-count:]]

def prod(lst):
    result = 1
    for x in lst:
        result *= x
    return result

def check_geometric_varying(numbers):
    if len(numbers) < 3:
        return False, None

    ratios = [numbers[i]/numbers[i-1] for i in range(1, len(numbers))]
    ratio_diffs = [ratios[i] - ratios[i-1] for i in range(1, len(ratios))]

    if len(set(round(diff, 6) for diff in ratio_diffs)) == 1:
        ratio_increment = ratio_diffs[0]
        next_ratio = ratios[-1] + ratio_increment
        return True, lambda n: numbers[-1] * next_ratio if n == len(numbers) + 1 else \
                             numbers[-1] * prod([ratios[-1] + (ratio_increment * i) for i in range(1, n - len(numbers) + 1)])
    return False, None

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

def identify_pattern_type(numbers):
    n = len(numbers)
    x = list(range(1, n + 1))

    if is_fibonacci(numbers):
        next_terms = next_fibonacci_terms(numbers)
        return "Fibonacci sequence: each number is the sum of the two preceding numbers", \
               lambda n: next_fibonacci_terms(numbers, 1)[0] if n == len(numbers) + 1 else \
                        next_fibonacci_terms(numbers, n - len(numbers))[-1] if n > len(numbers) else \
                        numbers[n-1]

    if len(set(np.diff(numbers))) == 1:
        d = numbers[1] - numbers[0]
        return f"arithmetic sequence: starts at {numbers[0]}, increases by {d}", \
               lambda n: numbers[0] + (n-1)*d

    if len(numbers) > 1:
        ratios = [numbers[i]/numbers[i-1] for i in range(1, len(numbers))]
        if len(set([round(r, 6) for r in ratios])) == 1:
            r = ratios[0]
            return f"geometric sequence: starts at {numbers[0]}, multiplies by {round(r,2)}", \
                   lambda n: numbers[0] * pow(r, n-1)

    if all(abs(i*i - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "square numbers: n²", square

    if all(abs(i**3 - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "cube numbers: n³", cube

    if all(abs(sqrt(i) - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "square roots: √n", square_root

    if all(abs((i*(i+1)/2) - num) < 0.1 for i, num in zip(range(1, n + 1), numbers)):
        return "triangular numbers: n(n+1)/2", lambda n: n*(n+1)/2

    is_varying_geometric, varying_formula = check_geometric_varying(numbers)
    if is_varying_geometric:
        ratios = [round(numbers[i]/numbers[i-1], 2) for i in range(1, len(numbers))]
        pattern_desc = f"geometric sequence with varying ratio (ratios: {ratios})"
        return pattern_desc, varying_formula

    coeffs = np.polyfit(x, numbers, min(n-1, 3))
    poly = np.poly1d(coeffs)

    residuals = [abs(poly(i+1) - num) for i, num in enumerate(numbers)]
    if max(residuals) < 0.1:
        return f"polynomial: degree {len(coeffs)-1}", poly

    try:
        model = LinearRegression()
        model.fit(np.array(x).reshape(-1, 1), numbers)
        next_value = model.predict(np.array([[n + 1]]))[0]
        return "Machine Learning (Linear Regression)", lambda n: model.predict(np.array([[n]]))[0]
    except Exception as e:
        print(f"ML Error: {e}")
        return "unknown pattern", lambda n: None

def generate_pattern(level):
    patterns = [
        (linear, "Linear"),
        (square, "Square"),
        (cube, "Cube"),
        (exponential, "Exponential"),
        (square_root, "Square Root"),
        (quadratic, "Quadratic"),
        (cubic_minus, "Cubic"),
        (fibonacci, "Fibonacci")
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
            return render_template('game.html', pattern=session['pattern'], error="Please enter a guess.", user=user)

        try:
            user_guess = float(user_guess)
        except ValueError:
            return render_template('game.html', pattern=session['pattern'], error="Invalid guess format.", user=user)

        if abs(user_guess - session['correct_answer']) < 0.1:
            if user:
                new_score = user.get('score', 0) + 10
                old_level = user.get('level', 1)
                new_level = min(new_score // 50 + 1, 8)
                update_data = {'score': new_score}
                if new_level > old_level:
                    update_data['level'] = new_level
                    update_data['level_score'] = new_score
                users_collection.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update_data})

            else:
                patterns_collection.insert_one({
                    'username': "not logged in",
                    'sequence': "game",
                    'solution': "game",
                    'timestamp': "na"
                })
            session['tries'] = 0
            session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(user.get('level', 1) if user else 1)

            return render_template('game.html', pattern=session['pattern'],
                                   success=f"Correct! Score: {user.get('score', 0) if user else 'Not logged in'}", user=user)
        else:
            session['tries'] += 1
            if session['tries'] >= 3:
                session['tries'] = 0
                if user:
                    user['score'] = user.get('level_score', 0)

                current_level = user.get('level', 1) if user else 1
                correct_answer = session['correct_answer']
                explanation = session['explanation']
                pattern_name = session['pattern_name']

                session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)

                return render_template('game.html',
                                    pattern=session['pattern'],
                                    error=f"Level reset! The answer was {correct_answer}. This was a {pattern_name} pattern: {explanation}",
                                    user=user)
            return render_template('game.html', pattern=session['pattern'],
                                error=f"Wrong! Tries left: {3-session['tries']}", user=user)

    current_level = user.get('level', 1) if user else 1
    session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_pattern(current_level)
    session['tries'] = 0
    return render_template('game.html', pattern=session['pattern'], user=user)

@app.route('/solver', methods=['GET', 'POST'])
def solver():
    user = None
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})

    if request.method == 'POST':
        pattern_str = request.form.get('pattern')
        try:
            numbers = [float(x.strip()) for x in pattern_str.split(',')]
            if len(numbers) < 3:
                return render_template('solver.html', error="Please enter at least 3 numbers", user=user)

            pattern_type, formula = identify_pattern_type(numbers)

            if formula is None:
                return render_template('solver.html',
                                    error="Unable to identify a clear pattern in the sequence",
                                    original_pattern=[format_number(x) for x in numbers],
                                    user=user)

            try:
                if callable(formula):
                    next_terms = [float(formula(len(numbers) + i)) for i in range(1, 4)]
                else:
                    next_terms = [float(formula(len(numbers) + i)) for i in range(1, 4)]

                next_terms = [format_number(x) for x in next_terms]
                formatted_original = [format_number(x) for x in numbers]

                if user:
                    patterns_collection.insert_one({
                        'username': user['username'],
                        'sequence': pattern_str,
                        'solution': f"Type: {pattern_type}, Next terms: {next_terms}",
                        'timestamp': datetime.datetime.now()
                    })
                else:
                    patterns_collection.insert_one({
                        'username': "not logged in",
                        'sequence': pattern_str,
                        'solution': f"Type: {pattern_type}, Next terms: {next_terms}",
                        'timestamp': "na"
                    })

                return render_template('solver.html',
                                    pattern_type=pattern_type,
                                    next_terms=next_terms,
                                    original_pattern=formatted_original,
                                    user=user)
            except (TypeError, ValueError):
                return render_template('solver.html',
                                    error="Unable to calculate next terms for this pattern",
                                    original_pattern=[format_number(x) for x in numbers],
                                    user=user)

        except (ValueError, ZeroDivisionError) as e:
            return render_template('solver.html', error=f"Invalid pattern format: {str(e)}", user=user)

    return render_template('solver.html', user=user)

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
