"""
Pattern Solver - Flask app for identifying and solving number sequences.
"""
import os
import re
import json
import datetime
import traceback
from math import sqrt, isclose

import bcrypt
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, 
    session, flash, send_from_directory
)
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['PatternSolverDB']
users = db['users']
patterns = db['patterns']
comments = db['comments']

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Load OEIS sequence data
OEIS_PATH = 'app/processed_oeis_data.json'
try:
    with open(OEIS_PATH, 'r', encoding='utf-8') as f:
        oeis_data = json.load(f)
    print(f"Loaded {len(oeis_data)} OEIS sequences")
except Exception as e:
    print(f"Failed to load OEIS data: {e}")
    oeis_data = {}

# ─────────────────────────────────────────────────────────────────────────────
# OEIS Lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_next_terms(full_seq, input_seq):
    """Find the next 3 terms after input_seq appears in full_seq."""
    n = len(input_seq)
    for i in range(len(full_seq) - n + 1):
        if full_seq[i:i + n] == input_seq:
            start = i + n
            return full_seq[start:start + 3]
    return []


def search_oeis(user_input, data):
    """Search OEIS data for a matching sequence."""
    nums = re.findall(r"-?\d+", user_input)
    if not nums:
        return None, None
    
    input_list = [int(n) for n in nums]
    search_str = "," + ",".join(nums) + ","
    
    # Check sequences in order
    for a_num in sorted(data.keys(), key=lambda x: int(x[1:])):
        entry = data[a_num]
        if search_str in ("," + entry["sequence_str"]):
            try:
                full_list = [int(n) for n in entry["sequence_str"].rstrip(',').split(',') if n]
                next_terms = find_next_terms(full_list, input_list)
                if next_terms:
                    return f"OEIS: {entry['description']} ({a_num})", next_terms
            except Exception:
                continue
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Analysis Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_sequence(text):
    """Parse comma or space-separated numbers into a list of ints."""
    clean = re.sub(r'[,\s]+', ' ', text.strip())
    return [int(x) for x in re.findall(r'-?\d+', clean)]


def is_perfect_square(n):
    return n >= 0 and isclose(sqrt(n), round(sqrt(n)))


def is_perfect_cube(n):
    root = round(abs(n) ** (1/3))
    if n < 0:
        root = -root
    return isclose(root ** 3, n)


def cube_root(n):
    """Return the integer cube root of n."""
    if n < 0:
        return -round(abs(n) ** (1/3))
    return round(n ** (1/3))


def is_arithmetic(nums):
    if len(nums) < 2:
        return False
    d = nums[1] - nums[0]
    return all(nums[i+1] - nums[i] == d for i in range(len(nums) - 1))


def is_geometric(nums):
    if len(nums) < 2 or 0 in nums:
        return False
    r = nums[1] / nums[0]
    return all(isclose(nums[i+1] / nums[i], r) for i in range(len(nums) - 1))


def is_fibonacci(nums):
    if len(nums) < 3:
        return False
    return all(nums[i] == nums[i-1] + nums[i-2] for i in range(2, len(nums)))


def is_triangular(nums):
    """Check if all numbers are triangular numbers."""
    def check(x):
        n = (-1 + sqrt(1 + 8*x)) / 2
        return isclose(n, round(n))
    return all(check(x) for x in nums)


def next_arithmetic(nums, count=3):
    d = nums[1] - nums[0]
    return [nums[-1] + d * (i + 1) for i in range(count)]


def next_geometric(nums, count=3):
    r = nums[1] / nums[0]
    return [nums[-1] * (r ** (i + 1)) for i in range(count)]


def next_fibonacci(nums, count=3):
    seq = nums[:]
    for _ in range(count):
        seq.append(seq[-1] + seq[-2])
    return seq[-count:]


def next_triangular(nums, count=3):
    n = int(round((-1 + sqrt(1 + 8 * nums[-1])) / 2))
    return [int((n + i) * (n + i + 1) // 2) for i in range(1, count + 1)]

def identify_pattern_type(nums):
    """
    Analyse a sequence and identify its pattern type.
    Returns (pattern_description, next_three_terms).
    """
    # Arithmetic - most common, check first
    if is_arithmetic(nums):
        return "Arithmetic sequence", next_arithmetic(nums)
    
    # Fibonacci
    if is_fibonacci(nums):
        return "Fibonacci sequence", next_fibonacci(nums)
    
    # Perfect squares (ascending or descending)
    if all(is_perfect_square(x) for x in nums):
        roots = [round(sqrt(x)) for x in nums]
        diffs = [roots[i+1] - roots[i] for i in range(len(roots) - 1)]
        if diffs and all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [roots[-1] + diffs[0] * (i + 1) for i in range(3)]
            return f"Square numbers (n², {direction})", [r ** 2 for r in next_roots]
    
    # Perfect cubes (ascending or descending)
    if all(is_perfect_cube(x) for x in nums):
        roots = [cube_root(x) for x in nums]
        diffs = [roots[i+1] - roots[i] for i in range(len(roots) - 1)]
        if diffs and all(isclose(d, diffs[0], abs_tol=1e-8) for d in diffs):
            direction = 'decreasing' if diffs[0] < 0 else 'increasing'
            next_roots = [roots[-1] + diffs[0] * (i + 1) for i in range(3)]
            return f"Cube numbers (n³, {direction})", [r ** 3 for r in next_roots]
    
    # Geometric
    if is_geometric(nums):
        return "Geometric sequence", next_geometric(nums)
    
    # Triangular
    if is_triangular(nums):
        return "Triangular numbers", next_triangular(nums)
    
    # OEIS lookup
    if oeis_data:
        desc, next_terms = search_oeis(','.join(str(x) for x in nums), oeis_data)
        if desc and next_terms:
            # Strip the OEIS prefix/suffix for cleaner output
            desc = re.sub(r'^OEIS:\s*', '', desc)
            desc = re.sub(r'\s*\(A\d+\)$', '', desc)
            return desc, next_terms
    
    # Varying geometric ratio
    ratios = []
    try:
        ratios = [nums[i+1] / nums[i] for i in range(len(nums) - 1) if nums[i] != 0]
    except Exception:
        pass
    
    if len(ratios) >= 2 and len(set(round(r, 6) for r in ratios)) > 1:
        r1, r2, r3 = ratios[-1], ratios[-2] if len(ratios) > 1 else ratios[-1], ratios[-3] if len(ratios) > 2 else ratios[-1]
        next_terms = [
            round(nums[-1] * r1, 2),
            round(nums[-1] * r1 * r2, 2),
            round(nums[-1] * r1 * r2 * r3, 2),
        ]
        return f"Geometric sequence with varying ratio (ratios: {[format(r, '.2f') for r in ratios]})", next_terms
    
    # Polynomial regression fallback
    degree = min(3, len(nums) - 1)
    x = np.arange(1, len(nums) + 1)
    y = np.array(nums)
    
    try:
        coefs = np.polyfit(x, y, degree)
        poly = np.poly1d(coefs)
        next_terms = [float(poly(i)) for i in range(len(nums) + 1, len(nums) + 4)]
        return f"Polynomial (degree {degree}) regression", [round(t, 2) for t in next_terms]
    except Exception:
        pass
    
    # Linear regression as last resort
    try:
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        next_terms = [model.predict([[len(nums) + i]])[0] for i in range(1, 4)]
        return "Machine Learning (Linear Regression)", [round(t, 2) for t in next_terms]
    except Exception:
        pass
    
    return "Unknown pattern", []

# ─────────────────────────────────────────────────────────────────────────────
# Password & Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password):
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def check_password(password, hashed):
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)


def format_num(num):
    """Format numbers for display - round floats, keep ints clean."""
    if isinstance(num, float):
        return int(num) if num.is_integer() else round(num, 2)
    return num


# Words we don't allow in usernames
BLOCKED_WORDS = [
    "anal", "anus", "arse", "ass", "ballsack", "balls", "bastard", "bitch", "biatch",
    "bloody", "blowjob", "blow job", "bollock", "bollok", "boner", "boob", "bugger",
    "bum", "butt", "buttplug", "clitoris", "cock", "coon", "crap", "cunt", "damn",
    "dick", "dildo", "dyke", "fag", "feck", "fellate", "fellatio", "felching", "fuck",
    "f u c k", "fudgepacker", "fudge packer", "flange", "Goddamn", "God damn", "hell",
    "homo", "jerk", "jizz", "knobend", "knob end", "labia", "lmao", "lmfao", "muff",
    "nigger", "nigga", "omg", "penis", "piss", "poop", "prick", "pube", "pussy",
    "queer", "scrotum", "sex", "shit", "sh1t", "slut", "smegma", "spunk", "tit",
    "tosser", "turd", "twat", "vagina", "wank", "whore", "wtf"
]


def get_current_user():
    """Fetch the current logged-in user, or None."""
    if 'user_id' not in session:
        return None
    return users.find_one({'_id': ObjectId(session['user_id'])})


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html', user=get_current_user())


@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method != 'POST':
            return render_template('register.html', user=None)
        
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check for inappropriate words
        for word in BLOCKED_WORDS:
            if word.lower() in username.lower():
                return render_template('register.html', error="Username contains inappropriate words.", user=None)
        
        if users.find_one({'username': username}):
            return render_template('register.html', error="Username already taken", user=None)
        
        users.insert_one({
            'username': username,
            'password': hash_password(password),
            'score': 0,
            'level': 1,
            'level_score': 0,
            'patterns': [],
        })
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    except Exception:
        print(traceback.format_exc())
        return "Internal Server Error", 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method != 'POST':
            return render_template('login.html', user=None)
        
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users.find_one({'username': username})
        if user and check_password(password, user['password']):
            session['user_id'] = str(user['_id'])
            return redirect(url_for('home'))
        
        flash('Invalid username or password')
        return render_template('login.html', user=None)
    except Exception:
        print(traceback.format_exc())
        return "Internal Server Error", 500


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    try:
        user = get_current_user()
        user_patterns = None
        if user:
            user_patterns = list(patterns.find({'username': user['username']}).sort('timestamp', -1).limit(10))
        return render_template('settings.html', user=user, patterns=user_patterns)
    except Exception:
        print(traceback.format_exc())
        return "Internal Server Error", 500

@app.route('/game', methods=['GET', 'POST'])
def game():
    user = get_current_user()
    
    if request.method == 'POST':
        guess = request.form.get('guess')
        if not guess:
            session['game_error'] = "Please enter a guess."
            return redirect(url_for('game'))
        
        try:
            guess = float(guess)
        except ValueError:
            session['game_error'] = "Invalid guess format."
            return redirect(url_for('game'))
        
        correct = session.get('correct_answer', 0)
        
        # Correct answer
        if abs(guess - correct) < 0.1:
            if user:
                new_score = user.get('score', 0) + 10
                old_level = user.get('level', 1)
                new_level = min(new_score // 50 + 1, 8)
                
                update = {'score': new_score}
                if new_level > old_level:
                    update['level'] = new_level
                    update['level_score'] = new_score
                
                users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update})
                user = get_current_user()
            else:
                # Track anonymous plays
                patterns.insert_one({
                    'username': "not logged in",
                    'sequence': "game",
                    'solution': "game",
                    'timestamp': datetime.datetime.now()
                })
            
            session['tries'] = 0
            level = user.get('level', 1) if user else 1
            session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_game_pattern(level)
            session['game_success'] = f"Correct! Score: {user.get('score', 0) if user else 'Not logged in'}"
            return redirect(url_for('game'))
        
        # Wrong answer
        session['tries'] = session.get('tries', 0) + 1
        
        if session['tries'] >= 3:
            # Out of tries - reset level
            session['tries'] = 0
            if user:
                level_score = user.get('level_score', 0)
                users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': {'score': level_score}})
                user = get_current_user()
            
            answer = session['correct_answer']
            name = session['pattern_name']
            explanation = session['explanation']
            
            level = user.get('level', 1) if user else 1
            session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_game_pattern(level)
            session['game_error'] = f"Level reset! The answer was {answer}. This was a {name} pattern: {explanation}"
            return redirect(url_for('game'))
        
        session['game_error'] = f"Wrong! Tries left: {3 - session['tries']}"
        return redirect(url_for('game'))
    
    # GET request
    level = user.get('level', 1) if user else 1
    if 'pattern' not in session or 'correct_answer' not in session:
        session['pattern'], session['correct_answer'], session['explanation'], session['pattern_name'] = generate_game_pattern(level)
        session['tries'] = 0
    
    return render_template(
        'game.html',
        pattern=session['pattern'],
        error=session.pop('game_error', None),
        success=session.pop('game_success', None),
        user=user
    )


# ─────────────────────────────────────────────────────────────────────────────
# Game Pattern Generation
# ─────────────────────────────────────────────────────────────────────────────

GAME_PATTERNS = [
    (lambda x: x, "Linear"),
    (lambda x: x**2, "Square"),
    (lambda x: x**3, "Cube"),
    (lambda x: 2**x, "Exponential"),
    (lambda x: sqrt(x), "Square Root"),
    (lambda x: x**2 + x, "Quadratic"),
    (lambda x: x**3 - x**2, "Cubic"),
    (lambda n: fib(n), "Fibonacci"),
]

PATTERN_EXPLANATIONS = {
    "Linear": "Linear pattern: Each number increases by 1",
    "Square": "Square numbers: Each number is n²",
    "Cube": "Cube numbers: Each number is n³",
    "Exponential": "Powers of 2: Each number is 2ⁿ",
    "Square Root": "Square root pattern: Each number is √n",
    "Quadratic": "Quadratic pattern: Each number is n² + n",
    "Cubic": "Cubic pattern: Each number is n³ - n²",
    "Fibonacci": "Fibonacci sequence: Each number is the sum of the two preceding numbers",
}


def fib(n):
    """Compute the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def generate_game_pattern(level):
    """Generate a random pattern for the game at the given difficulty level."""
    func, name = GAME_PATTERNS[min(level - 1, len(GAME_PATTERNS) - 1)]
    length = np.random.randint(4, 7)
    start = np.random.randint(1, 11)
    ascending = np.random.choice([True, False])
    
    if ascending:
        seq = [format_num(float(func(i))) for i in range(start, start + length)]
        answer = format_num(float(func(start + length)))
    else:
        seq = [format_num(float(func(i))) for i in range(start + length - 1, start - 1, -1)]
        answer = format_num(float(func(start - 1)))
    
    # Avoid negative/zero answers
    if answer <= 0:
        return generate_game_pattern(level)
    
    explanation = PATTERN_EXPLANATIONS.get(name, "Complex pattern: Try to spot the mathematical relationship")
    return seq, answer, explanation, name

@app.route('/solver', methods=['GET', 'POST'])
def solver():
    user = get_current_user()
    error = None
    pattern_type = None
    next_terms = None
    original = None
    pattern_input = ""
    
    if request.method == 'POST':
        pattern_input = request.form.get('pattern', '')
        try:
            nums = parse_sequence(pattern_input)
            if len(nums) < 2:
                raise ValueError("Please enter at least two numbers.")
            
            pattern_type, next_terms = identify_pattern_type(nums)
            original = nums
            
            # Save to history
            patterns.insert_one({
                'username': user['username'] if user else "not logged in",
                'sequence': pattern_input,
                'solution': f"Type: {pattern_type}, Next terms: {next_terms}",
                'timestamp': datetime.datetime.now()
            })
        except Exception as ex:
            error = f"Error: {str(ex)}"
    else:
        pattern_input = request.args.get('pattern', '')
    
    return render_template(
        'solver.html',
        error=error,
        pattern_type=pattern_type,
        next_terms=next_terms,
        original_pattern=original,
        user=user,
        pattern_input=pattern_input
    )


@app.route('/about')
def about():
    return render_template('about.html', user=get_current_user())


@app.route('/pattern_library')
def pattern_library():
    return render_template('pattern_library.html', user=get_current_user())


# ─────────────────────────────────────────────────────────────────────────────
# Blog
# ─────────────────────────────────────────────────────────────────────────────

BLOG_POSTS = {
    1: {"title": "The Beauty of Fibonacci Sequence", "date": "2025-02-12", "summary": "Discover the fascinating properties and applications of the Fibonacci sequence in nature and art.", "template": "fibonacci.html"},
    2: {"title": "Unlocking the Secrets of Prime Numbers", "date": "2025-02-12", "summary": "Learn about the mysteries of prime numbers and their role in cryptography and computer science.", "template": "prime_numbers.html"},
    3: {"title": "The Power of Geometric Patterns", "date": "2025-02-12", "summary": "Explore the applications of geometric patterns in architecture, design, and mathematics.", "template": "geometric_patterns.html"},
    4: {"title": "How to Identify Arithmetic Sequences", "date": "2025-02-12", "summary": "A guide to identifying and working with arithmetic sequences, with real-world examples.", "template": "arithmetic_sequences.html"},
    5: {"title": "Polynomial Patterns in Data Analysis", "date": "2025-02-12", "summary": "How polynomial functions can be used to model and analyse data patterns.", "template": "polynomial_patterns.html"},
    6: {"title": "The Mathematics of Music: Unveiling Hidden Patterns", "date": "2025-02-12", "summary": "Explore the connections between music theory and mathematical principles.", "template": "mathematics_of_music.html"},
    7: {"title": "The Art of Problem Solving: Strategies and Techniques", "date": "2025-02-12", "summary": "Discover effective strategies and techniques for tackling complex problems in various fields.", "template": "art_of_problem_solving.html"},
    8: {"title": "Mathematics in Nature: Beyond Fibonacci", "date": "2025-02-12", "summary": "Explore the mathematical principles that govern the natural world beyond just the Fibonacci sequence.", "template": "mathematics_in_nature.html"},
}


@app.route('/blog')
def blog():
    # Build summary dict for the template
    posts = {k: {"title": v["title"], "date": v["date"], "summary": v["summary"]} for k, v in BLOG_POSTS.items()}
    return render_template('blog.html', user=get_current_user(), blog_posts=posts)

@app.route('/blog/<int:post_id>')
def blog_post(post_id):
    post = BLOG_POSTS.get(post_id)
    if not post:
        return "Blog post not found", 404
    
    content = render_template(f'blog_content/{post["template"]}')
    post_comments = list(comments.find({'post_id': post_id}).sort('timestamp', -1))
    
    return render_template(
        'blog_post.html',
        title=post["title"],
        date=post["date"],
        content=content,
        user=get_current_user(),
        comments=post_comments,
        post_id=post_id
    )


@app.route('/add_comment', methods=['POST'])
def add_comment():
    if 'user_id' not in session:
        flash('You must be logged in to comment.')
        return redirect(url_for('login'))
    
    post_id = int(request.form.get('post_id'))
    text = request.form.get('comment')
    
    if not text:
        flash('Comment cannot be empty.')
        return redirect(url_for('blog_post', post_id=post_id))
    
    user = get_current_user()
    comments.insert_one({
        'post_id': post_id,
        'username': user['username'],
        'text': text,
        'timestamp': datetime.datetime.now()
    })
    
    flash('Comment added successfully!')
    return redirect(url_for('blog_post', post_id=post_id))


# ─────────────────────────────────────────────────────────────────────────────
# Static Files & Verification
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/google925b0ffc084ab8b1.html')
def serve_google_verification():
    return send_from_directory(app.static_folder, 'google925b0ffc084ab8b1.html')


@app.route('/471cae572a7e45cfbcf4e59b54108d04.txt')
def serve_txt_verification():
    return send_from_directory(app.static_folder, '471cae572a7e45cfbcf4e59b54108d04.txt')


@app.route('/sitemap.xml')
def serve_sitemap():
    return send_from_directory(app.static_folder, 'sitemap.xml')


@app.route('/ads.txt')
def serve_ads():
    return send_from_directory(app.static_folder, 'Ads.txt')


@app.route('/robots.txt')
def serve_robots():
    return send_from_directory(app.static_folder, 'robots.txt')


@app.route('/.well-known/pki-validation/7D3D3BA0B414B564BF03E299F5FCB2D3.txt')
def serve_pki_validation():
    return send_from_directory(app.static_folder, '7D3D3BA0B414B564BF03E299F5FCB2D3.txt')


@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy-policy.html', user=get_current_user())


@app.route('/cookie-consent')
def cookie_consent():
    return render_template('cookie_consent.html', user=get_current_user())


@app.route('/ezoic_scripts')
def ezoic_scripts():
    return render_template('ezoic_scripts.html', user=get_current_user())


@app.route('/leaderboard')
def leaderboard():
    top_scores = list(users.find().sort('score', -1).limit(10))
    return render_template('leaderboard.html', top_scores=top_scores, user=get_current_user())


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    user = get_current_user()
    
    if request.method == 'POST':
        text = request.form.get('feedback')
        if not text:
            flash('Feedback cannot be empty.')
            return render_template('feedback.html', user=user)
        
        db.feedback.insert_one({
            'username': user['username'],
            'feedback': text,
            'timestamp': datetime.datetime.now()
        })
        flash('Thank you for your feedback!')
        return redirect(url_for('home'))
    
    return render_template('feedback.html', user=user)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
